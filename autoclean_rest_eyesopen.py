# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "autoreject",
#     "mne",
#     "mne-bids",
#     "numpy",
#     "pandas",
#     "pathlib",
#     "rich",
#     "pylossless @ /media/bigdrive/data/Proj_SPG601/EegServer/pylossless",
#     "python-dotenv",
#     "openneuro-py",
#     "eeglabio",
#     "torch",
#     "pybv",
#     "pyyaml"
# ]
# ///
# test_cleaning_pipeline.py


#     "pylossless @ git+https://github.com/drpedapati/pylossless.git",
#     "pylossless @ /media/bigdrive/data/Proj_SPG601/EegServer/pylossless",

import mne
import mne_bids as mb
import autoreject
import pylossless as ll

from eeg_to_bids import convert_single_eeg_to_bids

import pandas as pd
import numpy as np
from rich.console import Console
from pathlib import Path

import json
import yaml
import os
import dotenv

import datetime
from eeglabio.utils import export_mne_epochs, export_mne_raw

console = Console()

dotenv.load_dotenv()

# Add debug mode flag
DEBUG_MODE = True

def debug_banner(message):
    if DEBUG_MODE:
        console.print(f"\n[bold yellow]{'=' * 20} DEBUG: {message} {'=' * 20}[/bold yellow]\n")

def get_cleaning_rejection_policy():
    rejection_policy = ll.RejectionPolicy()
    rejection_policy["ch_flags_to_reject"] = ["noisy", "uncorrelated", "bridged"]
    rejection_policy["ch_cleaning_mode"] = "interpolate"
    rejection_policy["interpolate_bads_kwargs"] = {"method": "MNE"}
    rejection_policy["ic_flags_to_reject"] = [
        "muscle",
        "heart",
        "eye",
        "channel noise",
        "line noise",
    ]
    rejection_policy["ic_rejection_threshold"] = 0.3  # More aggressive threshold
    rejection_policy["remove_flagged_ics"] = True

    return rejection_policy

def reject_bad_segs(raw, epoch_duration=2.0, baseline=None, preload=True): #GW note: is this used later? 
    events = mne.make_fixed_length_events(raw, duration=epoch_duration)
    epochs = mne.Epochs(
        raw, events, tmin=0, tmax=epoch_duration, baseline=baseline, preload=preload
    )

    return epochs

def run_autoreject(epochs, n_jobs=1, verbose=True):  #GW note: flagged epochs not rejected?  https://github.com/drpedapati/pylossless/blob/main/pylossless/config/rejection.py (line 136)
    ar = autoreject.AutoReject(random_state=11, n_jobs=n_jobs, verbose=verbose)
    ar.fit(epochs)
    epochs_ar, reject_log = ar.transform(epochs, return_log=True)
    return epochs_ar, reject_log

def set_eog_channels(raw, eog_channels=None):
    # Set channel types for EGI 128 channel Net
    if eog_channels is None:
        eog_channels = [
            f"E{ch}" for ch in sorted([1, 32, 8, 14, 17, 21, 25, 125, 126, 127, 128])
        ]
    raw.set_channel_types({ch: "eog" for ch in raw.ch_names if ch in eog_channels})
    return raw

def set_eeg_average_reference(raw):
    return raw.set_eeg_reference("average", projection=True)

def resample_data(raw, sfreq=250):
    return raw.resample(sfreq)

def run_pipeline(raw, pipeline, json_file):   
    pipeline.run_with_raw(raw)
    return pipeline

def pre_pipeline_processing(raw, json_file, debug_dir=None):
    
    apply_resample_toggle                   = True
    apply_eog_toggle                        = False 
    apply_average_reference_toggle          = False
    apply_trim_toggle                       = True
    apply_crop_toggle                       = True    
    apply_filter_toggle                     = False  

    # Resample
    if apply_resample_toggle:
        target_sfreq        = 250
        raw                 = resample_data(raw, sfreq=target_sfreq)
    else:
        target_sfreq        = None  
    
    # EOG Assignment
    if apply_eog_toggle:
        raw                 = set_eog_channels(raw)
    else:
        raw                 = raw
    
    # Average Reference
    if apply_average_reference_toggle:
        raw                 = set_eeg_average_reference(raw)
    else:
        raw                 = raw
    
    # Trim Edges
    if apply_trim_toggle:
        trim                 = 4
        start_time           = raw.times[0]
        end_time             = raw.times[-1]
        raw                  = raw.crop(tmin=start_time + trim, tmax=end_time - trim)   
    else:
        trim                 = None
        start_time           = None
        end_time             = None
    
    # Crop Duration
    if apply_crop_toggle:     
        target_crop_duration = 60 
        start_time           = raw.times[0]
        raw                  = raw.crop(tmin=start_time, tmax=start_time + target_crop_duration)
    else:
        target_crop_duration = None

    # Pre-Filter    
    if apply_filter_toggle:
        target_lfreq         = .1           #was 0.1, GW changed for testing (ica recommends a lowpass of at least 1hz)
        target_hfreq         = None         #was None, GW changed for testing
        raw.filter(l_freq=target_lfreq, h_freq=target_hfreq)
    else:
        target_lfreq         = None
        target_hfreq         = None 

    # update json file with new metadata
    with open(json_file, "r") as f:
        json_data = json.load(f)
    if "S02_PREPROCESS" not in json_data:
        json_data["S02_PREPROCESS"] = {}
    json_data["S02_PREPROCESS"]["ResampleHz"] = target_sfreq
    json_data["S02_PREPROCESS"]["TrimSec"] = trim
    json_data["S02_PREPROCESS"]["LowPassHz1"] = target_lfreq
    json_data["S02_PREPROCESS"]["HighPassHz1"] = target_hfreq
    json_data["S02_PREPROCESS"]["CropDurationSec"] = target_crop_duration
    json_data["S02_PREPROCESS"]["AverageReference"] = apply_average_reference_toggle

    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=4)
    
    if DEBUG_MODE and debug_dir is not None:
        debug_banner("Saving preprocessed raw data")
        export_mne_raw(raw, str(debug_dir / (Path(raw.filenames[0]).stem + "_prepipeline_raw.set")))

    return raw

def save_bids(fif_file, bids_dir, json_file, task, debug_dir=None):
    
    if fif_file is None:
        console.print("[red]Error: FIF file not found[/red]")
        return None
    
    try:
        bids_path = convert_single_eeg_to_bids(
            file_path=str(fif_file),
            output_dir=str(bids_dir),
            task=task,
            participant_id=None,  # Participant ID will be derived from filename
            line_freq=60.0,
            overwrite=True,
            study_name=fif_file.stem,
        )
        console.print("[green]BIDS conversion successful.[/green]")
        console.print(f"BIDS data saved at: {bids_path}")
        
        # delete fif file
        fif_file.unlink()
        
        # update json file with bids path
        with open(json_file, "r") as f:
            json_data = json.load(f)
        json_data["S01_IMPORT"]["BIDSPath"] = str(bids_path)
        json_data["S01_IMPORT"]["BIDSRoot"] = str(bids_path.root)
        json_data["S01_IMPORT"]["BIDSBasename"] = bids_path.basename
        with open(json_file, "w") as f:
            json.dump(json_data, f, indent=4)
            
        if DEBUG_MODE and debug_dir is not None:
            raw = mb.read_raw_bids(bids_path, verbose="ERROR", extra_params={"preload": True})
            debug_banner("Saving BIDS raw data")
            export_mne_raw(raw, str(debug_dir / (Path(fif_file).stem + "_bids_raw.set")))

        return bids_path

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return None
    
def convert_unprocessed_to_fif_set(unprocessed_file, metadata_dir, eeg_system, task, debug_dir=None):
    
    if not Path(unprocessed_file).exists():
        console.print("[red]Error: Input file not found[/red]")
        return None

    unprocessed_file = Path(unprocessed_file)
    fif_file = metadata_dir / f"{unprocessed_file.stem}_raw.fif"
    json_file = metadata_dir / f"{unprocessed_file.stem}.json"
    console.print(f"Processing metadata file: {json_file}")

    try:
        if eeg_system == "EGI128_RAW":
            montage_tag = "GSN-HydroCel-129"
            raw = mne.io.read_raw_egi(input_fname=unprocessed_file, preload=False)
            montage = mne.channels.make_standard_montage(montage_tag)
            montage.ch_names[128] = "E129"
            raw.set_montage(montage, match_case=False)
            raw.pick_types(eeg=True, exclude=[])
            raw.save(fif_file, overwrite=True)
            console.print("[green]Successfully converted EGI data to FIF format[/green]")
            
            if DEBUG_MODE and debug_dir is not None:
                debug_banner("Exporting raw EEG data to SET format")
                export_mne_raw(raw, str(debug_dir / (Path(unprocessed_file).stem + "_import.set")))

    except Exception as e:
        console.print(f"[red]Error: Failed to convert EGI data to FIF format - {str(e)}[/red]")
        return None

    try:
        json_data = {
            "S01_IMPORT": {
                "CreationDateTime": datetime.datetime.now().isoformat(),
                "Task": task,
                "EegSystem": eeg_system,
                "Montage": montage_tag,
                "SampleRate": raw.info["sfreq"],
                "ChannelCount": len(raw.ch_names),
                "DurationSec": int(raw.n_times) / raw.info["sfreq"],
                "n_samples": int(raw.n_times),
                "n_channels": len(raw.ch_names),
            }
        }

        with open(json_file, "w") as f:
            json.dump(json_data, f, indent=4)
            console.print("[green]Successfully created metadata JSON file[/green]")

    except Exception as e:
        console.print(f"[red]Error: Failed to create metadata JSON file - {str(e)}[/red]")
        return None
    
    if fif_file.exists():
        return fif_file, json_file
    else:
        return None

def entrypoint(unprocessed_file, eeg_system, task, config_file):
    
    autoclean_dir, bids_dir,  metadata_dir, clean_dir, debug_dir = prepare_directories(task)
    
    fif_file, json_file = convert_unprocessed_to_fif_set(unprocessed_file, metadata_dir, eeg_system, task, debug_dir)
    
    bids_path           = save_bids(fif_file, bids_dir, json_file, task="rest", debug_dir=debug_dir)
    
    raw                 = mb.read_raw_bids(bids_path, verbose="ERROR", extra_params={"preload": True})
    
    raw                 = pre_pipeline_processing(raw, json_file, debug_dir)



    pipeline                    = ll.LosslessPipeline(str(config_file))
    derivatives_path            = pipeline.get_derivative_path(bids_path)
    derivatives_path.suffix     = "eeg"
    pipeline                    = run_pipeline(raw, pipeline, json_file)

    breakpoint() 
    pipeline.save(derivatives_path, overwrite=True, format="auto")
    pipeline.raw.load_data()
    
    clean_rejection_policy = get_cleaning_rejection_policy()
    cleaned_raw = clean_rejection_policy.apply(pipeline)
    
    pipeline.save(derivatives_path, overwrite=True, format="auto") #GW note to self: preprocessed output directory

    export_mne_raw(pipeline.raw, str(debug_dir / (Path(unprocessed_file).stem + "_raw_cleaned_pipeline.set")))
    export_mne_raw(cleaned_raw, str(debug_dir / (Path(unprocessed_file).stem + "_postcomp_raw.set")))
    ica_path = derivatives_path.directory / "sub-400257_task-rest_ica2_ica.fif"
    ica = mne.preprocessing.read_ica(ica_path)


    # Read the IC labels TSV file
    ic_labels_path = derivatives_path.directory / "sub-400257_task-rest_iclabels.tsv"
    ic_labels_df = pd.read_csv(ic_labels_path, sep='\t')

    mask = (ic_labels_df['confidence'] > 0.3) & (~ic_labels_df['ic_type'].isin(['brain', 'other']))
    high_conf_idx = ic_labels_df[mask].index.tolist()
    console.print(f"Components with confidence > 0.3 and not brain/other: {high_conf_idx}")

    reconst_raw = ica.apply(cleaned_raw, exclude=high_conf_idx)
    export_mne_raw(reconst_raw, str(debug_dir / (Path(unprocessed_file).stem + "_reconstructed_raw.set")))
    
    # Load and apply ICA to raw file
    raw = mb.read_raw_bids(bids_path, verbose="ERROR", extra_params={"preload": True})
    
    

    if DEBUG_MODE:
        debug_banner("Saving pipeline processed raw data")
        #pipeline.raw.save(debug_dir / (Path(unprocessed_file).stem + "_raw_pipeline.fif"), overwrite=True)
        export_mne_raw(pipeline.raw, str(debug_dir / (Path(unprocessed_file).stem + "_raw_pipeline.set")))

    
    if raw is not None:
        
        derivatives_path = pipeline.get_derivative_path(bids_path)
        derivatives_path.suffix = "eeg"
        pipeline = run_pipeline(raw, pipeline, json_file)
        pipeline.save(derivatives_path, overwrite=True, format="EEGLAB") #GW note to self: preprocessed output directory

    # Update JSON file with postcomps stage information
    try:
        with open(json_file, "r") as f:
            json_data = json.load(f)
        # Add basic information about the raw data
        json_data["S04_POSTCOMPS"] = {
            "raw_info": {
                "sample_rate": pipeline.raw.info['sfreq'],
                "duration_seconds": pipeline.raw.n_times / pipeline.raw.info['sfreq'],
                "n_samples": int(pipeline.raw.n_times),
                "n_channels": pipeline.raw.info['nchan'],
                "highpass_filter": pipeline.raw.info['highpass'],
                "lowpass_filter": pipeline.raw.info['lowpass'],
            },
            "bad_channels": {
                "noisy_channels": pipeline.flags['ch']['noisy'].tolist() if isinstance(pipeline.flags['ch']['noisy'], np.ndarray) else list(pipeline.flags['ch']['noisy']),
                "bridged_channels": pipeline.flags['ch']['bridged'].tolist() if isinstance(pipeline.flags['ch']['bridged'], np.ndarray) else list(pipeline.flags['ch']['bridged']),
                "rank_deficient_channels": pipeline.flags['ch']['rank'].tolist() if isinstance(pipeline.flags['ch']['rank'], np.ndarray) else list(pipeline.flags['ch']['rank']),
                "uncorrelated_channels": pipeline.flags['ch']['uncorrelated'].tolist() if isinstance(pipeline.flags['ch']['uncorrelated'], np.ndarray) else list(pipeline.flags['ch']['uncorrelated'])
            },
            "bad_epochs": {
                "noisy_epochs": pipeline.flags['epoch']['noisy'].tolist() if isinstance(pipeline.flags['epoch']['noisy'], np.ndarray) else list(pipeline.flags['epoch']['noisy']),
                "uncorrelated_epochs": pipeline.flags['epoch']['uncorrelated'].tolist() if isinstance(pipeline.flags['epoch']['uncorrelated'], np.ndarray) else list(pipeline.flags['epoch']['uncorrelated']),
                "noisy_ICs": pipeline.flags['epoch']['noisy_ICs'].tolist() if isinstance(pipeline.flags['epoch']['noisy_ICs'], np.ndarray) else list(pipeline.flags['epoch']['noisy_ICs'])
            },
            "bad_ics": {
                "brain_ICs": pipeline.ica2.labels_['brain'],
                "muscle_ICs": pipeline.ica2.labels_['muscle'],
                "heart_ICs": pipeline.ica2.labels_['ecg'],
                "eye_ICs": pipeline.ica2.labels_['eog'],
                "channel_noise_ICs": pipeline.ica2.labels_['ch_noise'],
                "line_noise_ICs": pipeline.ica2.labels_['line_noise'],
                "other_ICs": pipeline.ica2.labels_['other']
            }
        }
        
        # Add events section
        annotations_list = []
        for ann in pipeline.raw.annotations:
            annotation_dict = {
                "onset": float(ann['onset']),
                "duration": float(ann['duration']),
                "description": str(ann['description']),
                "orig_time": ann['orig_time'].isoformat() if ann['orig_time'] else None
            }
            # Convert numpy int64 to regular Python int
            for key, value in annotation_dict.items():
                if isinstance(value, np.int64):
                    annotation_dict[key] = int(value)
            annotations_list.append(annotation_dict)
        
        json_data["S04_POSTCOMPS"]["events"] = annotations_list
        
        with open(json_file, "w") as f:
            json.dump(json_data, f, indent=4)
        console.print(f"[green]JSON file updated with postcomps information and events: {json_file}[/green]")
    except Exception as e:
        console.print(f"[red]Error updating JSON file with postcomps information and events: {e}[/red]")

    # Apply rejection policy
    pipeline.raw.load_data()
    clean_rejection_policy = get_cleaning_rejection_policy()
    cleaned_raw = clean_rejection_policy.apply(pipeline)
    if DEBUG_MODE:
        debug_banner("Saving cleaned raw data")
        #cleaned_raw.save(debug_dir / (Path(unprocessed_file).stem + "_postcomp_raw.fif"), overwrite=True)
        export_mne_raw(cleaned_raw, str(debug_dir / (Path(unprocessed_file).stem + "_postcomp_raw.set")))

    epochs = mne.make_fixed_length_epochs(cleaned_raw, duration=2)
    epochs.load_data()
    epochs_ar, reject_log = run_autoreject(epochs)              #GW note: epoch rejection should be happening here
    epochs_ar.export(clean_dir / (Path(unprocessed_file).stem + "_postcomp_epo.set"), fmt='eeglab', overwrite=True)
    
    ar_plot_file = str(debug_dir / (Path(unprocessed_file).stem + "_autoreject_plot.png"))
    fig = reject_log.plot(show=False)
    fig.savefig(ar_plot_file)

    try:
        # get indices of bad epochs by counting True values
        bad_epoch_index = np.where(reject_log.bad_epochs)[0]

        # Count and calculate percentages for each label type
        labels = reject_log.labels
        total_elements = labels.size
        label_counts = {
            "good_data": int(np.sum(labels == 0)),
            "bad_data": int(np.sum(labels == 1)),
            "interpolated": int(np.sum(labels == 2))
        }
        label_percentages = {
            label: float(count / total_elements * 100)
            for label, count in label_counts.items()
        }

        # Add autoreject log to json
        reject_log_dict = {
            'ar_bad_epochs': bad_epoch_index.tolist(),
            'label_counts': label_counts,
            'label_percentages': label_percentages
        }
        # with autoreject log
        json_data["S04_POSTCOMPS"]["autoreject_log"] = reject_log_dict
        
        # Write updated json_data back to file
        with open(json_file, "w") as f:
            json.dump(json_data, f, indent=4)
    except Exception as e:
        console.print(f"[red]Error adding autoreject log to JSON: {e}[/red]")
    if DEBUG_MODE:
        debug_banner("Saving autoreject epochs")
        #epochs_ar.save(debug_dir / (Path(unprocessed_file).stem + "_postcomp_epo.fif"), overwrite=True)
        export_mne_epochs(epochs_ar, str(debug_dir / (Path(unprocessed_file).stem + "_postcomp_epo.set")))  #GW note- is this saving epochs that should be rejected as the postcomp_epo.set?

    return

def prepare_directories(task):
    autoclean_dir = Path(os.getenv("AUTOCLEAN_DIR"))
    bids_dir = autoclean_dir / task / "bids"
    metadata_dir = autoclean_dir / task / "metadata"
    clean_dir = autoclean_dir / task / "postcomps"
    debug_dir = autoclean_dir / task / "debug"
    
    if not bids_dir.exists():
        bids_dir.mkdir(parents=True, exist_ok=True)
    
    if not metadata_dir.exists():
        metadata_dir.mkdir(parents=True, exist_ok=True)
    
    if not clean_dir.exists():
        clean_dir.mkdir(parents=True, exist_ok=True)
    
    if not debug_dir.exists():
        debug_dir.mkdir(parents=True, exist_ok=True)
        
    return autoclean_dir, bids_dir, metadata_dir, clean_dir, debug_dir
    
def main():
    
    eeg_system = "EGI128_RAW"
    task = "rest_eyesopen"
    config_file = Path("lossless_config_rest_eyesopen.yaml")
    
    autoclean_dir, bids_dir, metadata_dir, clean_dir, debug_dir = prepare_directories(task)

    # Define Test File
    test_file = "2287_rest.raw"
    unprocessed_dir = Path(os.getenv("UNPROCESSED_DIR"))
    unprocessed_file = unprocessed_dir / test_file
    console.print(f"[bold yellow]Unprocessed file: {unprocessed_file}[/bold yellow]")

    entrypoint(unprocessed_file, eeg_system, task, config_file)

if __name__ == "__main__":
    main()
