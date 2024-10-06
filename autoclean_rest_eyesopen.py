# test_cleaning_pipeline.py

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
DEBUG_MODE = False

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

def reject_bad_segs(raw, epoch_duration=2.0, baseline=None, preload=True):
    events = mne.make_fixed_length_events(raw, duration=epoch_duration)
    epochs = mne.Epochs(
        raw, events, tmin=0, tmax=epoch_duration, baseline=baseline, preload=preload
    )

    return epochs

def run_autoreject(epochs, n_jobs=1, verbose=True):
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

def pre_pipeline_processing(raw, json_file):
    
    target_sfreq = 250
    target_crop_duration = 30
    target_lfreq = 0.1
    target_hfreq = None
    
    
    raw = resample_data(raw, sfreq=target_sfreq)

    # raw.filter(l_freq=target_lfreq, h_freq=target_hfreq)

    raw = set_eog_channels(raw)
    

    # Get the start and end times of the data
    start_time = raw.times[0]
    end_time = raw.times[-1]

    # Crop the data, trimming 2 seconds from start and end
    trim = 4
    raw = raw.crop(tmin=start_time + trim, tmax=end_time - trim)

    raw.crop(tmin=0, tmax=target_crop_duration)

    raw = set_eeg_average_reference(raw)
    
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
    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=4)

    return raw

def save_bids(fif_file, autoclean_dir, json_file, task):
    try:
        bids_dir = autoclean_dir
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

        return bids_path

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return None

def convert_unprocessed_to_fif_set(unprocessed_file, metadata_dir, eeg_system, task):

    unprocessed_file = Path(unprocessed_file)
    fif_file = metadata_dir / f"{unprocessed_file.stem}_raw.fif"
    json_file = metadata_dir / f"{unprocessed_file.stem}.json"
    print(json_file)

    try:
        if eeg_system == "EGI128_RAW":
            montage_tag = "GSN-HydroCel-129"
            raw = mne.io.read_raw_egi(input_fname=unprocessed_file, preload=False)
            montage = mne.channels.make_standard_montage(montage_tag)
            montage.ch_names[128] = "E129"
            raw.set_montage(montage, match_case=False)
            raw.pick_types(eeg=True, exclude=[])
            raw.save(fif_file, overwrite=True)
            console.print(f"[green]FIF file saved: {fif_file}[/green]")
            
            # if DEBUG_MODE:
            #     debug_banner("Exporting raw EEG data to SET format")
            #     export_mne_raw(raw, str(metadata_dir / (Path(unprocessed_file).stem + "_import.set")))


    except Exception as e:
        console.print(f"[red]Error converting unprocessed file to FIF set: {e}[/red]")
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
            console.print(f"[green]JSON file saved: {json_file}[/green]")

    except Exception as e:
        console.print(f"[red]Error creating JSON file: {e}[/red]")
        return None
    
    if fif_file.exists():
        return fif_file, json_file
    else:
        return None

def entrypoint(unprocessed_file, eeg_system, task, config_file):
    
    autoclean_dir, metadata_dir, clean_dir, debug_dir = prepare_directories(task)
    
    if Path(unprocessed_file).exists():
        fif_file, json_file = convert_unprocessed_to_fif_set(unprocessed_file, metadata_dir, eeg_system, task)
    else:
        console.print(f"[red]File not found: {unprocessed_file}[/red]")
    
    if fif_file is not None:
        bids_path = save_bids(fif_file, autoclean_dir, json_file, task="rest")
        raw = mb.read_raw_bids(bids_path, verbose="ERROR", extra_params={"preload": True})
        if DEBUG_MODE:
            debug_banner("Saving raw BIDS data")
            #raw.save(debug_dir / (Path(unprocessed_file).stem + "_raw.fif"), overwrite=True)
            export_mne_raw(raw, str(debug_dir / (Path(unprocessed_file).stem + "_bids_raw.set")))

    if raw is not None:
        raw = pre_pipeline_processing(raw, json_file)
        if DEBUG_MODE:
            debug_banner("Saving preprocessed raw data")
            #raw.save(debug_dir / (Path(unprocessed_file).stem + "_raw_preproc.fif"), overwrite=True)
            export_mne_raw(raw, str(debug_dir / (Path(unprocessed_file).stem + "_preprocess_raw.set")))

    if raw is not None:
        pipeline = ll.LosslessPipeline(str(config_file))
        derivatives_path = pipeline.get_derivative_path(bids_path)
        derivatives_path.suffix = "eeg"
        pipeline = run_pipeline(raw, pipeline, json_file)
        pipeline.save(derivatives_path, overwrite=True, format="BrainVision")
        if DEBUG_MODE:
            debug_banner("Saving pipeline processed raw data")
            #pipeline.raw.save(debug_dir / (Path(unprocessed_file).stem + "_raw_pipeline.fif"), overwrite=True)
            export_mne_raw(pipeline.raw, str(debug_dir / (Path(unprocessed_file).stem + "_raw_pipeline.set")))

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
    epochs_ar, reject_log = run_autoreject(epochs)
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
        export_mne_epochs(epochs_ar, str(debug_dir / (Path(unprocessed_file).stem + "_postcomp_epo.set")))

    return

def prepare_directories(task):
    bids_dir = Path(os.getenv("AUTOCLEAN_DIR")) / task / "bids"
    metadata_dir =Path(os.getenv("AUTOCLEAN_DIR")) / task / "metadata"
    clean_dir = Path(os.getenv("AUTOCLEAN_DIR")) / task / "postcomps"
    debug_dir = Path(os.getenv("AUTOCLEAN_DIR")) / task / "debug"
    
    if not bids_dir.exists():
        bids_dir.mkdir(parents=True, exist_ok=True)
    
    if not metadata_dir.exists():
        metadata_dir.mkdir(parents=True, exist_ok=True)
    
    if not clean_dir.exists():
        clean_dir.mkdir(parents=True, exist_ok=True)
    
    if not debug_dir.exists():
        debug_dir.mkdir(parents=True, exist_ok=True)
        
    return bids_dir, metadata_dir, clean_dir, debug_dir
    
def main():
    
    unprocessed_file = "/Users/ernie/Documents/GitHub/EegServer/unprocessed/0354_rest.raw"
    eeg_system = "EGI128_RAW"
    task = "rest_eyesopen"
    config_file = Path("lossless_config_rest_eyesopen.yaml")

    entrypoint(unprocessed_file, eeg_system, task, config_file)

if __name__ == "__main__":
    main()
