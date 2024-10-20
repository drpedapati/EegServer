# test_cleaning_pipeline.py

import mne
import mne_bids as mb
import autoreject
import pylossless as ll
import asrpy

from eeg_to_bids import convert_single_eeg_to_bids

import pandas as pd
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.box import ROUNDED

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
    console.print("[green]EOG channels set successfully[/green]")
    return raw

def set_eeg_average_reference(raw):
    try:
        raw.set_eeg_reference("average", projection=True)
        #raw.apply_proj()
        console.print("[green]Average reference applied and projections updated[/green]")
    except Exception as e:
        console.print(f"[red]Error applying average reference: {e}[/red]")
    return raw

def resample_data(raw, sfreq=250):
    if sfreq is not None:
        console.print(f"[green]Data resampled to {sfreq} Hz[/green]")
        return raw.resample(sfreq)
    else:
        return raw

def trim_data(raw, tmin=0, tmax=None):
    if tmax is not None:
        console.print(f"[green]Data trimmed to {tmin} to {tmax} seconds[/green]")
        return raw.crop(tmin=tmin, tmax=tmax)
    else:
        return raw

def lowpass_filter(raw, lfreq=0.1):
    if lfreq is not None:
        console.print(f"[green]Data lowpass filtered to {lfreq} Hz[/green]")
        return raw.filter(l_freq=lfreq, h_freq=None)
    else:
        return raw

def run_pipeline(raw, pipeline):
    global params
    
    if raw is None:
        console.print("[red]Error: Raw data not found to pass to pipeline.[/red]")
        return None
    
    derivatives_path = pipeline.get_derivative_path(params["bids_path"])
    derivatives_path.suffix = "eeg"
    
    json_file = params["json_file"]
    pipeline.run_with_raw(raw)
    pipeline.save(derivatives_path, overwrite=True, format="BrainVision")
    
    # -- UPDATE JSON FILE -- #
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

    return pipeline

def run_asr(raw, cutoff=20):
    if cutoff is not None:
        console.print("[bold blue]Applying Artifact Subspace Reconstruction (ASR)...[/bold blue]")
        import asrpy
        raw.load_data()
        asr = asrpy.ASR(sfreq=raw.info["sfreq"], cutoff=cutoff)
        asr.fit(raw)
        raw = asr.transform(raw)
        console.print("[green]ASR applied successfully[/green]")
    else:
        console.print("[bold blue]ASR not applied[/bold blue]")
    return raw

def sanitize_task(task):
    sanitized_task = task.replace('-', '').replace('_', '').replace('/', '')
    console.print(f"[green]Task name sanitized: {task} to {sanitized_task}[/green]")
    return sanitized_task


def save_bids():
    
    global params
    
    fif_file = params["fif_file"]
    bids_dir = params["bids_dir"]
    json_file = params["json_file"]
    task = params["task"]
    events = params["events"]
    event_id = params["event_id"]
    event_dict = params["event_dict"]
    sanitized_task = sanitize_task(task)

    if not fif_file.exists():
        console.print(f"[red]FIF file not found: {fif_file}[/red]")
        return None
    try:
        bids_path = convert_single_eeg_to_bids(
            file_path=str(fif_file),
            output_dir=str(bids_dir),
            task=sanitized_task,
            participant_id=None,  # Participant ID will be derived from filename
            line_freq=60.0,
            overwrite=True,
            study_name=fif_file.stem,
            events=events,
            event_id=event_dict
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

def import_unprocessed():
    
    global params
    
    if not Path(params["unprocessed_file"]).exists():
        console.print(f"[red]File not found: {params["unprocessed_file"]}[/red]")
        return None, None, None

    unprocessed_file = Path(params["unprocessed_file"])
    fif_file = params["metadata_dir"] / f"{unprocessed_file.stem}_raw.fif"
    json_file = params["metadata_dir"] / f"{unprocessed_file.stem}.json"
    set_file = params["debug_dir"] / f"{unprocessed_file.stem}_import.set"
    
    params["json_file"] = json_file
    params["fif_file"] = fif_file
    params["set_file"] = set_file
    
    try:
        if params["eeg_system"] == "EGI128_RAW":
            montage_tag = "GSN-HydroCel-129"
            
            raw = mne.io.read_raw_egi(input_fname=unprocessed_file, preload=True, events_as_annotations=True)
            events, event_id = update_events(raw)
            
            params['event_id'] = {'DI66': 1}
            target_event_id = params['event_id'] 
            rev_target_event_id = dict(map(reversed, target_event_id.items()))

            annotations = mne.annotations_from_events(events, raw.info['sfreq'], event_desc=rev_target_event_id)
            raw.set_annotations(None)
            raw.set_annotations(annotations)
            
            # Print STIM channels using rich
            #stim_channels = mne.pick_types(raw.info, stim=True, exclude=[])
            #stim_channel_names = [raw.ch_names[ch] for ch in stim_channels]
            #console.print("[bold]STIM channels:[/bold]", stim_channel_names)
            
            # Convert the event_id dictionary to use standard Python strings as keys
            event_dict = {str(key): int(value) for key, value in event_id.items()}

            # Print the resulting event dictionary
            print("Event Dictionary:")
            for key, value in event_dict.items():
                print(f"{key}: {value}")
            

            params["event_dict"] = event_dict
            #params["event_id"] = target_event_id
            params["events"] = events
 
            montage = mne.channels.make_standard_montage(montage_tag)
            montage.ch_names[128] = "E129"
            raw.set_montage(montage, match_case=False)
            raw.pick_types(eeg=True, exclude=[])
            raw.save(fif_file, overwrite=True)
            console.print(f"[green]FIF file saved: {fif_file}[/green]")
            
            if DEBUG_MODE:
                debug_banner("Exporting raw EEG data to SET format")
                export_mne_raw(raw, str(set_file))


    except Exception as e:
        console.print(f"[red]Error converting unprocessed file to FIF set: {e}[/red]")
        return None

    try:
        json_file = params["json_file"]
        json_data = {
            "S01_IMPORT": {
                "CreationDateTime": datetime.datetime.now().isoformat(),
                "Task": params["task"],
                "EegSystem": params["eeg_system"],
                "Montage": montage_tag,
                "SampleRate": raw.info["sfreq"],
                "ChannelCount": len(raw.ch_names),
                "DurationSec": int(raw.n_times) / raw.info["sfreq"],
                "n_samples": int(raw.n_times),
                "n_channels": len(raw.ch_names),
            }
        }
        update_json_file(json_file, json_data)
    except Exception as e:
        console.print(f"[red]Error creating JSON file: {e}[/red]")
        return None

    if fif_file.exists():
        return fif_file, json_file, events
    else:
        return None, None, None

def update_json_file(json_file, json_data):
    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=4)
        console.print(f"[green]JSON file updated: {json_file}[/green]")

def initial_banner(unprocessed_file, eeg_system, task, config_file):
    console.print(f"[green]Starting autoclean for {task}[/green]")
    table = Table(
        title="Autoclean Configuration",
        show_header=True,
        header_style="bold green",
        box=ROUNDED,
        show_lines=True,
        expand=True,
    )
    table.add_column("Parameter", style="cyan", justify="right")
    table.add_column("Value", style="magenta")
    table.add_row("Unprocessed file", str(unprocessed_file))
    table.add_row("EEG system", eeg_system)
    table.add_row("Task", task)
    table.add_row("Config file", str(config_file))
    console.print(
    Panel(
        table,
        title="Autoclean Configuration",
        border_style="green",
    )
    )

def directories_banner(bids_dir, metadata_dir, clean_dir, debug_dir):
    table = Table(
        title="Autoclean Directories",
        show_header=True,
        header_style="bold green",
        box=ROUNDED,
        show_lines=True,
        expand=True,
    )
    table.add_column("Parameter", style="cyan", justify="right")
    table.add_column("Value", style="magenta")
    table.add_row("Bids directory", str(bids_dir))
    table.add_row("Metadata directory", str(metadata_dir))
    table.add_row("Clean directory", str(clean_dir))
    table.add_row("Debug directory", str(debug_dir))
    console.print(Panel(table, title="Autoclean Directories", border_style="green"))

params = {}
    # -- EPOCH DATA -- #

def update_events(raw):
    global params
    #event_id = params['event_id']
    events, event_id = mne.events_from_annotations(raw, event_id=params['event_id'])
    return events, event_id

def epoch_data(raw):
    global params
    events, event_id = update_events(raw)
    import numpy as np
    
    def modify_filename_ending(file_path, old_ending, new_ending):
        path = Path(file_path)
        new_name = path.name.replace(old_ending, new_ending)
        return path.with_name(new_name)
    
    clean_raw_set = modify_filename_ending(params['set_file'], '_import.set', '_clean_raw.set')

    print(f"Original file: {params['set_file']}")
    print(f"Modified file: {clean_raw_set}")
    
    export_mne_raw(raw, str(clean_raw_set))

    # Assuming you have your annotations object
    annotations = raw.annotations

    # Get unique descriptions
    unique_descriptions = np.unique(annotations.description)
    
    # Create a new event_dict
    event_dict = {str(desc): i+1 for i, desc in enumerate(unique_descriptions)}

    print("New event_dict:")
    for key, value in event_dict.items():
        print(f"{key}: {value}")

    # If you need to create events from these annotations
    events, event_id = mne.events_from_annotations(raw, event_id=event_dict)
    
    #breakpoint()
    epochs = mne.Epochs(raw, events=events, event_id={'DI66': event_id['DI66']}, tmin=-.5, tmax=2.7, baseline=None, reject_by_annotation=True, event_repeated='merge')    
    epochs.get_data().shape
    epochs.drop_log
    return epochs

def entrypoint(unprocessed_file, eeg_system, task, config_file):
    
    global params
    
    initial_banner(unprocessed_file, eeg_system, task, config_file)

    params = {
        "unprocessed_file": unprocessed_file,
        "eeg_system": eeg_system,
        "task": task,
        "config_file": config_file,
        "sanitized_task": sanitize_task(task),
        "lowpass_filter": .1,
        "resample_data": 250,
        "eog_channels": [f"E{ch}" for ch in sorted([1, 32, 8, 14, 17, 21, 25, 125, 126, 127, 128])],
        "trim_data": None,
        "asr_cutoff": 20,
        "bids_path": None,
        "json_file": None,
        'event_id': {'DI66': 1}
    }
    
    params["bids_dir"], params["metadata_dir"], params["clean_dir"], params["debug_dir"] = prepare_directories(params["task"])
    
    # -- CONVERT UNPROCESSED TO FIF SET -- #
    params["fif_file"], params["json_file"], params["events"] = import_unprocessed()
    
    # -- JSON FILE -- #
    json_file = params["json_file"]
    
    # -- SAVE BIDS -- #
    params["bids_path"] = save_bids()
    
    #return

#def overflow_preprocessing(raw):
    raw = mb.read_raw_bids(params["bids_path"], verbose="ERROR", extra_params={"preload": True})
    
    raw = lowpass_filter(raw, lfreq=params["lowpass_filter"])
    
    raw = resample_data(raw, sfreq=params["resample_data"])
    
    raw = trim_data(raw, tmin=0, tmax=params["trim_data"])
    
    raw = set_eeg_average_reference(raw)
    
    # raw = run_asr(raw, cutoff=params["asr_cutoff"])

    # update json file with new metadata
    with open(json_file, "r") as f:
        json_data = json.load(f)
    if "S02_PREPROCESS" not in json_data:
        json_data["S02_PREPROCESS"] = {}
    json_data["S02_PREPROCESS"]["ResampleHz"] = params["resample_data"]
    json_data["S02_PREPROCESS"]["TrimSec"] = params["trim_data"]
    json_data["S02_PREPROCESS"]["LowPassHz1"] = params["lowpass_filter"]
    json_data["S02_PREPROCESS"]["HighPassHz1"] = params["asr_cutoff"]
    json_data["S02_PREPROCESS"]["CropDurationSec"] = params["trim_data"]
    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=4)
        
    
    # -- RUN PIPELINE -- #
    pipeline = ll.LosslessPipeline(str(params["config_file"]))
    pipeline = run_pipeline(raw, pipeline)

    # -- APPLY REJECTION POLICY -- #
    pipeline.raw.load_data()
    clean_rejection_policy = get_cleaning_rejection_policy()
    cleaned_raw = clean_rejection_policy.apply(pipeline)
    epochs = epoch_data(cleaned_raw)
    epochs.export(params["clean_dir"] / (Path(unprocessed_file).stem + "_postcomp_epo.set"), fmt='eeglab', overwrite=True)
    
    epochs.load_data()
    epochs_ar, reject_log = run_autoreject(epochs)
    epochs_ar.export(params["clean_dir"] / (Path(unprocessed_file).stem + "_postcomp_ar_epo.set"), fmt='eeglab', overwrite=True)
    
    ar_plot_file = str(params["debug_dir"] / (Path(unprocessed_file).stem + "_autoreject_plot.png"))
    fig = reject_log.plot(show=False)
    fig.savefig(ar_plot_file)

    try:

        # get indices of bad epochs by counting True values
        bad_epoch_index = np.where(reject_log.bad_epochs)[0] if reject_log is not None and reject_log.bad_epochs is not None else None

        # Count and calculate percentages for each label type
        labels = reject_log.labels if reject_log is not None and hasattr(reject_log, 'labels') else None
        total_elements = labels.size if labels is not None else 0
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
            'ar_bad_epochs': bad_epoch_index.tolist() if bad_epoch_index is not None else None,
            'label_counts': label_counts if label_counts is not None else {},
            'label_percentages': label_percentages if label_percentages is not None else {}
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
        export_mne_epochs(epochs_ar, str(params["debug_dir"] / (Path(unprocessed_file).stem + "_postcomp_epo.set")))

    return
    

def prepare_directories(task):
    root_dir = Path(os.getenv("AUTOCLEAN_DIR"))
    bids_dir = root_dir / task / "bids"
    metadata_dir = root_dir / task / "metadata"
    clean_dir = root_dir / task / "postcomps"
    debug_dir = root_dir / task / "debug"
    
    if not bids_dir.exists():
        bids_dir.mkdir(parents=True, exist_ok=True)
    
    if not metadata_dir.exists():
        metadata_dir.mkdir(parents=True, exist_ok=True)
    
    if not clean_dir.exists():
        clean_dir.mkdir(parents=True, exist_ok=True)
    
    if not debug_dir.exists():
        debug_dir.mkdir(parents=True, exist_ok=True)
        
    directories_banner(bids_dir, metadata_dir, clean_dir, debug_dir)
        
    return bids_dir, metadata_dir, clean_dir, debug_dir
    
def main():
    
    #unprocessed_file = "/Users/ernie/Documents/GitHub/EegServer/unprocessed/0006_chirp.raw"
    eeg_system = "EGI128_RAW"
    task = "assr_default"
    config_file = Path("lossless_config_assr_default.yaml")

    # Directory containing the raw files
    import socket
    hostname = socket.gethostname()
    if hostname == "Ernies-MacBook-Pro.local" or hostname == "ew19-04419.chmccorp.cchmc.org":
        print("Running on MacBook")
        unprocessed_dir = Path("/Users/ernie/Documents/GitHub/EegServer/unprocessed")
    else:
        unprocessed_dir = Path("/home/ernie/srv/RAWDATA/1_NBRT_LAB_STUDIES/Proj_SPG601/SSCT")
    
    # List all .raw files in the directory
    raw_files = [f for f in unprocessed_dir.glob("*.raw") if f.name.lower().endswith("_ssct.raw")]

    # Loop through each raw file and process it
    for raw_file in raw_files:
        print(f"Processing file: {raw_file}")
        entrypoint(str(raw_file), eeg_system, task, config_file)

    # If no raw files are found, print a message
    if not raw_files:
        print("No .raw files found in the unprocessed directory.")

    #print("Starting autoclean for chirp_default")
    #entrypoint(unprocessed_file, eeg_system, task, config_file)

if __name__ == "__main__":
    main()
