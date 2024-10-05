# test_cleaning_pipeline.py


import mne
import mne_bids as mb
import autoreject
import pylossless as ll

from eeg_to_bids import convert_single_eeg_to_bids

import pandas as pd
from rich.console import Console
from pathlib import Path

import json
import yaml
import os
import dotenv

import datetime
from signalflow import import_egi, export_eeglab, save_fif
from eeglabio.utils import export_mne_epochs, export_mne_raw

console = Console()


dotenv.load_dotenv()


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







def run_pipeline(raw, pipeline):
    pipeline.run_with_raw(raw)
    return pipeline

def create_report(bids_path, output_dir):
    """
    Merges specific metadata files into a single JSON report.

    Parameters:
    - bids_path (BIDSPath): MNE-BIDS BIDSPath object pointing to the EEG file.
    - output_dir (Path): Path object where the merged report will be saved.
    """
    try:
        # Initialize an empty dictionary to hold all metadata
        merged_metadata = {}

        # Define the main JSON file path
        main_json_path = bids_path.fpath.with_suffix(".json")

        # List of additional metadata files with their suffixes and types
        metadata_files = {
            "iclabels": {"suffix": "_iclabels.tsv", "type": "tsv"},
            "ll_config": {"suffix": "_ll_config.yaml", "type": "yaml"},
            "ll_FlaggedChs": {"suffix": "_ll_FlaggedChs.tsv", "type": "tsv"},
            "events": {"suffix": "_events.tsv", "type": "tsv"},
        }

        # Load the main JSON file
        if main_json_path.exists():
            try:
                with open(main_json_path, "r") as f:
                    main_json = json.load(f)
                merged_metadata["main_json"] = main_json
                console.print(
                    f"[green]Loaded main JSON file: {main_json_path.name}[/green]"
                )
            except json.JSONDecodeError as jde:
                console.print(
                    f"[red]JSON decode error in file {main_json_path.name}: {jde}[/red]"
                )
            except Exception as e:
                console.print(
                    f"[red]Error loading main JSON file {main_json_path.name}: {e}[/red]"
                )
        else:
            console.print(
                f"[yellow]Main JSON file not found: {main_json_path}[/yellow]"
            )

        # Iterate through additional metadata files
        for key, file_info in metadata_files.items():
            file_suffix = file_info["suffix"]
            file_type = file_info["type"]
            metadata_path = bids_path.fpath.with_suffix("").with_suffix(file_suffix)
            # Explanation: bids_path.fpath.with_suffix('') removes existing suffix, then adds new suffix

            if metadata_path.exists():
                if file_type == "tsv":
                    try:
                        df = pd.read_csv(metadata_path, sep="\t")
                        # Convert DataFrame to dictionary
                        if df.shape[0] > 1:
                            data = df.to_dict(orient="records")
                        else:
                            data = df.to_dict(orient="records")[0]
                        merged_metadata[key] = data
                        console.print(
                            f"[green]Loaded TSV file: {metadata_path.name}[/green]"
                        )
                    except pd.errors.ParserError as pe:
                        console.print(
                            f"[red]Parser error in TSV file {metadata_path.name}: {pe}[/red]"
                        )
                    except Exception as e:
                        console.print(
                            f"[red]Error loading TSV file {metadata_path.name}: {e}[/red]"
                        )
                elif file_type == "yaml":
                    try:
                        with open(metadata_path, "r") as f:
                            yaml_data = yaml.safe_load(f)
                        merged_metadata[key] = yaml_data
                        console.print(
                            f"[green]Loaded YAML file: {metadata_path.name}[/green]"
                        )
                    except yaml.YAMLError as ye:
                        console.print(
                            f"[red]YAML error in file {metadata_path.name}: {ye}[/red]"
                        )
                    except Exception as e:
                        console.print(
                            f"[red]Error loading YAML file {metadata_path.name}: {e}[/red]"
                        )
            else:
                console.print(
                    f"[yellow]{file_info['type'].upper()} file not found: {metadata_path}[/yellow]"
                )

        # Define the report filename based on the original EEG file
        report_filename = bids_path.stem + "_merged_metadata.json"
        report_path = Path(output_dir) / report_filename

        # Ensure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the merged metadata as a single JSON file
        try:
            with open(report_path, "w") as f:
                json.dump(merged_metadata, f, indent=4)
            console.print(f"[green]Merged metadata saved to {report_path}[/green]")
        except Exception as e:
            console.print(
                f"[red]Error saving merged metadata to {report_path}: {e}[/red]"
            )

    except Exception as e:
        console.print(f"[red]Error creating report: {e}[/red]")
    finally:
        return

def pre_pipeline_processing(raw, json_file):
    
    target_sfreq = 250
    target_crop_duration = 180
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
    json_file = fif_file.with_suffix(".json")
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
            
            export_mne_raw(raw, str(metadata_dir / (Path(unprocessed_file).stem + "_import.set")))


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
        raw.save(debug_dir / (Path(unprocessed_file).stem + "_raw.fif"), overwrite=True)
        export_mne_raw(raw, str(debug_dir / (Path(unprocessed_file).stem + "_bids_raw.set")))

    if raw is not None:
        raw = pre_pipeline_processing(raw, json_file)
        raw.save(debug_dir / (Path(unprocessed_file).stem + "_raw_preproc.fif"), overwrite=True)
        export_mne_raw(raw, str(debug_dir / (Path(unprocessed_file).stem + "_preprocess_raw.set")))

    
    if raw is not None:
        pipeline = ll.LosslessPipeline(str(config_file))
        derivatives_path = pipeline.get_derivative_path(bids_path)
        derivatives_path.suffix = "eeg"
        pipeline = run_pipeline(raw, pipeline)
        pipeline.save(derivatives_path, overwrite=True, format="BrainVision")
        pipeline.raw.save(debug_dir / (Path(unprocessed_file).stem + "_raw_pipeline.fif"), overwrite=True)


    # Apply rejection policy
    pipeline.raw.load_data()
    clean_rejection_policy = get_cleaning_rejection_policy()
    cleaned_raw = clean_rejection_policy.apply(pipeline)
    cleaned_raw.save(clean_dir / (Path(unprocessed_file).stem + "_postcomp_raw.fif"), overwrite=True)
    cleaned_raw.save(debug_dir / (Path(unprocessed_file).stem + "_postcomp_raw.fif"), overwrite=True)

    epochs = mne.make_fixed_length_epochs(cleaned_raw, duration=2)
    epochs.load_data()
    epochs_ar, reject_log = run_autoreject(epochs)
    epochs_ar.save(clean_dir / (Path(unprocessed_file).stem + "_postcomp_epo.fif"), overwrite=True)
    epochs_ar.export(clean_dir / (Path(unprocessed_file).stem + "_postcomp_epo.set"), fmt='eeglab', overwrite=True)
    epochs_ar.save(debug_dir / (Path(unprocessed_file).stem + "_postcomp_epo.fif"), overwrite=True)
    # epochs_ar.export_set(debug_dir / (Path(unprocessed_file).stem + "_postcomp_epo.set"), fmt='eeglab', overwrite=True)

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
    config_file = Path("lossless_config_imax.yaml")

    entrypoint(unprocessed_file, eeg_system, task, config_file)
    
if __name__ == "__main__":
    main()
