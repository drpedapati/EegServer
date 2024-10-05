import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
import uuid
import json
from pathlib import Path
from rich.console import Console
import dotenv
import pandas as pd
import shutil
from signalfloweeg.io import load_eeg
from eeg_to_bids import convert_eeg_to_bids

import mne_bids as mb
import autoreject
import mne
import pylossless as ll

PROJECT_DIRECTORIES = 'project_directories.json'
PYLOSSLESS_CONFIG   = 'pylossless_config.json'
ENV_FILE            = '.env'

console = Console()
dotenv.load_dotenv(ENV_FILE)

# Load project directories
def load_project_directories():
    with open('project_directories.json', 'r') as f:
        directories = json.load(f)
    
    # Resolve environment variables
    for dir_name, dir_info in directories.items():
        dir_info['path'] = os.path.expandvars(dir_info['path'])
        dir_info['path'] = Path(dir_info['path'])
    
    # Print out results in a pretty way with rich
    console.print("[bold magenta]Project Directories:[/bold magenta]")
    for dir_name, dir_info in directories.items():
        console.print(f"[bold blue]{dir_name}:[/bold blue] {dir_info['path']}")
    
    return directories

# Setup logger
def main_setup_logger(log_file, project_name=None):
    """
    Set up a logger that writes structured JSON entries to a file.

    Args:
    log_file (str): The name of the log file (default: 'progress_log.jsonl').
    project_name (str): The name of the project (optional).

    Returns:
    logging.Logger: Configured logger object.
    """
    logger = logging.getLogger(project_name if project_name else __name__)
    logger.setLevel(logging.DEBUG)

    # Create a custom JSON formatter
    class JsonFormatter(logging.Formatter):

        def format(self, record):
            import getpass
            try:
                # Base log entry with standard fields
                log_entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "logger": record.name,
                    "project": project_name,
                    "log_id": str(uuid.uuid4()),
                    "file": record.filename,
                    "line": record.lineno,
                    "function": record.funcName,
                    "user": getpass.getuser()
                }

                # Define standard LogRecord attributes to exclude
                standard_attrs = {
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                    'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
                    'created', 'msecs', 'relativeCreated', 'thread', 'threadName',
                    'processName', 'process', 'message'
                }

                # Extract extra attributes
                extra_attrs = {k: v for k, v in record.__dict__.items() if k not in standard_attrs}
                log_entry.update(extra_attrs)

                return json.dumps(log_entry)
            except Exception as e:
                # Fallback in case of formatting error
                return json.dumps({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "level": "ERROR",
                    "message": f"Failed to format log record: {e}",
                    "logger": record.name,
                    "file": record.filename,
                    "line": record.lineno,
                    "function": record.funcName
                })

    # File Handler
    file_handler = RotatingFileHandler(log_file, maxBytes=10000000, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JsonFormatter())

    # Console Handler (optional, for development)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(JsonFormatter())

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Copy metadata fro

def eeg_pipeline():
    pass

# [ProjectID]: A short identifier for your project   
project_id          =  os.getenv('PROJECT_ID')

# Project Path Dictionary (name, path, description, temporary)
project_dict        = load_project_directories()
path_project        = project_dict['project']['path']

# if path_project is not a directory, create it
if not os.path.isdir(path_project):
    os.makedirs(path_project)

# Log File
log_file = os.path.join(path_project, f'{project_id}_progress_log-00.json')
logger = main_setup_logger(log_file=log_file, project_name=project_id)

# Calculate project folder size
def main_calc_project_folder_size(directory):
    def get_size(path='.'):
        return sum(f.stat().st_size for f in Path(path).glob('**/*') if f.is_file())
    size_bytes = get_size(directory)
    size_gb = size_bytes / (1024 * 1024 * 1024)
    formatted_size = f"{size_gb:.2f} GB"
    console.print(f"[green]Project folder size: {formatted_size}[/green]")
    return formatted_size

# Copy metadata from original file to target file
def main_copy_metadata(original_file, target_file):
    console.print(f"[blue]Reading metadata from the original file: {original_file}[/blue]")
    df = pd.read_csv(original_file)
    
    console.print(f"[blue]Writing metadata to the target file: {target_file}[/blue]")
    df.to_csv(target_file, index=False)
    
    console.print(f"[green]Successfully copied metadata from {original_file} to {target_file}[/green]")


def copy_files(source_dir, target_dir):
    import shutil
    import os

    # Ensure the destination directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Copy all files and subdirectories from source to destination
    for root, dirs, files in os.walk(source_dir):
        for item in dirs + files:
            source_path = os.path.join(root, item)
            rel_path = os.path.relpath(source_path, source_dir)
            dest_path = os.path.join(target_dir, rel_path)

            if os.path.isdir(source_path):
                os.makedirs(dest_path, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                if not os.path.exists(dest_path):
                    shutil.copy2(source_path, dest_path)
                    print(f"Copied: {source_path} to {dest_path}")
                else:
                    print(f"Skipped existing file: {dest_path}")
        
    print(f"Files and subdirectories processed from {source_dir} to {target_dir}")
    
# Copy data from source to dataset
def main_copy_data(source_dir, target_dir, file_list):
    copy_files(source_dir=source_dir, target_dir=target_dir)

def create_file_list(input_dir, output_filename, valid_exts=('.set', '.fdt'), file_hash=False):
    import csv
    import os
    import hashlib
    from ulid import ULID

    file_list = []
    output_file_path = output_filename  
    with open(output_file_path, 'w', newline='') as csvfile:
        fieldnames = ['ulid', 'subfolder', 'filename', 'file_size', 'file_hash', 'full_path', 'file_type']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for root, dirs, files in os.walk(input_dir):
            for item in files:
                if item.endswith(valid_exts):  # Only process files with valid extensions
                    file_path = os.path.join(root, item)
                    rel_path = os.path.relpath(file_path, input_dir)
                    subfolder = os.path.dirname(rel_path)
                    file_size = os.path.getsize(file_path)
                    file_type = os.path.splitext(item)[1][1:]  # Get the file extension without the dot
                   
                    # Calculate file hash
                    if file_hash:
                        hasher = hashlib.sha256()
                        with open(file_path, 'rb') as f:
                            for chunk in iter(lambda: f.read(4096), b""):
                                hasher.update(chunk)
                        file_hash = hasher.hexdigest()
                    else:
                        file_hash = None

                    # Generate ULID
                    ulid = str(ULID())

                    writer.writerow({
                        'ulid': ulid,
                        'subfolder': subfolder,
                        'filename': item,
                        'file_size': file_size,
                        'file_hash': file_hash,
                        'full_path': os.path.abspath(file_path),
                        'file_type': file_type
                    })

                    file_list.append(file_path)

    print(f"File list created: {output_file_path}")
    return file_list

def copy_file_list(file_list, target_dir):
    for file in file_list:
        shutil.copy2(file, target_dir)

def run_eeg_pipeline():
    pass

def resample_data(raw, sfreq=250):
    """
    Resample the raw EEG data to the specified sampling frequency.

    Parameters:
    -----------
    raw : mne.io.Raw
        The raw EEG data.
    sfreq : float, optional
        The desired sampling frequency in Hz. Default is 250 Hz.

    Returns:
    --------
    mne.io.Raw
        The resampled raw EEG data.
    """
    return raw.resample(sfreq)

def set_eog_channels(raw):
    # Set channel types for EGI 128 channel Net
    eog_channels = [
        f"E{ch}" for ch in sorted([1, 32, 8, 14, 17, 21, 25, 125, 126, 127, 128])
    ]
    raw.set_channel_types({ch: "eog" for ch in raw.ch_names if ch in eog_channels})
    return raw

def crop_data(raw, tmin=0, tmax=None):
    if tmax is None:
        tmax = raw.times[-1]  # Use the maximum time available
    return raw.load_data().crop(tmin=tmin, tmax=tmax)

def set_eeg_reference(raw):
    return raw.set_eeg_reference("average", projection=True)

def apply_bandpass_filter(raw, l_freq=1, h_freq=40):
    return raw.filter(l_freq=l_freq, h_freq=h_freq)

def get_cleaning_rejection_policy():
    # Create a new rejection policy for cleaning channels and removing ICs
    rejection_policy = ll.RejectionPolicy()

    # Set parameters for channel rejection
    rejection_policy["ch_flags_to_reject"] = ["noisy", "uncorrelated", "bridged"]
    rejection_policy["ch_cleaning_mode"] = "interpolate"
    rejection_policy["interpolate_bads_kwargs"] = {"method": "MNE"}

    # Set parameters for IC rejection
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

def run_autoreject( epochs ):

    ar = autoreject.AutoReject(random_state=11,
                            n_jobs=20, verbose=True)
    ar.fit(epochs)  # fit on a few epochs to save time
    epochs_ar, reject_log = ar.transform(epochs, return_log=True)

    return epochs_ar, reject_log

def pre_pipeline_processing(raw):
    raw = resample_data(raw, sfreq=250)
    raw.filter(l_freq=.5)
    raw = set_eog_channels(raw)
    #raw = crop_data(raw, tmin=0, tmax=120)
    raw = set_eeg_reference(raw)
    return raw

def convert_raw_to_set_and_fif( raw_queue, path ):    
    # run just the first 2 files
    import asyncio

    async def process_file(row, path):
        fif_name = row.fif_name
        set_name = row.set_name
        raw_name = row.full_path
        raw = load_eeg(raw_name, recording_type="EGI_128_RAW")
        
        raw = pre_pipeline_processing(raw)
        
        await asyncio.to_thread(raw.save, path['fif'] / fif_name, overwrite=True)
        console.print(f"[green]✅ Saved: {fif_name}[/green]")
        await asyncio.to_thread(raw.export, path['set'] / set_name, fmt='eeglab', overwrite=True)
        console.print(f"[green]✅ Converted: {set_name}[/green]")

    async def process_all_files():
        tasks = []
        for row in raw_queue.itertuples():
            tasks.append(asyncio.create_task(process_file(row, path)))
        await asyncio.gather(*tasks)

    asyncio.run(process_all_files())


def reject_bad_segs(raw, epoch_duration=2.0, tmin=0, tmax=None, baseline=None, preload=True):
    """
    Creates epochs from the raw data and provides a simple overview of the process.

    Parameters:
    -----------
    raw : mne.io.Raw
        The raw EEG data.
    epoch_duration : float, optional
        Duration of each epoch in seconds. Default is 2.0.
    tmin : float, optional
        Start time of the epoch in seconds. Default is 0.
    tmax : float or None, optional
        End time of the epoch in seconds. If None, it will be set to epoch_duration. Default is None.
    baseline : tuple or None, optional
        The baseline interval. Default is None.
    preload : bool, optional
        If True, preload the data. Default is True.

    Returns:
    --------
    epochs : mne.Epochs
        The created epochs.
    """
    if tmax is None:
        tmax = epoch_duration

    total_duration = raw.times[-1]
    print(f"Total recording duration: {total_duration:.2f} seconds")
    print(f"Number of channels: {len(raw.ch_names)}")
    print(f"Sampling frequency: {raw.info['sfreq']} Hz")

    events = mne.make_fixed_length_events(raw, duration=epoch_duration)
    epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, baseline=baseline, preload=preload)
    print(f"Number of epochs created: {len(epochs)}")

    return epochs


def run_pylossless(path):
    bids_root = path['bids']
    participants_tsv = bids_root / "participants.tsv"
    participants_df = pd.read_csv(participants_tsv, sep="\t")
    config_path = Path(path['project'] / "lossless_config.yaml")
    
    mb.print_dir_tree(bids_root, max_depth=4)
    
    sessions = mb.get_entity_vals(bids_root, "session", ignore_sessions="on")
    print(sessions)
    datatype = "eeg"
    extensions = [".vhdr"]  # ignore .json files
    bids_paths = mb.find_matching_paths(
    bids_root, datatypes=datatype, sessions=sessions, extensions=extensions
    )
    # print(bids_paths)
    
    cycle_number = 1
    
    for bids_path in bids_paths:
        if cycle_number == 1:
            # Read raw data
            raw = mb.read_raw_bids(
                    bids_path, verbose="ERROR", extra_params={"preload": True}
                )
            # print(raw)
            # Initialize and run pipeline
            pipeline = ll.LosslessPipeline(config_path)
            pipeline.run_with_raw(raw)
            # Apply rejection policy
            clean_rejection_policy = get_cleaning_rejection_policy()
            annotated_raw = clean_rejection_policy.apply(pipeline)
            clean_epoched = reject_bad_segs(annotated_raw)
            epochs_ar, reject_log = run_autoreject(clean_epoched)
            
            # Create a new directory for cleaned files
            clean_dir = path['project'] / "dataset_clean"
            clean_dir.mkdir(exist_ok=True)
            
            # save as fif
            fif_name = bids_path.basename.replace('.vhdr', '_epo.fif')
            fif_path = clean_dir / fif_name
            epochs_ar.save(fif_path, overwrite=True)
            print(f"Saved cleaned file: {fif_path}")
            
                
            # Get derivative path
            derivatives_path = pipeline.get_derivative_path(
                bids_path, derivative_name="pylossless_v1"
            )

            # event_id = pipeline.get_all_event_ids()

            # Save processed data
            pipeline.save(
                derivatives_path, overwrite=True, format="BrainVision")

        
        #cycle_number += 1
        else:
            break
    
def main():
    
    project_dict = load_project_directories()
    
    # ===============================
    #          Project Setup
    # ===============================

    # Project Directories
    path = {}
    path['project']        = project_dict['project']['path']
    path['unprocessed']    = project_dict['unprocessed']['path']
    path['raw']            = project_dict['raw']['path']
    path['set']            = project_dict['set']['path']
    path['fif']            = project_dict['fif']['path']
    path['bids']           = project_dict['bids']['path']
    # check if path directories exist, if not create them
    for p in path.values():
        if not os.path.exists(p):
            os.makedirs(p)
    
    # Project File Lists
    filelist = {}
    filelist['unprocessed'] = path_project / f'{project_id}_filelist-00-unprocessed.csv'
    filelist['raw']         = path_project / f'{project_id}_filelist-01-raw.csv'
    filelist['raw02']       = path_project / f'{project_id}_filelist-02-raw.csv'
    filelist['set']         = path_project / f'{project_id}_filelist-02-set.csv'
    filelist['fif']         = path_project / f'{project_id}_filelist-03-fif.csv'
    filelist['bids']        = path_project / f'{project_id}_filelist-03-bids.csv'
    
    # Project Queues
    queue = {}
    queue['unprocessed'] = {}
    queue['raw']        = {}
    queue['raw02']      = {}
    queue['set']        = {}
    queue['fif']        = {}
    queue['bids']       = {}
    
    # Metadata
    metadata = {}
    metadata['unprocessed'] = os.getenv('UNPROCESSED_METADATA')
    metadata['main'] = Path(path_project) / f'{project_id}_metadata-00.csv'
    
    # Logging Header
    logger.info("Starting EEG Pipeline")
    logger.info("Project Information", extra={
        "project_id": project_id,
        "project_path": str(path_project),
        "project_folder_size": main_calc_project_folder_size(path_project),
        "path_unprocessed": str(path['unprocessed']),
    })
    
    main_copy_metadata(original_file=metadata['unprocessed'], target_file=metadata['main'])

    create_file_list(path['unprocessed'], output_filename=filelist['unprocessed'], valid_exts=('.raw'))

    # Copy unprocessed files to the raw directory
    unprocessed_files = pd.read_csv(filelist['unprocessed'])['full_path'].tolist()
    copy_file_list(unprocessed_files, path['raw'])
    
    # Create filenames for set and fif files (MNE convention)
    create_file_list(path['raw'], output_filename=filelist['raw'], valid_exts=('.raw'))
    queue['raw'] = pd.read_csv(filelist['raw'])
    queue['raw']['set_name'] = queue['raw']['filename'].str.replace('.raw', '_raw.set')
    queue['raw']['fif_name'] = queue['raw']['filename'].str.replace('.raw', '_raw.fif')
    
    # queue['raw02'] = pd.read_csv(filelist['raw02'])
    # queue['raw02']['set_name'] = queue['raw02']['filename'].str.replace('.raw', '_raw.set')
    # queue['raw02']['fif_name'] = queue['raw02']['filename'].str.replace('.raw', '_raw.fif')
    
    
    # Convert to FIF format
    #convert_raw_to_set_and_fif(queue['raw02'], path)

    create_file_list(path['set'], output_filename=filelist['set'], valid_exts=('.set'), file_hash=False)
    create_file_list(path['fif'], output_filename=filelist['fif'], valid_exts=('.fif'), file_hash=False)
    
    # Create BIDS dataset
    bids_metadata = os.getenv('BIDS_METADATA')
    #convert_eeg_to_bids(metadata_csv=bids_metadata, output_dir=path['bids'], study_name=project_id, overwrite=True)
    
    # get 
    
    run_pylossless(path)




if __name__ == "__main__":
    main()