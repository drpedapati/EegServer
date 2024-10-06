import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
import uuid
import json
from pathlib import Path
from rich.console import Console
from eeg_to_bids import convert_single_eeg_to_bids

import mne_bids as mb
from mne_bids import BIDSPath
import autoreject
import mne
import pylossless as ll

console = Console()

# Setup logger
def setup_logger(log_file):
    """
    Set up a logger that writes structured JSON entries to a file.

    Args:
    log_file (str): The name of the log file (default: 'progress_log.jsonl').

    Returns:
    logging.Logger: Configured logger object.
    """
    logger = logging.getLogger(__name__)
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

def run_autoreject(epochs):
    ar = autoreject.AutoReject(random_state=11,
                            n_jobs=1, verbose=True)
    ar.fit(epochs)  # fit on a few epochs to save time
    epochs_ar, reject_log = ar.transform(epochs, return_log=True)

    return epochs_ar, reject_log

def pre_pipeline_processing(raw):
    # Implement any pre-processing steps before running the pipeline
    # For example, resampling, filtering, setting channel types, etc.
    return raw

def process_single_fif_file(fif_file_path, bids_dir, output_dir, logger):
    """
    Processes a single FIF file:
    - Converts it to BIDS format.
    - Runs the cleaning pipeline.
    - Exports the cleaned data as a .set file into the output directory.
    """
    try:
        # Ensure the FIF file exists
        fif_file_path = Path(fif_file_path)
        if not fif_file_path.exists():
            console.print(f"[red]File not found: {fif_file_path}[/red]")
            return

        # Create directories if they don't exist
        bids_dir = Path(bids_dir).resolve()
        bids_dir.mkdir(parents=True, exist_ok=True)

        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert the FIF file to BIDS format
        bids_path = convert_single_eeg_to_bids(
            file_path=str(fif_file_path),
            output_dir=str(bids_dir),
            task='resting',  # Adjust task name as needed
            participant_id=None,  # Participant ID will be derived from filename
            line_freq=60.0,
            overwrite=True,
            study_name='EEG Study'
        )

        # Read the BIDS-formatted raw data
        bids_root = bids_path.root
        # Get participant ID from the BIDS structure
        subject_ids = mb.get_entity_vals(bids_root, 'subject')
        if not subject_ids:
            console.print(f"[red]No subject IDs found in BIDS directory[/red]")
            return

        # Assuming single subject for simplicity
        subject_id = subject_ids[0]
        # Build the BIDSPath
        bids_path = BIDSPath(subject=subject_id, task='resting', root=bids_root, datatype='eeg', suffix='eeg', extension='.fif')

        raw = mb.read_raw_bids(bids_path, verbose="ERROR", extra_params={"preload": True})

        # Pre-processing before pipeline
        raw = pre_pipeline_processing(raw)

        # Initialize and run pipeline
        # Assuming lossless_config.yaml is in the same directory as the script
        config_path = Path(__file__).parent / "lossless_config.yaml"
        if not config_path.exists():
            console.print(f"[red]Configuration file not found: {config_path}[/red]")
            return
        pipeline = ll.LosslessPipeline(str(config_path))
        pipeline.run_with_raw(raw)

        # Apply rejection policy
        clean_rejection_policy = get_cleaning_rejection_policy()
        annotated_raw = clean_rejection_policy.apply(pipeline)
        clean_epoched = reject_bad_segs(annotated_raw)
        epochs_ar, reject_log = run_autoreject(clean_epoched)

        # Save cleaned data as .set file to output directory
        set_filename = fif_file_path.stem + '_cleaned.set'
        set_file_path = output_dir / set_filename
        epochs_ar.save(str(set_file_path), overwrite=True, fmt='eeglab')
        console.print(f"[green]Cleaned data saved to {set_file_path}[/green]")

        # Log success
        logger.info(f"Successfully processed {fif_file_path}", extra={"file_processed": str(fif_file_path)})

    except Exception as e:
        console.print(f"[red]Error processing {fif_file_path}: {e}[/red]")
        logger.error(f"Error processing {fif_file_path}: {e}", extra={"file_error": str(fif_file_path)})

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Process a single FIF file through the EEG pipeline.')
    parser.add_argument('fif_file_path', help='Path to the FIF file to process.')
    parser.add_argument('bids_dir', help='Path to the temporary BIDS directory.')
    parser.add_argument('output_dir', help='Path to the output directory for the cleaned file.')
    parser.add_argument('--log_file', default='progress_log.json', help='Path to the log file.')
    args = parser.parse_args()

    # Setup logger
    logger = setup_logger(log_file=args.log_file)

    # Process the single FIF file
    process_single_fif_file(args.fif_file_path, args.bids_dir, args.output_dir, logger)

if __name__ == "__main__":
    main()
