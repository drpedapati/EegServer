# tasks.py
import os
import json
from redis import Redis
from rq import Queue
from dotenv import load_dotenv
from sidecar_manager import SidecarManager
import mne
import time
from pathlib import Path
from signalflow import import_egi, export_eeglab, save_fif

# Load environment variables
load_dotenv()
BASE_DIR = os.getenv('BASE_DIR')
SIDECAR_DIR = os.getenv('SIDECAR_DIR')

# Initialize Redis connection
redis_conn = Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=int(os.getenv('REDIS_PORT', 6379)), db=int(os.getenv('REDIS_DB', 0)))

# Initialize RQ queues
sidecar_q = Queue('sidecar-tasks', connection=redis_conn)
process_q = Queue('process-sidecar', connection=redis_conn)


# Initialize SidecarManager
sidecar_manager = SidecarManager()

def process_new_file(file_path):
    """
    Task to process a new file by creating its sidecar JSON.
    """
    try:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processing new file: {file_path}")
        sidecar_manager.create_sidecar(file_path)
    except Exception as e:
        print(f"Error processing new file {file_path}: {e}")
        raise

def clean_up_raw(file_path):
    """
    Task to erase corresponding json in catalog if raw file is deleted.
    """
    try:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Cleaning up raw file: {file_path}")
        sidecar_manager.clean_up_raw(file_path)
    except Exception as e:
        print(f"Error cleaning up raw file {file_path}: {e}")
        raise

def process_sidecar(sidecar_path):
    """
    Task to process a sidecar JSON file:
    - Load sidecar data.
    - Load EEG data using MNE.
    - Process and save as a set file to the processed directory.
    - Update the sidecar JSON as processed.
    """
    try:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processing sidecar: {sidecar_path}")
        # Load sidecar data
        with open(sidecar_path, 'r') as f:
            sidecar_data = json.load(f)
            
        raw_file_path = sidecar_data['fullpath_file']
        processed_flag = sidecar_data['processed']

        if processed_flag:
            print(f"File {raw_file_path} is already processed.")
            return

        # Check if original file exists
        if not os.path.exists(raw_file_path):
            print(f"Original file {raw_file_path} does not exist.")
            return

        # Load EEG data using MNE
        print(f"Loading EEG data from {raw_file_path}")
        if raw_file_path.lower().endswith('.raw'):
            raw = import_egi(raw_file_path, "EGI_128_RAW")
            # Perform necessary processing here
            # For example, let's apply a band-pass filter
            # raw.filter(1., 40., fir_design='firwin')
            # Save processed data as set file
            
            raw['info']['unprocessed_file'] = str(raw_file_path)
            
            set_dir = Path(BASE_DIR) / "resting_eyesopen" / "processed" / "set_EGI128" / "S01_IMPORT"
            os.makedirs(set_dir, exist_ok=True)
            
            fif_dir = Path(BASE_DIR) / "resting_eyesopen" / "processed" / "fif_EGI128" / "S01_IMPORT"
            os.makedirs(set_dir, exist_ok=True)

            basename = Path(raw_file_path).stem
            set_file_path = set_dir / f"{basename}.set"
            fif_file_path = fif_dir / f"{basename}_raw.fif"
            
            # Saving as FIF format; adjust if you need EEGLAB's set format
            export_eeglab(raw, set_file_path)
            save_fif(raw, fif_file_path)
            
            print(f"Saved processed data to {set_file_path}")
            print(f"Saved processed data to {fif_file_path}")
        elif raw_file_path.lower().endswith('.set'):
            # If the original file is already a set file, handle accordingly
            print(f"Set file {raw_file_path} detected. Processing may vary based on requirements.")
            # Implement processing for set files if needed
            pass
        else:
            print(f"Unsupported file extension {raw_file_path} for file {raw_file_path}")
            return

        # Mark as processed in sidecar
        sidecar_manager.mark_processed(raw_file_path, processed=True)
        print(f"Marked {raw_file_path} as processed in sidecar.")

    except Exception as e:
        print(f"Error processing sidecar {sidecar_path}: {e}")
        raise
