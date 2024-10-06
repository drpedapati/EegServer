import os
import json
import time
import mne

def import_egi(file_path, recording_type):
    """
    Import EGI128 recording, apply montage, and perform epoching if necessary.
    """

    if recording_type == "EGI_128_RAW":
        print("Importing EGI 128 Channel RAW data...")
        # Implement the specific steps for importing EGI128 data
        raw = mne.io.read_raw_egi(input_fname=file_path, preload=False)
        montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
        montage.ch_names[128] = "E129"
        raw.set_montage(montage, match_case=False)
        raw.pick_types(eeg=True, exclude=[])

    if recording_type == "EGI_128_SET":
        print("Importing EGI 128 Channel EEGLAB SET data...")
        try:
            raw = mne.io.read_epochs_eeglab(file_path)
            montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
            montage.ch_names[128] = "E129"
            raw.set_montage(montage, match_case=False)
        except Exception as e:
            print(f"Failed to read raw EEG data, will attempt import with Epochs: {e}")
            raw = mne.io.read_raw_eeglab(file_path, preload=True)

    # This code block is not used in the script. It shows how to load
    # continuous EEG data from a .set file with preload=True. Preloading
    # is useful for preprocessing that needs the full dataset before epoching.

    return raw


def export_eeglab(raw, output_path):
    """
    Export processed data as EEGLAB SET file.
    
    Parameters:
    -----------
    raw : mne.io.Raw
        The processed MNE Raw object to be exported.
    output_path : str or Path
        The full path where the EEGLAB SET file will be saved.
    
    Returns:
    --------
    None
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Export the data
    raw.export(output_path, fmt='eeglab', overwrite=True)
    
    print(f"Processed data saved as EEGLAB SET file: {output_path}")

def save_fif(raw, output):
    """
    Save the processed data as a FIF file.
    
    Parameters:
    -----------
    raw : mne.io.Raw
        The processed MNE Raw object to be saved.
    output : str or Path
        The full path where the FIF file will be saved.
    
    Returns:
    --------
    None
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output), exist_ok=True)
    
    # Save the data as FIF file
    raw.save(output, overwrite=True)
    
    print(f"Processed data saved as FIF file: {output}")