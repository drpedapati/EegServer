import os
import logging
from datetime import datetime
from pathlib import Path
from rich.console import Console
import dotenv
import pandas as pd
import mne

# Initialize Console for pretty printing
console = Console()

# Load environment variables
dotenv.load_dotenv('.env')

# Setup Logger
def setup_logger(log_file='eeg_pipeline.log'):
    logger = logging.getLogger('EEG_Pipeline')
    logger.setLevel(logging.DEBUG)
    
    # Create handlers
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create formatter and add to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

logger = setup_logger()

# Define processing steps
def load_raw_file(file_path, preload=True):
    logger.info(f"Loading raw EEG file from {file_path}")
    raw = mne.io.read_raw_fif(file_path, preload=preload)
    logger.info("Raw EEG data loaded successfully")
    return raw

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

def resample_data(raw, sfreq=250):
    logger.info(f"Resampling data to {sfreq} Hz")
    raw.resample(sfreq)
    logger.info("Resampling completed")
    return raw

def set_eog_channels(raw, eog_channel_names):
    logger.info("Setting EOG channels")
    raw.set_channel_types({ch: 'eog' for ch in eog_channel_names if ch in raw.ch_names})
    logger.info("EOG channels set successfully")
    return raw

def apply_filter(raw, l_freq=None, h_freq=None):
    filter_desc = []
    if l_freq:
        filter_desc.append(f"high-pass={l_freq} Hz")
    if h_freq:
        filter_desc.append(f"low-pass={h_freq} Hz")
    filter_desc = ', '.join(filter_desc)
    logger.info(f"Applying filter: {filter_desc}")
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    logger.info("Filtering completed")
    return raw

def set_eeg_reference(raw, ref='average'):
    logger.info(f"Setting EEG reference to {ref}")
    raw.set_eeg_reference(ref)
    logger.info("EEG reference set")
    return raw

def save_processed_data(raw, output_path):
    logger.info(f"Saving processed data to {output_path}")
    raw.save(output_path, overwrite=True)
    logger.info("Processed data saved successfully")
    return

import os
import json
from datetime import datetime

def generate_unique_filename(base_name, output_dir, ext=".png", metadata=None):
    """ 
    Generate a unique filename using a base name and current timestamp.
    Also generates a JSON sidecar with metadata.

    Parameters:
    -----------
    base_name: str
        The base name of the file to be generated.
    output_dir: str
        Directory where the file and its sidecar will be saved.
    ext: str, optional
        File extension (default is ".png").
    metadata: dict, optional
        Dictionary containing additional metadata to store in the JSON sidecar.

    Returns:
    --------
    str: The path to the generated file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}{ext}"
    file_path = os.path.join(output_dir, filename)
    
    # Generate JSON sidecar
    sidecar_metadata = {
        "filename": filename,
        "timestamp": timestamp,
        "file_path": file_path,
        "file_extension": ext,
        "creation_time": datetime.now().isoformat(),
        "base_name": base_name,
        "output_directory": output_dir
    }
    
    # Add user-provided metadata, if any
    if metadata:
        sidecar_metadata.update(metadata)
    
    # Save the JSON sidecar
    sidecar_path = file_path.replace(ext, '.json')  # Create a .json file next to the output file
    with open(sidecar_path, 'w') as json_file:
        json.dump(sidecar_metadata, json_file, indent=4)
    
    return file_path, sidecar_path


import os
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import mne

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import mne

def plot_topoplot_for_bands(raw, output_dir, bands=None, metadata=None):
    """
    Generate and save a single high-resolution topographical map image 
    for multiple EEG frequency bands arranged horizontally. Also creates a 
    JSON sidecar with metadata for database ingestion.

    Parameters:
    -----------
    raw : mne.io.Raw
        The raw EEG data.
    output_dir : str
        Directory where the topoplot image and JSON sidecar will be saved.
    bands : list of tuple, optional
        List of frequency bands to plot. Each tuple should contain 
        (band_name, lower_freq, upper_freq).
    metadata : dict, optional
        Additional metadata to include in the JSON sidecar.

    Returns:
    --------
    image_path : str
        Path to the saved topoplot image.
    sidecar_path : str
        Path to the saved JSON sidecar.
    """
    
    # Define default frequency bands if none provided
    if bands is None:
        bands = [
            ("Delta", 1, 4),
            ("Theta", 4, 8),
            ("Alpha", 8, 12),
            ("Beta", 12, 30),
            ("Gamma1", 30, 60),
            ("Gamma2", 60, 80)
        ]
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize dictionary to store band powers for metadata
    band_powers_metadata = {}
    
    # Compute PSD using compute_psd
    spectrum = raw.compute_psd(method='welch', fmin=1, fmax=80, picks='eeg')
    
    # Compute band power for each frequency band
    band_powers = []
    for band_name, l_freq, h_freq in bands:
        # Get band power using the spectrum object
        band_power = spectrum.get_data(fmin=l_freq, fmax=h_freq).mean(axis=-1)
        band_powers.append(band_power)
        
        # Store in metadata
        band_powers_metadata[band_name] = {
            "frequency_band": f"{l_freq}-{h_freq} Hz",
            "band_power_mean": float(np.mean(band_power)),
            "band_power_std": float(np.std(band_power))
        }
    # Create a figure with subplots arranged horizontally
    num_bands = len(bands)
    fig, axes = plt.subplots(1, num_bands, figsize=(5*num_bands, 6))  # Increased height
    
    # If only one band, axes is not a list
    if num_bands == 1:
        axes = [axes]
    
    # Add a title to the entire figure with the filename
    fig.suptitle(os.path.basename(raw.filenames[0]), fontsize=16)
    
    for ax, (band, power) in zip(axes, zip(bands, band_powers)):
        band_name, l_freq, h_freq = band
        # Plot topomap
        mne.viz.plot_topomap(
            power, raw.info, axes=ax, show=False, contours=0, cmap='jet'
        )
        ax.set_title(f"{band_name}\n({l_freq}-{h_freq} Hz)", fontsize=12)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to accommodate suptitle
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_filename = f"topoplot_bands_{timestamp}.png"
    image_path = os.path.join(output_dir, image_filename)
    
    # Save the figure in high resolution
    fig.savefig(image_path, dpi=300)
    plt.close(fig)
    
    # Prepare metadata for JSON sidecar
    sidecar_metadata = {
        "filename": image_filename,
        "timestamp": timestamp,
        "file_path": image_path,
        "file_extension": ".png",
        "creation_time": datetime.now().isoformat(),
        "processing_steps": "plot_topoplot_for_bands",
        "frequency_bands": bands,
        "band_powers_summary": band_powers_metadata
    }
    
    # Add user-provided metadata, if any
    if metadata:
        sidecar_metadata.update(metadata)
    
    # Save the JSON sidecar
    sidecar_filename = image_filename.replace(".png", ".json")
    sidecar_path = os.path.join(output_dir, sidecar_filename)
    with open(sidecar_path, 'w') as json_file:
        json.dump(sidecar_metadata, json_file, indent=4)
    
    print(f"Topoplot image saved to: {image_path}")
    print(f"Metadata sidecar saved to: {sidecar_path}")
    
    return image_path, sidecar_path

def save_topoplot(raw, output_dir, time_point=0):
    """ Save high-resolution topoplot at a specific time point. """
    data, times = raw[:, int(raw.info['sfreq'] * time_point)]
    fig, ax = plt.subplots()
    
    mne.viz.plot_topomap(data.ravel(), raw.info, axes=ax, show=False)
    
        # Example usage:
    # Save a file and sidecar, and add some custom metadata
    file_path, sidecar_path = generate_unique_filename(
        base_name="time_series_sample",
        output_dir=output_dir,
        ext=".png",
        metadata={
            "eeg_file_path": os.path.basename(raw.filenames[0]),
            'filepath': raw.filenames[0],
            'n_channels': len(raw.ch_names),
            'sfreq': raw.info['sfreq'],
            'n_samples': len(raw.times),
            "processing_step": "topoplot",
            "version": "1.0",
            "notes": ""
        }
    )
    print(f"File saved to: {file_path}")
    print(f"Metadata sidecar saved to: {sidecar_path}")
    
    # Save figure in high resolution
    fig.savefig(file_path, dpi=300)
    plt.close(fig)
    return file_path

import mne
import matplotlib.pyplot as plt
import numpy as np
import os

def save_spectrograms(raw, output_dir, fmin=1, fmax=40, tmin=0, tmax=None):
    """ Save high-resolution spectrograms for all channels using MNE. """
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute PSD
    spectrum = raw.compute_psd(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, 
                               method='welch', n_fft=256, n_overlap=128)
    
    # Get data and frequencies
    psds, freqs = spectrum.get_data(return_freqs=True)
    
    # Get times
    if tmax is None:
        tmax = raw.times[-1]
    times = raw.times[raw.time_as_index(tmin):raw.time_as_index(tmax)]
    
    # Convert power to dB scale
    psds_db = 10 * np.log10(psds)
    
    # Create a plot for each channel
    for ch_idx, ch_name in enumerate(raw.ch_names):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Check if the channel has any non-zero values
        if np.any(psds_db[ch_idx] != -np.inf):
            mesh = ax.pcolormesh(times, freqs, psds_db[ch_idx].T, cmap='viridis', shading='auto')
            
            # Set labels and title
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title(f"Spectrogram for {ch_name}")
            
            # Add colorbar
            cbar = plt.colorbar(mesh)
            cbar.set_label('Power/Frequency (dB/Hz)')
        else:
            ax.text(0.5, 0.5, 'No data available for this channel', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Spectrogram for {ch_name} - No Data")
        
        # Generate unique filename
        fig_name = os.path.join(output_dir, f"spectrogram_{ch_name.replace(' ', '_')}.png")
        
        # Save figure in high resolution
        fig.savefig(fig_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Spectrograms saved in {output_dir}")

# Example usage:
# raw = mne.io.read_raw_fif('your_raw_file.fif', preload=True)
# output_dir = 'path/to/output/directory'
# save_spectrograms(raw, output_dir, fmin=1, fmax=40)

# Main pipeline function
def eeg_pipeline(raw_file_path, output_dir, apply_high_pass=False, high_pass_freq=0.1):
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load raw data
    raw = import_egi(raw_file_path, recording_type="EGI_128_RAW")
    
    # Pick only EEG channels
    raw.pick_types(eeg=True, exclude=[])
    
    raw = resample_data(raw, sfreq=250)
    raw.crop(tmin=2, tmax=120)
    
    print(raw)
    
    # fig_name = save_topoplot(raw, output_dir, time_point=2)
    

    plot_topoplot_for_bands(raw, output_dir)
    # print(f"Time series saved to: {fig_name}")
    
    # save_spectrograms(raw, output_dir, fmin=1, fmax=40)


    # Optional High-Pass Filter
    # if apply_high_pass:
    #     logger.info(f"Applying high-pass filter at {high_pass_freq} Hz before other processing steps")
    #     raw = apply_filter(raw, l_freq=high_pass_freq, h_freq=None)
    
    # # Resample
    # raw = resample_data(raw, sfreq=250)
    
    # # Set EOG Channels (modify based on your channel names)
    # eog_channels = ['EOG 061', 'EOG 062']  # Example channel names
    # raw = set_eog_channels(raw, eog_channels)
    
    # # Apply Band-Pass Filter
    # raw = apply_filter(raw, l_freq=1, h_freq=40)
    
    # # Set EEG Reference
    # raw = set_eeg_reference(raw, ref='average')
    
    # # Define output file name
    # filename = Path(raw_file_path).stem + '_processed.fif'
    # output_path = Path(output_dir) / filename
    
    # # Save processed data
    # save_processed_data(raw, output_path)
    

    # Get basename of the raw file
    basename = Path(raw_file_path).stem
    
    # Create output filename with _raw.set suffix
    output_filename = f"{basename}_raw.set"
    
    # Construct full output path
    output_path = Path(output_dir) / output_filename
    
    logger.info(f"Saving processed data as EEGLAB SET file: {output_path}")
    raw.export(output_path, fmt='eeglab', overwrite=True)
    
    logger.info("EEG processing pipeline completed successfully")
    return

# Example usage
if __name__ == "__main__":
    # Define paths (update these paths as needed)
    raw_eeg_file = os.getenv('RAW_EEG_FILE')  # Path to the raw EEG file
    output_directory = os.getenv('PROCESSED_EEG_DIR', './processed_eeg')
    
    # Check if raw EEG file exists
    if not os.path.isfile(raw_eeg_file):
        logger.error(f"Raw EEG file not found at {raw_eeg_file}")
        exit(1)
    
    # Run the pipeline with optional high-pass filter
    processed_file = eeg_pipeline(
        raw_file_path=raw_eeg_file,
        output_dir=output_directory,
        apply_high_pass=True,        # Toggle high-pass filter
        high_pass_freq=0.1            # High-pass frequency in Hz
    )
    
    logger.info(f"Processed EEG file saved at {processed_file}")
