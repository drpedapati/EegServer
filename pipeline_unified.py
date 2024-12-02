# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mne",
#     "rich", 
#     "numpy",
#     "python-dotenv",
#     "openneuro-py",
#     "pyyaml",
#     "schema",
#     "mne-bids",
#     "pandas",
#     "pathlib",
#     "pybv",
#     "torch",
#     "pyprep",
#     "eeglabio",
#     "autoreject",
#     "python-ulid",
#     "pylossless @ /Users/ernie/Documents/GitHub/EegServer/pylossless",
# ]
# ///

"""
autoclean: Automated EEG Processing Pipeline

A streamlined pipeline for automated EEG data processing and cleaning.
Handles BIDS conversion, preprocessing, artifact rejection, and quality control.
"""

from pathlib import Path
from typing import Union
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
import os
import sys
import logging
from dotenv import load_dotenv
from rich.table import Table
from rich.logging import RichHandler

from eeg_to_bids_unified import convert_raw_to_bids

import mne_bids as mb
import autoreject
import mne
import pylossless as ll

import yaml
from schema import Schema, Optional, Or

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

import json

from datetime import datetime
import datetime




# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autoclean.log'),
        #logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('autoclean')

load_dotenv()

# sys.tracebacklimit = 0

console = Console()

def resample_data(raw, resample_freq):
    """Resample data using frequency from config."""
    return raw.resample(resample_freq)

def set_eog_channels(raw, eog_channels):
    """Set EOG channels based on config."""
    eog_channels = [
        f"E{ch}" for ch in sorted(eog_channels)
    ]
    raw.set_channel_types({ch: "eog" for ch in raw.ch_names if ch in eog_channels})
    return raw

def crop_data(raw, crop_start, crop_end):
    """Crop data based on config settings."""
    if crop_end is None:
        tmax = raw.times[-1]  # Use the maximum time available
    return raw.load_data().crop(tmin=crop_start, tmax=crop_end)

def set_eeg_reference(raw, ref_type):
    """Set EEG reference based on config."""
    return raw.set_eeg_reference(ref_type, projection=True)

def log_pipeline_start(unprocessed_file: Union[str, Path], eeg_system: str, task: str, autoclean_config_file: Union[str, Path]) -> None:
    """Log and display pipeline startup information."""
    logger.info("Starting autoclean pipeline")
    logger.debug(f"Input parameters: file={unprocessed_file}, system={eeg_system}, task={task}, config={autoclean_config_file}")
    
    console.print(Panel("[bold blue]autoclean: Processing EEG Data[/bold blue]"))
    console.print(f"[cyan]Input File:[/cyan] {unprocessed_file}")
    console.print(f"[cyan]EEG System:[/cyan] {eeg_system}")
    console.print(f"[cyan]Task:[/cyan] {task}")
    console.print(f"[cyan]AutocleanConfig File:[/cyan] {autoclean_config_file}")

def log_pipeline_progress(step: int) -> None:
    """Log pipeline processing progress."""
    logger.debug(f"Processing step {step+1}/3")

def log_pipeline_completion() -> None:
    """Log and display pipeline completion."""
    logger.info("Pipeline processing completed successfully")
    console.print(Panel("[bold green]✅ autoclean Processing Complete![/bold green]"))
    console.print(f"[cyan]Log location:[/cyan] {os.path.abspath('autoclean.log')}\n")

def import_raw_eeg(autoclean_dict: dict, preload: bool = True) -> mne.io.Raw:
    """Import and configure raw EEG data.
    
    Args:
        autoclean_dict: Dictionary containing pipeline configuration including:
                       - unprocessed_file: Path to raw EEG data file
                       - eeg_system: Name of the EEG system montage
        preload: If True, data will be loaded into memory at initialization.
                If False, data will not be loaded until explicitly called.
        
    Returns:
        mne.io.Raw: Imported and configured raw EEG data with appropriate montage set
        
    Raises:
        ValueError: If the specified EEG system is not supported
        FileNotFoundError: If the input file does not exist
        RuntimeError: If there is an error importing or configuring the data
    """
    unprocessed_file = autoclean_dict["unprocessed_file"]
    eeg_system = autoclean_dict["eeg_system"]
    
    logger.info(f"Importing raw EEG data from {unprocessed_file} using {eeg_system} system")
    console.print("[cyan]Importing raw EEG data...[/cyan]")
    
    try:
        # Import based on EEG system type
        if eeg_system == "GSN-HydroCel-129":
            raw = mne.io.read_raw_egi(input_fname=unprocessed_file, preload=preload, events_as_annotations=False)
            montage = mne.channels.make_standard_montage(eeg_system)
            montage.ch_names[128] = "E129"
            raw.set_montage(montage, match_case=False)
            raw.pick('eeg')
        else:
            raise ValueError(f"Unsupported EEG system: {eeg_system}")
            
        logger.info("Raw EEG data imported successfully")
        console.print("[green]✓ Raw EEG data imported successfully[/green]")

        metadata = {
            "import_raw_eeg": {
                "creationDateTime": datetime.datetime.now().isoformat(),
                "sampleRate": raw.info["sfreq"],
                "channelCount": len(raw.ch_names),
                "durationSec": int(raw.n_times) / raw.info["sfreq"],
                "numberSamples": int(raw.n_times)
            }
        }

        # Save metadata
        handle_metadata(autoclean_dict, metadata, mode='save')

        return raw
        
    except Exception as e:
        logger.error(f"Failed to import raw EEG data: {str(e)}")
        console.print("[red]Error importing raw EEG data[/red]")
        raise

def load_config(config_file: Union[str, Path]) -> dict:
    """Load and validate the autoclean configuration file.
    
    Args:
        config_file: Path to YAML configuration file
        
    Returns:
        dict: Task-specific configuration settings
    """
    
    logger.info(f"Loading configuration from {config_file}")
    console.print("[cyan]Loading configuration...[/cyan]")

    # Define configuration schema matching autoclean_config.yaml structure
    config_schema = Schema({
        'tasks': {
            str: {
                'mne_task': str,
                'description': str,
                'lossless_config': str,
                'settings': {
                    'resample_step': {
                        'enabled': bool,
                        'value': int
                    },
                    'eog_step': {
                        'enabled': bool,
                        'value': list
                    },
                    'trim_step': {
                        'enabled': bool,
                        'value': int
                    },
                    'crop_step': {
                        'enabled': bool,
                        'value': {
                            'start': int,
                            'end': Or(float, None)
                        }
                    },
                    'reference_step': {
                        'enabled': bool,
                        'value': str
                    },
                    'filter_step': {
                        'enabled': bool,
                        'value': {
                            'l_freq': Or(float, None),
                            'h_freq': Or(float, None)
                        }
                    },
                    'montage': {
                        'enabled': bool,
                        'value': str
                    }
                },
                'rejection_policy': {
                    'ch_flags_to_reject': list,
                    'ch_cleaning_mode': str,
                    'interpolate_bads_kwargs': {
                        'method': str
                    },
                    'ic_flags_to_reject': list,
                    'ic_rejection_threshold': float,
                    'remove_flagged_ics': bool
                }
            }
        },
        'stage_files': {
            str: {
                'enabled': bool,
                'suffix': str
            }
        }
    })

    try:
        with open(config_file) as f:
            config = yaml.safe_load(f)
            
        # Validate against schema
        autoclean_dict = config_schema.validate(config)
    
        # Log loaded configuration
        logger.info("Configuration loaded successfully")
        console.print("[green]✓ Configuration loaded successfully[/green]")
        logger.debug(f"Task configurations: {autoclean_dict}")
        
        return autoclean_dict
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        console.print("[red]Error loading configuration file[/red]")
        raise

def get_available_tasks(autoclean_dict: dict) -> list[str]:
    """Get available tasks from autoclean dictionary."""
    return list(autoclean_dict['tasks'].keys())

def select_task_config(task: str, task_configs: dict) -> dict:
    """Select and return the configuration for a given task."""
    return task_configs[task]

def handle_metadata(autoclean_dict, metadata=None, mode='load'):
    """
    Handle metadata operations - load, save, or update JSON metadata file.
    
    Args:
        autoclean_dict: Dictionary containing pipeline configuration
        metadata: Optional dictionary of new data to add/update
        mode: Operation mode - 'load' or 'save'
    
    Returns:
        dict: The loaded or updated metadata
    """
    metadata_dir = autoclean_dict["metadata_dir"]
    unprocessed_file = Path(autoclean_dict["unprocessed_file"]) 
    json_file = metadata_dir / f"{unprocessed_file.stem}_autoclean_metadata.json"
    autoclean_dict["metadata_file"] = json_file

    if mode == 'load':
        try:
            with open(json_file, "r") as f:
                metadata = json.load(f)
            # Add unprocessed key if it doesn't exist
            if "handle_metadata" not in metadata:
                metadata["handle_metadata"] = {
                    "creationDateTime": datetime.datetime.now().isoformat(),
                    "unprocessedFile": str(autoclean_dict["unprocessed_file"].name),
                    "unprocessedPath": str(autoclean_dict["unprocessed_file"].parent),
                    "task": autoclean_dict["task"],
                    "eegSystem": autoclean_dict["eeg_system"], 
                    "configFile": str(autoclean_dict["config_file"])
                }
                # Save updated metadata
                with open(json_file, "w") as f:
                    json.dump(metadata, f, indent=4)
            return metadata
        except FileNotFoundError:
            # Initialize new metadata on first run
            metadata = {
                "handle_metadata": {
                    "creationDateTime": datetime.datetime.now().isoformat(),
                    "unprocessedFile": str(autoclean_dict["unprocessed_file"].name),
                    "unprocessedPath": str(autoclean_dict["unprocessed_file"].parent),
                    "task": autoclean_dict["task"],
                    "eegSystem": autoclean_dict["eeg_system"],
                    "configFile": str(autoclean_dict["config_file"])
                }
            }
            # Save initial metadata
            with open(json_file, "w") as f:
                json.dump(metadata, f, indent=4)
            return metadata
            
    elif mode == 'save':
        # Load existing metadata if available
        existing_metadata = {}
        if json_file.exists():
            with open(json_file, "r") as f:
                existing_metadata = json.load(f)
                
            # Add unprocessed key if it doesn't exist
            if "handle_metadata" not in existing_metadata:
                existing_metadata["handle_metadata"] = {
                    "creationDateTime": datetime.datetime.now().isoformat(),
                    "unprocessedFile": str(autoclean_dict["unprocessed_file"].name),
                    "unprocessedPath": str(autoclean_dict["unprocessed_file"].parent),
                    "task": autoclean_dict["task"],
                    "eegSystem": autoclean_dict["eeg_system"],
                    "configFile": str(autoclean_dict["config_file"])
                }
        
        # Update with new metadata if provided
        if metadata is not None:
            # If save_raw_to_set exists, handle as list of dicts
            if "save_raw_to_set" in metadata:
                if "save_raw_to_set" not in existing_metadata:
                    existing_metadata["save_raw_to_set"] = [metadata["save_raw_to_set"]]
                else:
                    if not isinstance(existing_metadata["save_raw_to_set"], list):
                        existing_metadata["save_raw_to_set"] = [existing_metadata["save_raw_to_set"]]
                    existing_metadata["save_raw_to_set"].append(metadata["save_raw_to_set"])
                del metadata["save_raw_to_set"]
            
            # Update remaining metadata
            existing_metadata.update(metadata)

        # Save updated metadata
        with open(json_file, "w") as f:
            json.dump(existing_metadata, f, indent=4)
            console.print("[green]Successfully saved metadata JSON file[/green]")
            
        return existing_metadata


def get_cleaning_rejection_policy(autoclean_dict: dict) -> dict:

    task = autoclean_dict["task"]
    # Create a new rejection policy for cleaning channels and removing ICs
    rejection_policy = ll.RejectionPolicy()

    # Set parameters for channel rejection
    rejection_policy["ch_flags_to_reject"] = autoclean_dict['tasks'][task]['rejection_policy']['ch_flags_to_reject']
    rejection_policy["ch_cleaning_mode"] = autoclean_dict['tasks'][task]['rejection_policy']['ch_cleaning_mode']
    rejection_policy["interpolate_bads_kwargs"] = {"method": autoclean_dict['tasks'][task]['rejection_policy']['interpolate_bads_kwargs']['method']}

    # Set parameters for IC rejection
    rejection_policy["ic_flags_to_reject"] = autoclean_dict['tasks'][task]['rejection_policy']['ic_flags_to_reject']
    rejection_policy["ic_rejection_threshold"] = autoclean_dict['tasks'][task]['rejection_policy']['ic_rejection_threshold']
    rejection_policy["remove_flagged_ics"] = autoclean_dict['tasks'][task]['rejection_policy']['remove_flagged_ics']

    # Add metadata about rejection policy
    metadata = {
        "rejection_policy": {
            "creationDateTime": datetime.datetime.now().isoformat(),
            "task": task,
            "ch_flags_to_reject": rejection_policy["ch_flags_to_reject"],
            "ch_cleaning_mode": rejection_policy["ch_cleaning_mode"],
            "interpolate_method": rejection_policy["interpolate_bads_kwargs"]["method"],
            "ic_flags_to_reject": rejection_policy["ic_flags_to_reject"],
            "ic_rejection_threshold": rejection_policy["ic_rejection_threshold"],
            "remove_flagged_ics": rejection_policy["remove_flagged_ics"]
        }
    }
    handle_metadata(autoclean_dict, metadata, mode='save')

    # Log rejection policy details
    console.print("[bold blue]Rejection Policy Settings:[/bold blue]")
    console.print(f"Channel flags to reject: {rejection_policy['ch_flags_to_reject']}")
    console.print(f"Channel cleaning mode: {rejection_policy['ch_cleaning_mode']}")
    console.print(f"Interpolation method: {rejection_policy['interpolate_bads_kwargs']['method']}")
    console.print(f"IC flags to reject: {rejection_policy['ic_flags_to_reject']}")
    console.print(f"IC rejection threshold: {rejection_policy['ic_rejection_threshold']}")
    console.print(f"Remove flagged ICs: {rejection_policy['remove_flagged_ics']}")

    return rejection_policy


def plot_ica_components_full_duration(pipeline, autoclean_dict):
    """
    Plot ICA components over the full duration with their labels and probabilities.
    
    Parameters:
    -----------
    pipeline : pylossless.Pipeline
        PyLossless pipeline object containing raw data and ICA
    autoclean_dict : dict
        Autoclean dictionary containing metadata
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Get raw and ICA from pipeline
    raw = pipeline.raw
    ica = pipeline.ica2
    ic_labels = pipeline.flags['ic']

    # Get ICA activations and create time vector
    ica_sources = ica.get_sources(raw)
    ica_data = ica_sources.get_data()
    sfreq = raw.info['sfreq']
    times = raw.times
    n_components, n_samples = ica_data.shape

    # Normalize each component individually for better visibility
    for idx in range(n_components):
        component = ica_data[idx]
        # Scale to have a consistent peak-to-peak amplitude
        ptp = np.ptp(component)
        if ptp == 0:
            scaling_factor = 2.5  # Avoid division by zero
        else:
            scaling_factor = 2.5 / ptp
        ica_data[idx] = component * scaling_factor

    # Determine appropriate spacing
    spacing = 2  # Fixed spacing between components

    # Calculate figure size proportional to duration
    total_duration = times[-1] - times[0]
    width_per_second = 0.1  # Increased from 0.02 to 0.1 for wider view
    fig_width = total_duration * width_per_second
    max_fig_width = 200  # Doubled from 100 to allow wider figures
    fig_width = min(fig_width, max_fig_width)
    fig_height = max(6, n_components * 0.5)  # Ensure a minimum height

    # Create plot with wider figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Create a colormap for the components
    cmap = plt.cm.get_cmap('tab20', n_components)
    line_colors = [cmap(i) for i in range(n_components)]

    # Plot components in original order
    for idx in range(n_components):
        offset = idx * spacing
        ax.plot(times, ica_data[idx] + offset, color=line_colors[idx], linewidth=0.5)

    # Set y-ticks and labels
    yticks = [idx * spacing for idx in range(n_components)]
    yticklabels = []
    for idx in range(n_components):
        label_text = f"IC{idx + 1}: {ic_labels['ic_type'][idx]} ({ic_labels['confidence'][idx]:.2f})"
        yticklabels.append(label_text)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=8)

    # Customize axes
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_title('ICA Component Activations (Full Duration)', fontsize=14)
    ax.set_xlim(times[0], times[-1])

    # Adjust y-axis limits
    ax.set_ylim(-spacing, (n_components - 1) * spacing + spacing)

    # Remove y-axis label as we have custom labels
    ax.set_ylabel('')

    # Invert y-axis to have the first component at the top
    ax.invert_yaxis()

    # Color the labels red or black based on component type
    artifact_types = ['eog', 'muscle', 'ecg', 'other']
    for ticklabel, idx in zip(ax.get_yticklabels(), range(n_components)):
        ic_type = ic_labels['ic_type'][idx]
        if ic_type in artifact_types:
            ticklabel.set_color('red')
        else:
            ticklabel.set_color('black')

    # Adjust layout
    plt.tight_layout()

    # Get output path for ICA components figure
    derivatives_path = pipeline.get_derivative_path(autoclean_dict["bids_path"])
    target_figure = str(derivatives_path.copy().update(
        suffix='ica_components_full_duration',
        extension='.png',
        datatype='eeg'
    ))

    # Save figure with higher DPI for better resolution of wider plot
    fig.savefig(target_figure, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return fig

def plot_raw_channels_overlay_full_duration(raw_original, raw_cleaned, pipeline, autoclean_dict, suffix=''):
    """
    Plot raw data channels over the full duration, overlaying the original and cleaned data.
    Original data is plotted in red, cleaned data in black.

    Parameters:
    -----------
    raw_original : mne.io.Raw
        Original raw EEG data before cleaning.
    raw_cleaned : mne.io.Raw
        Cleaned raw EEG data after preprocessing.
    pipeline : pylossless.Pipeline
        Pipeline object (can be None if not used).
    autoclean_dict : dict
        Autoclean dictionary containing metadata.
    suffix : str
        Suffix for the filename.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Ensure that the original and cleaned data have the same channels and times
    if raw_original.ch_names != raw_cleaned.ch_names:
        raise ValueError("Channel names in raw_original and raw_cleaned do not match.")
    if raw_original.times.shape != raw_cleaned.times.shape:
        raise ValueError("Time vectors in raw_original and raw_cleaned do not match.")

    # Get raw data
    channel_labels = raw_original.ch_names
    n_channels = len(channel_labels)
    sfreq = raw_original.info['sfreq']
    times = raw_original.times
    n_samples = len(times)
    data_original = raw_original.get_data()
    data_cleaned = raw_cleaned.get_data()

    # Increase downsample factor to reduce file size
    desired_sfreq = 100  # Reduced sampling rate to 100 Hz
    downsample_factor = int(sfreq // desired_sfreq)
    if downsample_factor > 1:
        data_original = data_original[:, ::downsample_factor]
        data_cleaned = data_cleaned[:, ::downsample_factor]
        times = times[::downsample_factor]

    # Normalize each channel individually for better visibility
    data_original_normalized = np.zeros_like(data_original)
    data_cleaned_normalized = np.zeros_like(data_cleaned)
    spacing = 10  # Fixed spacing between channels
    for idx in range(n_channels):
        # Original data
        channel_data_original = data_original[idx]
        channel_data_original = channel_data_original - np.mean(channel_data_original)  # Remove DC offset
        std = np.std(channel_data_original)
        if std == 0:
            std = 1  # Avoid division by zero
        data_original_normalized[idx] = channel_data_original / std  # Normalize to unit variance

        # Cleaned data
        channel_data_cleaned = data_cleaned[idx]
        channel_data_cleaned = channel_data_cleaned - np.mean(channel_data_cleaned)  # Remove DC offset
        # Use same std for normalization to ensure both signals are on the same scale
        data_cleaned_normalized[idx] = channel_data_cleaned / std

    # Multiply by a scaling factor to control amplitude
    scaling_factor = 2  # Adjust this factor as needed for visibility
    data_original_scaled = data_original_normalized * scaling_factor
    data_cleaned_scaled = data_cleaned_normalized * scaling_factor

    # Calculate offsets for plotting
    offsets = np.arange(n_channels) * spacing

    # Create plot
    total_duration = times[-1] - times[0]
    width_per_second = 0.1  # Adjust this factor as needed
    fig_width = min(total_duration * width_per_second, 50)
    fig_height = max(6, n_channels * 0.25)  # Adjusted for better spacing

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Plot channels
    for idx in range(n_channels):
        ch_name = channel_labels[idx]
        offset = offsets[idx]

        # Plot original data in red
        ax.plot(times, data_original_scaled[idx] + offset, color='red', linewidth=0.5, linestyle='-')

        # Plot cleaned data in black
        ax.plot(times, data_cleaned_scaled[idx] + offset, color='black', linewidth=0.5, linestyle='-')

    # Set y-ticks and labels
    ax.set_yticks(offsets)
    ax.set_yticklabels(channel_labels, fontsize=8)

    # Customize axes
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_title('Raw Data Channels: Original vs Cleaned (Full Duration)', fontsize=14)
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(-spacing, offsets[-1] + spacing)
    ax.set_ylabel('')
    ax.invert_yaxis()

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=0.5, linestyle='-', label='Original Data'),
        Line2D([0], [0], color='black', lw=0.5, linestyle='-', label='Cleaned Data')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()

    # Create Artifact Report
    derivatives_path  = pipeline.get_derivative_path(autoclean_dict["bids_path"])

    # Independent Components
    target_figure = str(derivatives_path.copy().update(
        suffix='data_trace_overlay',
        extension='.png',
        datatype='eeg'
    ))

    # Save as PNG with high DPI for quality
    fig.savefig(target_figure, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Raw channels overlay full duration plot saved to {target_figure}")


    return fig
def plot_psd_overlay(raw_original, raw_cleaned, pipeline, autoclean_dict, suffix=''):
    """
    Generate and save PSD overlays for the original and cleaned data.

    Parameters:
    -----------
    raw_original : mne.io.Raw
        Original raw EEG data before cleaning.
    raw_cleaned : mne.io.Raw
        Cleaned raw EEG data after preprocessing.
    pipeline : pylossless.Pipeline
        Pipeline object (can be None if not used).
    autoclean_dict : dict
        Autoclean dictionary containing metadata.
    suffix : str
        Suffix for the filename.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import mne

    # Select all EEG channels
    picks = mne.pick_types(raw_original.info, eeg=True)
    if len(picks) == 0:
        raise ValueError("No EEG channels found in raw data.")

    # Parameters for PSD
    fmin = 0
    fmax = 100
    n_fft = int(raw_original.info['sfreq'] * 2)  # Window length of 2 seconds

    # Compute PSD for original data
    psd_original = raw_original.compute_psd(
        method='welch',
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        picks=picks,
        average='mean',
        verbose=False
    )
    freqs = psd_original.freqs
    psd_original_data = psd_original.get_data()

    # Compute PSD for cleaned data
    psd_cleaned = raw_cleaned.compute_psd(
        method='welch',
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        picks=picks,
        average='mean',
        verbose=False
    )
    psd_cleaned_data = psd_cleaned.get_data()

    # Average PSD across channels
    psd_original_mean = np.mean(psd_original_data, axis=0)
    psd_cleaned_mean = np.mean(psd_cleaned_data, axis=0)

    # Convert power to dB
    psd_original_db = 10 * np.log10(psd_original_mean)
    psd_cleaned_db = 10 * np.log10(psd_cleaned_mean)

    # Create figure for PSD
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, psd_original_db, color='red', label='Original')
    plt.plot(freqs, psd_cleaned_db, color='black', label='Cleaned')

    # Add vertical lines for power bands
    power_bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 100)
    }

    for band_name, (f_start, f_end) in power_bands.items():
        plt.axvline(f_start, color='grey', linestyle='--', linewidth=1)
        plt.axvline(f_end, color='grey', linestyle='--', linewidth=1)
        # Fill the band area
        plt.fill_betweenx(plt.ylim(), f_start, f_end, color='grey', alpha=0.1)
        # Annotate band names
        plt.text((f_start + f_end) / 2, plt.ylim()[1] - 5, band_name,
                 horizontalalignment='center', verticalalignment='top', fontsize=9, color='grey')

    plt.xlim(fmin, fmax)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB)')
    plt.title('Average Power Spectral Density (0–100 Hz)')
    plt.legend()
    plt.tight_layout()

    # Get output path for bad channels figure
    derivatives_path = pipeline.get_derivative_path(autoclean_dict["bids_path"])
    target_figure = str(derivatives_path.copy().update(
        suffix='psd_overlay',
        extension='.png',
        datatype='eeg'
    ))


    plt.savefig(target_figure, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"PSD overlay saved to {target_figure}")


def plot_topoplot_for_bands(raw, pipeline, autoclean_dict, bands=None, metadata=None):
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

        # Create Artifact Report
    derivatives_path            = pipeline.get_derivative_path(autoclean_dict["bids_path"])

    # Independent Components
    target_figure = str(derivatives_path.copy().update(
        suffix='topoplot',
        extension='.png',
        datatype='eeg'
    ))

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
    # os.makedirs(output_dir, exist_ok=True)
    
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
    image_path = target_figure
    
    # Save the figure in high resolution
    fig.savefig(image_path, dpi=300)
    plt.close(fig)
    
    return image_path

def run_pylossless(autoclean_dict):

    task                    = autoclean_dict["task"]
    bids_path               = autoclean_dict["bids_path"]
    config_path             = autoclean_dict["tasks"][task]["lossless_config"] 
    derivative_name         = "pylossless"
    raw = mb.read_raw_bids(
        bids_path, verbose="ERROR", extra_params={"preload": True}
    )

    
    try:
        pipeline = ll.LosslessPipeline(config_path)
        pipeline.run_with_raw(raw)
        
        derivatives_path = pipeline.get_derivative_path(
            bids_path, derivative_name
        )
        pipeline.save(
            derivatives_path, overwrite=True, format="BrainVision")

    except Exception as e:
        console.print(f"[red]Error: Failed to run pylossless: {str(e)}[/red]")
        raise e

    try:
        pylossless_config = yaml.safe_load(open(config_path))
        metadata = {
            "run_pylossless": {
                "creationDateTime": datetime.datetime.now().isoformat(),
                "derivativeName": derivative_name,
                "configFile": str(config_path),
                "pylossless_config": pylossless_config
            }
        }
        handle_metadata(autoclean_dict, metadata, mode='save')
    except Exception as e:
        console.print(f"[red]Error: Failed to load pylossless config: {str(e)}[/red]")
        raise e

    return pipeline



import numpy as np
from scipy import signal

def detect_bad_eog_channels_fast(raw, decimate_factor=10, window_size=100, threshold_std=5):
    # Decimate the data first
    data = raw.get_data(picks=['eog'])
    decimated_data = signal.decimate(data, decimate_factor, axis=1)
    
    # Adjust window size for decimated data
    adj_window = window_size // decimate_factor
    
    # Use stride tricks on decimated data
    strides = np.lib.stride_tricks.sliding_window_view(decimated_data, 
                                                     window_shape=adj_window, 
                                                     axis=-1)
    
    # Vectorized operations on smaller data
    rolling_std = np.std(strides, axis=-1)
    global_std = np.std(rolling_std, axis=1, keepdims=True)
    global_mean = np.mean(rolling_std, axis=1, keepdims=True)
    outlier_mask = np.any(rolling_std > global_mean + threshold_std * global_std, axis=1)
    
    bad_channels = [raw.ch_names[i] for i in np.where(outlier_mask)[0]]
    return bad_channels


def clean_bad_channels(raw):
    
    from time import perf_counter

    import mne
    import numpy as np
    from scipy import signal as signal

    from pyprep.find_noisy_channels import NoisyChannels


    # bad_channels = detect_bad_eog_channels_fast(raw)

    # Temporarily switch EOG channels to EEG type
    eog_picks = mne.pick_types(raw.info, eog=True, exclude=[])
    eog_ch_names = [raw.ch_names[idx] for idx in eog_picks]
    raw.set_channel_types({ch: 'eeg' for ch in eog_ch_names})

    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])

    cleaned_raw = NoisyChannels(raw, random_state=1337)
    cleaned_raw.find_all_bads(ransac=True, channel_wise=False, max_chunk_size=None)


    # Set EOG channels to EEG temporarily
    #raw.set_channel_types({ch: 'eog' for ch in eog_ch_names})

    threshold=3.0

    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    # start_time = perf_counter()
    # cleaned_raw.find_bad_by_ransac(channel_wise=True)
    # cleaned_raw.find_bad_by_SNR()
    # cleaned_raw.find_bad_by_correlation()
    # cleaned_raw.find_bad_by_deviation(deviation_threshold=5.0)
    # cleaned_raw.find_bad_by_hfnoise(HF_zscore_threshold=5.0)
    # cleaned_raw.find_bad_by_nan_flat()
    # cleaned_raw.get_bads()
    # print("--- %s seconds ---" % (perf_counter() - start_time))
    # breakpoint()

    print(raw.info["bads"])
    raw.info["bads"].extend(cleaned_raw.get_bads())

    return raw

import mne
import numpy as np
import pandas as pd
from autoreject import AutoReject
import random
from typing import Tuple, Dict, Optional
import logging

def clean_epochs(
    epochs: mne.Epochs,
    gfp_threshold: float = 3.0,
    number_of_epochs: Optional[int] = None,
    apply_autoreject: bool = False,
    random_seed: Optional[int] = None
) -> Tuple[mne.Epochs, Dict[str, any]]:
    """
    Clean an MNE Epochs object by applying artifact rejection and removing outlier epochs based on GFP.
    
    Args:
        epochs (mne.Epochs): The input epoched EEG data.
        gfp_threshold (float, optional): Z-score threshold for GFP-based outlier detection. 
                                         Epochs with GFP z-scores above this value are removed.
                                         Defaults to 3.0.
        number_of_epochs (int, optional): If specified, randomly selects this number of epochs from the cleaned data.
                                           If None, retains all cleaned epochs. Defaults to None.
        apply_autoreject (bool, optional): Whether to apply AutoReject for artifact correction. Defaults to True.
        random_seed (int, optional): Seed for random number generator to ensure reproducibility when selecting epochs.
                                     Defaults to None.
    
    Returns:
        Tuple[mne.Epochs, Dict[str, any]]: A tuple containing the cleaned Epochs object and a dictionary of statistics.
    
    Raises:
        ValueError: If after cleaning, the number of epochs is less than `number_of_epochs` when specified.
    """
    logger = logging.getLogger('clean_epochs')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        # Prevent adding multiple handlers in interactive environments
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    logger.info("Starting epoch cleaning process.")
    
    # Force preload to avoid RuntimeError
    if not epochs.preload:
        epochs.load_data()
    
    # Step 1: Artifact Rejection using AutoReject
    if apply_autoreject:
        logger.info("Applying AutoReject for artifact rejection.")
        ar = AutoReject()
        epochs_clean = ar.fit_transform(epochs)
        logger.info(f"Artifacts rejected: {len(epochs) - len(epochs_clean)} epochs removed by AutoReject.")
    else:
        epochs_clean = epochs.copy()
        logger.info("AutoReject not applied. Proceeding without artifact rejection.")
    
    # Step 2: Calculate Global Field Power (GFP)
    logger.info("Calculating Global Field Power (GFP) for each epoch.")
    gfp = np.sqrt(np.mean(epochs_clean.get_data() ** 2, axis=(1, 2)))  # Shape: (n_epochs,)
    
    # Step 3: Epoch Statistics
    epoch_stats = pd.DataFrame({
        'epoch': np.arange(len(gfp)),
        'gfp': gfp,
        'mean_amplitude': epochs_clean.get_data().mean(axis=(1, 2)),
        'max_amplitude': epochs_clean.get_data().max(axis=(1, 2)),
        'min_amplitude': epochs_clean.get_data().min(axis=(1, 2)),
        'std_amplitude': epochs_clean.get_data().std(axis=(1, 2))
    })
    
    # Step 4: Remove Outlier Epochs based on GFP
    logger.info("Removing outlier epochs based on GFP z-scores.")
    gfp_mean = epoch_stats['gfp'].mean()
    gfp_std = epoch_stats['gfp'].std()
    z_scores = np.abs((epoch_stats['gfp'] - gfp_mean) / gfp_std)
    good_epochs_mask = z_scores < gfp_threshold
    removed_by_gfp = np.sum(~good_epochs_mask)
    epochs_final = epochs_clean[good_epochs_mask]
    epoch_stats_final = epoch_stats[good_epochs_mask]
    logger.info(f"Outlier epochs removed based on GFP: {removed_by_gfp}")
    
    # Step 5: Randomly Select a Specified Number of Epochs
    if number_of_epochs is not None:
        if len(epochs_final) < number_of_epochs:
            error_msg = (f"Requested number_of_epochs={number_of_epochs} exceeds the available cleaned epochs={len(epochs_final)}.")
            logger.error(error_msg)
            raise ValueError(error_msg)
        if random_seed is not None:
            random.seed(random_seed)
        selected_indices = random.sample(range(len(epochs_final)), number_of_epochs)
        epochs_final = epochs_final[selected_indices]
        epoch_stats_final = epoch_stats_final.iloc[selected_indices]
        logger.info(f"Randomly selected {number_of_epochs} epochs from the cleaned data.")
    
    # Compile Statistics
    stats = {
        'initial_epochs': len(epochs),
        'after_autoreject': len(epochs_clean),
        'removed_by_autoreject': len(epochs) - len(epochs_clean),
        'removed_by_gfp': removed_by_gfp,
        'final_epochs': len(epochs_final),
        'mean_amplitude': epoch_stats_final['mean_amplitude'].mean(),
        'max_amplitude': epoch_stats_final['max_amplitude'].max(),
        'min_amplitude': epoch_stats_final['min_amplitude'].min(),
        'std_amplitude': epoch_stats_final['std_amplitude'].mean(),
        'mean_gfp': epoch_stats_final['gfp'].mean(),
        'gfp_threshold': gfp_threshold,
        'removed_total': (len(epochs) - len(epochs_clean)) + removed_by_gfp
    }
    
    logger.info("Epoch cleaning process completed.")
    logger.info(f"Final number of epochs: {stats['final_epochs']}")
    
    return epochs_final, stats

def pre_pipeline_processing(raw, autoclean_dict):
    console.print("\n[bold]Pre-pipeline Processing Steps[/bold]")

    task                           = autoclean_dict["task"]
    
    # Get enabled/disabled status for each step
    apply_resample_toggle                   = autoclean_dict["tasks"][task]["settings"]["resample_step"]["enabled"]
    apply_eog_toggle                        = autoclean_dict["tasks"][task]["settings"]["eog_step"]["enabled"]
    apply_average_reference_toggle          = autoclean_dict["tasks"][task]["settings"]["reference_step"]["enabled"]
    apply_trim_toggle                       = autoclean_dict["tasks"][task]["settings"]["trim_step"]["enabled"]
    apply_crop_toggle                       = autoclean_dict["tasks"][task]["settings"]["crop_step"]["enabled"]    
    apply_filter_toggle                     = autoclean_dict["tasks"][task]["settings"]["filter_step"]["enabled"]

    # Print status of each step
    console.print(f"{'✓' if apply_resample_toggle else '✗'} Resample: [{'green' if apply_resample_toggle else 'red'}]{apply_resample_toggle}[/]")
    console.print(f"{'✓' if apply_eog_toggle else '✗'} EOG Assignment: [{'green' if apply_eog_toggle else 'red'}]{apply_eog_toggle}[/]")
    console.print(f"{'✓' if apply_average_reference_toggle else '✗'} Average Reference: [{'green' if apply_average_reference_toggle else 'red'}]{apply_average_reference_toggle}[/]")
    console.print(f"{'✓' if apply_trim_toggle else '✗'} Edge Trimming: [{'green' if apply_trim_toggle else 'red'}]{apply_trim_toggle}[/]")
    console.print(f"{'✓' if apply_crop_toggle else '✗'} Duration Cropping: [{'green' if apply_crop_toggle else 'red'}]{apply_crop_toggle}[/]")
    console.print(f"{'✓' if apply_filter_toggle else '✗'} Filtering: [{'green' if apply_filter_toggle else 'red'}]{apply_filter_toggle}[/]\n")

    # Initialize metadata
    metadata = {
        "pre_pipeline_processing": {
            "creationDateTime": datetime.datetime.now().isoformat(),
            "ResampleHz": None,
            "TrimSec": None,
            "LowPassHz1": None, 
            "HighPassHz1": None,
            "CropDurationSec": None,
            "AverageReference": apply_average_reference_toggle,
            "EOGChannels": None
        }
    }

    # Resample
    if apply_resample_toggle:
        console.print("[cyan]Resampling data...[/cyan]")
        target_sfreq = autoclean_dict["tasks"][task]["settings"]["resample_step"]["value"]
        raw = resample_data(raw, target_sfreq)
        console.print(f"[green]✓ Data resampled to {target_sfreq} Hz[/green]")
        metadata["pre_pipeline_processing"]["ResampleHz"] = target_sfreq
        save_raw_to_set(raw, autoclean_dict, 'post_resample')
    
    # EOG Assignment
    if apply_eog_toggle:
        console.print("[cyan]Setting EOG channels...[/cyan]")
        eog_channels = autoclean_dict["tasks"][task]["settings"]["eog_step"]["value"]
        raw = set_eog_channels(raw, eog_channels)
        console.print("[green]✓ EOG channels assigned[/green]")
        metadata["pre_pipeline_processing"]["EOGChannels"] = eog_channels
    
    # Average Reference
    if apply_average_reference_toggle:
        console.print("[cyan]Applying average reference...[/cyan]")
        ref_type = autoclean_dict["tasks"][task]["settings"]["reference_step"]["value"]
        raw = set_eeg_reference(raw, ref_type)
        console.print("[green]✓ Average reference applied[/green]")
        save_raw_to_set(raw, autoclean_dict, 'post_reference')
    
    # Trim Edges
    if apply_trim_toggle:
        console.print("[cyan]Trimming data edges...[/cyan]")
        trim = autoclean_dict["tasks"][task]["settings"]["trim_step"]["value"]
        start_time = raw.times[0]
        end_time = raw.times[-1]
        raw.crop(tmin=start_time + trim, tmax=end_time - trim)   
        console.print(f"[green]✓ Data trimmed by {trim}s from each end[/green]")
        metadata["pre_pipeline_processing"]["TrimSec"] = trim
        save_raw_to_set(raw, autoclean_dict, 'post_trim')
    # Crop Duration
    if apply_crop_toggle:
        console.print("[cyan]Cropping data duration...[/cyan]")
        start_time = autoclean_dict["tasks"][task]["settings"]["crop_step"]["value"]['start']
        end_time = autoclean_dict["tasks"][task]["settings"]["crop_step"]["value"]['end']
        if end_time is None:
            end_time = raw.times[-1]  # Use full duration if end is null
        raw.crop(tmin=start_time, tmax=end_time)
        target_crop_duration = raw.times[-1] - raw.times[0]
        console.print(f"[green]✓ Data cropped to {target_crop_duration:.1f}s[/green]")
        metadata["pre_pipeline_processing"]["CropDurationSec"] = target_crop_duration
        metadata["pre_pipeline_processing"]["CropStartSec"] = start_time
        metadata["pre_pipeline_processing"]["CropEndSec"] = end_time
        save_raw_to_set(raw, autoclean_dict, 'post_crop')

    # Pre-Filter    
    if apply_filter_toggle:
        console.print("[cyan]Applying frequency filters...[/cyan]")
        target_lfreq = float(autoclean_dict["tasks"][task]["settings"]["filter_step"]["value"]["l_freq"])
        target_hfreq = float(autoclean_dict["tasks"][task]["settings"]["filter_step"]["value"]["h_freq"])
        raw.filter(l_freq=target_lfreq, h_freq=target_hfreq)
        console.print(f"[green]✓ Applied bandpass filter: {target_lfreq}-{target_hfreq} Hz[/green]")
        metadata["pre_pipeline_processing"]["LowPassHz1"] = target_lfreq
        metadata["pre_pipeline_processing"]["HighPassHz1"] = target_hfreq
        save_raw_to_set(raw, autoclean_dict, 'post_filter')
    
    # Save metadata
    handle_metadata(autoclean_dict, metadata, mode='save')
    return raw

def create_bids_path(raw, autoclean_dict):

    unprocessed_file        = autoclean_dict["unprocessed_file"]
    task                    = autoclean_dict["task"]
    mne_task                = autoclean_dict["tasks"][task]["mne_task"]
    bids_dir                = autoclean_dict["bids_dir"]
    eeg_system              = autoclean_dict["eeg_system"]
    config_file             = autoclean_dict["config_file"]

    try:
        bids_path = convert_raw_to_bids(
            raw,
            output_dir=str(bids_dir),
            task=mne_task,
            participant_id=None,
            line_freq=60.0,
            overwrite=True,
            study_name=unprocessed_file.stem
        )

        autoclean_dict["bids_path"] = bids_path
        autoclean_dict["bids_basename"] = bids_path.basename

        metadata = {
            "convert_raw_to_bids": {
                "creationDateTime": datetime.datetime.now().isoformat(),
                "bids_output_dir": str(bids_dir),
                "bids_path": str(bids_path),
                "bids_basename": bids_path.basename,
                "study_name": unprocessed_file.stem,
                "task": mne_task,
                "participant_id": None,
                "line_freq": 60.0,
                "eegSystem": eeg_system,
                "configFile": str(config_file)
            }
        }

        handle_metadata(autoclean_dict, metadata, mode='save')

        return raw, autoclean_dict
    
    except Exception as e:
        console.print(f"[red]Error converting raw to bids: {e}[/red]")
        raise e 
def save_raw_to_set(raw, autoclean_dict, stage="post_import", output_path=None):
    """Save raw EEG data to SET format with descriptive filename.
    
    Args:
        raw: MNE Raw object containing EEG data
        output_path: Path object specifying output directory
        autoclean_dict: Dictionary containing configuration and paths
        stage: Processing stage to get suffix from stage_files config
        
    Returns:
        Path: Path to the saved SET file
    """
    #Only save if enabled for this stage in stage_files config
    if not autoclean_dict['stage_files'][stage]['enabled']:
        return None
        
    # Get suffix from stage_files config
    suffix = autoclean_dict['stage_files'][stage]['suffix']

    # Create subfolder using suffix name
    if output_path is None:
        output_path = autoclean_dict["stage_dir"]

    subfolder = output_path / suffix
    subfolder.mkdir(exist_ok=True)

    basename = Path(autoclean_dict["unprocessed_file"]).stem
    set_path = subfolder / f"{basename}{suffix}_raw.set"
    
    raw.export(set_path, fmt='eeglab', overwrite=True)
    console.print(f"[green]Saved stage file for {stage} to: {basename}[/green]")

    metadata = {
        "save_raw_to_set": {
            "creationDateTime": datetime.datetime.now().isoformat(),
            "stage": stage,
            "outputPath": str(set_path),
            "suffix": suffix,
            "basename": basename,
            "format": "eeglab"
        }
    }
    handle_metadata(autoclean_dict, metadata, mode='save')

    return set_path

def save_epochs_to_set(epochs, autoclean_dict, stage="post_import", output_path=None):
    """Save epoched EEG data to SET format with descriptive filename.
    
    Args:
        epochs: MNE Epochs object containing EEG data
        output_path: Path object specifying output directory
        autoclean_dict: Dictionary containing configuration and paths
        stage: Processing stage to get suffix from stage_files config
        
    Returns:
        Path: Path to the saved SET file
    """
    #Only save if enabled for this stage in stage_files config
    if not autoclean_dict['stage_files'][stage]['enabled']:
        return None
        
    # Get suffix from stage_files config
    suffix = autoclean_dict['stage_files'][stage]['suffix']

    # Create subfolder using suffix name
    if output_path is None:
        output_path = autoclean_dict["stage_dir"]

    subfolder = output_path / suffix
    subfolder.mkdir(exist_ok=True)

    basename = Path(autoclean_dict["unprocessed_file"]).stem
    set_path = subfolder / f"{basename}{suffix}_epo.set"
    
    epochs.export(set_path, fmt='eeglab', overwrite=True)
    console.print(f"[green]Saved stage file for {stage} to: {basename}[/green]")

    metadata = {
        "save_raw_to_set": {
            "creationDateTime": datetime.datetime.now().isoformat(),
            "stage": stage,
            "outputPath": str(set_path),
            "suffix": suffix,
            "basename": basename,
            "format": "eeglab",
            "n_epochs": len(epochs),
            "tmin": epochs.tmin,
            "tmax": epochs.tmax
        }
    }
    handle_metadata(autoclean_dict, metadata, mode='save')

    return set_path

def plot_bad_channels_full_duration(raw_original, raw_cleaned, pipeline, autoclean_dict):
    """
    Plot only the bad channels over the full duration, overlaying the original
    and interpolated data. Original data is plotted in red, interpolated data in black.

    Parameters:
    -----------
    raw_original : mne.io.Raw
        Original raw EEG data before cleaning.
    raw_cleaned : mne.io.Raw
        Cleaned raw EEG data after interpolation of bad channels.
    pipeline : pylossless.Pipeline
        Pipeline object containing flags and raw data.
    autoclean_dict : dict
        Autoclean dictionary containing metadata.
    suffix : str
        Suffix for the filename.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Collect bad channels and their reasons
    bad_channels_info = {}

    # Mapping from channel to reason(s)
    for reason, channels in pipeline.flags['ch'].items():
        for ch in channels:
            if ch in bad_channels_info:
                if reason not in bad_channels_info[ch]:
                    bad_channels_info[ch].append(reason)
            else:
                bad_channels_info[ch] = [reason]

    bad_channels = list(bad_channels_info.keys())

    if not bad_channels:
        print("No bad channels were identified.")
        return

    # Get data for the bad channels from both original and cleaned data
    picks_original = mne.pick_channels(raw_original.ch_names, bad_channels)
    picks_cleaned = mne.pick_channels(raw_cleaned.ch_names, bad_channels)

    data_original, times = raw_original.get_data(picks=picks_original, return_times=True)
    data_cleaned = raw_cleaned.get_data(picks=picks_cleaned)

    channel_labels = [raw_original.ch_names[i] for i in picks_original]
    n_channels = len(channel_labels)

    # Increase downsample factor to reduce file size
    sfreq = raw_original.info['sfreq']
    desired_sfreq = 100  # Reduced sampling rate to 100 Hz
    downsample_factor = int(sfreq // desired_sfreq)
    if downsample_factor > 1:
        data_original = data_original[:, ::downsample_factor]
        data_cleaned = data_cleaned[:, ::downsample_factor]
        times = times[::downsample_factor]

    # Normalize each channel individually for better visibility
    data_original_normalized = np.zeros_like(data_original)
    data_cleaned_normalized = np.zeros_like(data_cleaned)
    spacing = 10  # Fixed spacing between channels
    for idx in range(n_channels):
        channel_data_original = data_original[idx]
        channel_data_cleaned = data_cleaned[idx]
        # Remove DC offset
        channel_data_original = channel_data_original - np.mean(channel_data_original)
        channel_data_cleaned = channel_data_cleaned - np.mean(channel_data_cleaned)
        # Use standard deviation of original data for normalization
        std = np.std(channel_data_original)
        if std == 0:
            std = 1  # Avoid division by zero
        data_original_normalized[idx] = channel_data_original / std
        data_cleaned_normalized[idx] = channel_data_cleaned / std

    # Multiply by a scaling factor to control amplitude
    scaling_factor = 2  # Adjust this factor as needed for visibility
    data_original_scaled = data_original_normalized * scaling_factor
    data_cleaned_scaled = data_cleaned_normalized * scaling_factor

    # Calculate offsets for plotting
    offsets = np.arange(n_channels) * spacing

    # Create plot
    total_duration = times[-1] - times[0]
    width_per_second = 0.1  # Adjusted for better scaling
    fig_width = min(total_duration * width_per_second, 50)
    fig_height = max(6, n_channels * 0.5)  # Adjusted for better spacing

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Plot bad channels
    for idx in range(n_channels):
        ch_name = channel_labels[idx]
        offset = offsets[idx]
        reasons = bad_channels_info.get(ch_name, [])
        label = f"{ch_name} ({', '.join(reasons)})"

        # Plot the original data in red
        ax.plot(times, data_original_scaled[idx] + offset, color='red', linewidth=0.5, linestyle='-')

        # Plot the cleaned (interpolated) data in black
        ax.plot(times, data_cleaned_scaled[idx] + offset, color='black', linewidth=0.5, linestyle='-')

        # Add channel label
        ax.text(times[0] - (0.01 * total_duration), offset, label, horizontalalignment='right', fontsize=8)

    # Customize axes
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_title('Bad Channels: Original vs Interpolated (Full Duration)', fontsize=14)
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(-spacing, offsets[-1] + spacing)
    ax.set_yticks([])  # Hide y-ticks as we have labels next to each channel
    ax.invert_yaxis()

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=0.5, linestyle='-', label='Original Data'),
        Line2D([0], [0], color='black', lw=0.5, linestyle='-', label='Interpolated Data')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    # Enhance the layout
    plt.tight_layout()

    # Get output path for bad channels figure
    derivatives_path = pipeline.get_derivative_path(autoclean_dict["bids_path"])
    target_figure = str(derivatives_path.copy().update(
        suffix='bad_channels',
        extension='.png',
        datatype='eeg'
    ))

    # Save as PNG with high DPI for quality
    fig.savefig(target_figure, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Bad channels full duration plot saved to {target_figure}")

    return fig


def clean_artifacts_continuous(pipeline, autoclean_dict):

    bids_path = autoclean_dict["bids_path"]
    
    # Apply rejection policy
    rejection_policy = get_cleaning_rejection_policy(autoclean_dict)
    cleaned_raw = rejection_policy.apply(pipeline)

    # Save cleaned raw data
    derivatives_path            = pipeline.get_derivative_path(bids_path)
    derivatives_path.suffix     = "eeg"
    pipeline.save(derivatives_path, overwrite=True, format="BrainVision")

    # Save cleaned raw data to set
    save_raw_to_set(cleaned_raw, autoclean_dict, 'post_rejection_policy')

    # # Plot bad channels separately
    # plot_bad_channels_full_duration(pipeline.raw, cleaned_raw, pipeline, autoclean_dict)

    # # Plot topoplot for bands
    # plot_topoplot_for_bands(cleaned_raw, pipeline,autoclean_dict)

    # # Generate ICA reports
    # generate_ica_reports(pipeline, cleaned_raw, autoclean_dict, duration=60)


    # # Plot ICA components full duration
    # plot_ica_components_full_duration(pipeline, autoclean_dict)

    # # Call the function to plot and save the overlay
    # plot_raw_channels_overlay_full_duration(
    #     pipeline.raw,
    #     cleaned_raw,
    #     pipeline,
    #     autoclean_dict,
    #     suffix='overlay'
    # )

    plot_psd_overlay(pipeline.raw, cleaned_raw, pipeline, autoclean_dict, suffix='spectrogram')


    epochs = mne.make_fixed_length_epochs(cleaned_raw, duration=2)
    cleaned_epochs, stats = clean_epochs(epochs, number_of_epochs=80)
    cleaned_epochs.load_data()
    save_epochs_to_set(cleaned_epochs, autoclean_dict, 'post_clean_epochs')

    # report_artifact_rejection(pipeline, cleaned_raw, autoclean_dict)

    return pipeline, autoclean_dict
  
def plot_ica_components(pipeline, cleaned_raw, autoclean_dict, duration=60, components='all'):
    """
    Plots ICA components with labels and saves reports.

    Parameters:
    -----------
    pipeline : pylossless.Pipeline
        Pipeline object containing raw data and ICA.
    autoclean_dict : dict
        Autoclean dictionary containing metadata.
    duration : int
        Duration in seconds to plot.
    components : str
        'all' to plot all components, 'rejected' to plot only rejected components.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.gridspec import GridSpec

    # Get raw and ICA from pipeline
    raw = pipeline.raw
    ica = pipeline.ica2
    ic_labels = pipeline.flags['ic']

    # Determine components to plot
    if components == 'all':
        component_indices = range(ica.n_components_)
        report_name = 'ica_components_all'
    elif components == 'rejected':
        component_indices = ica.exclude
        report_name = 'ica_components_rejected'
        if not component_indices:
            print("No components were rejected. Skipping rejected components report.")
            return
    else:
        raise ValueError("components parameter must be 'all' or 'rejected'.")

    # Get ICA activations
    ica_sources = ica.get_sources(raw)
    ica_data = ica_sources.get_data()

    # Limit data to specified duration
    sfreq = raw.info['sfreq']
    n_samples = int(duration * sfreq)
    times = raw.times[:n_samples]

    # Create output path for the PDF report
    derivatives_path = pipeline.get_derivative_path(autoclean_dict["bids_path"])
    pdf_path = str(
        derivatives_path.copy().update(suffix=report_name, extension='.pdf')
    )

    # Remove existing file
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    with PdfPages(pdf_path) as pdf:
        # First page: Component topographies overview
        fig_topo = ica.plot_components(picks=component_indices, show=False)
        if isinstance(fig_topo, list):
            for f in fig_topo:
                pdf.savefig(f)
                plt.close(f)
        else:
            pdf.savefig(fig_topo)
            plt.close(fig_topo)

        # If rejected components, add overlay plot
        if components == 'rejected':
            fig_overlay = plt.figure()
            end_time = min(30., pipeline.raw.times[-1])
            fig_overlay = pipeline.ica2.plot_overlay(pipeline.raw, start=0, stop=end_time, exclude=component_indices, show=False)
            fig_overlay.set_size_inches(15, 10)  # Set size after creating figure

            pdf.savefig(fig_overlay)
            plt.close(fig_overlay)

        # For each component, create detailed plots
        for idx in component_indices:
            fig = plt.figure(constrained_layout=True, figsize=(12, 8))
            gs = GridSpec(nrows=3, ncols=3, figure=fig)

            # Axes for ica.plot_properties
            ax1 = fig.add_subplot(gs[0, 0])  # Data
            ax2 = fig.add_subplot(gs[0, 1])  # Epochs image
            ax3 = fig.add_subplot(gs[0, 2])  # ERP/ERF
            ax4 = fig.add_subplot(gs[1, 0])  # Spectrum
            ax5 = fig.add_subplot(gs[1, 1])  # Topomap
            ax_props = [ax1, ax2, ax3, ax4, ax5]

            # Plot properties
            ica.plot_properties(
                raw,
                picks=[idx],
                axes=ax_props,
                dB=True,
                plot_std=True,
                log_scale=False,
                reject='auto',
                show=False
            )

            # Add time series plot
            ax_timeseries = fig.add_subplot(gs[2, :])  # Last row, all columns
            ax_timeseries.plot(times, ica_data[idx, :n_samples], linewidth=0.5)
            ax_timeseries.set_xlabel('Time (seconds)')
            ax_timeseries.set_ylabel('Amplitude')
            ax_timeseries.set_title(f'Component {idx + 1} Time Course ({duration}s)')

            # Add labels
            comp_info = ic_labels.iloc[idx]
            label_text = (
                f"Component {idx + 1}\n"
                f"Type: {comp_info['ic_type']}\n"
                f"Confidence: {comp_info['confidence']:.2f}"
            )

            fig.suptitle(
                label_text,
                fontsize=14,
                fontweight='bold',
                color='red' if comp_info['ic_type'] in ['eog', 'muscle', 'ecg', 'other'] else 'black'
            )

            # Save the figure
            pdf.savefig(fig)
            plt.close(fig)

        print(f"Report saved to {pdf_path}")

def generate_ica_reports(pipeline, cleaned_raw, autoclean_dict, duration=60):
    """
    Generates two reports:
    1. All ICA components.
    2. Only the rejected ICA components.

    Parameters:
    -----------
    pipeline : pylossless.Pipeline
        The pipeline object containing the ICA and raw data.
    autoclean_dict : dict
        Dictionary containing configuration and paths.
    duration : int
        Duration in seconds for plotting time series data.
    """
    # Generate report for all components
    plot_ica_components(pipeline, cleaned_raw, autoclean_dict, duration=duration, components='all')

    # Generate report for rejected components
    plot_ica_components(pipeline, cleaned_raw, autoclean_dict, duration=duration, components='rejected')


def report_artifact_rejection(pipeline, cleaned_raw, autoclean_dict):
    """Generate a report for artifact rejection, plotting only the removed ICA components with improved layout."""
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.gridspec import GridSpec

    # Get the list of removed ICA components
    removed_ics = pipeline.ica2.exclude
    cleaned_raw.info['temp'] = {}
    cleaned_raw.info['temp']['removed_ics'] = removed_ics

    # Create the path for saving the artifact report
    derivatives_path = pipeline.get_derivative_path(autoclean_dict["bids_path"])
    ic_flags_path = str(
        derivatives_path.copy().update(suffix='ic_flags', extension='.pdf')
    )

    # Remove file if it exists
    if os.path.exists(ic_flags_path):
        os.remove(ic_flags_path)

    # Prepare to save figures to PDF
    with PdfPages(ic_flags_path) as pdf:
        # For each removed component, create a figure with improved layout
        for idx in removed_ics:
            # Create figure with constrained layout
            fig = plt.figure(constrained_layout=True, figsize=(12, 8))
            gs = GridSpec(nrows=3, ncols=3, figure=fig)

            # Axes for ica.plot_properties (5 axes)
            ax1 = fig.add_subplot(gs[0, 0])  # Data
            ax2 = fig.add_subplot(gs[0, 1])  # Epochs image
            ax3 = fig.add_subplot(gs[0, 2])  # ERP/ERF
            ax4 = fig.add_subplot(gs[1, 0])  # Spectrum
            ax5 = fig.add_subplot(gs[1, 1])  # Topomap
            ax_props = [ax1, ax2, ax3, ax4, ax5]

            # Plot the properties into the axes
            pipeline.ica2.plot_properties(
                cleaned_raw,
                picks=[idx],
                axes=ax_props,
                dB=True,
                plot_std=True,
                log_scale=False,
                reject='auto',
                show=False
            )

            # Add time series plot
            ax_timeseries = fig.add_subplot(gs[2, :])  # Last row, all columns
            # Get the ICA activations (sources)
            ica_sources = pipeline.ica2.get_sources(cleaned_raw)
            ica_data = ica_sources.get_data()
            times = cleaned_raw.times
            # Plot the time series for the component
            ax_timeseries.plot(times, ica_data[idx], linewidth=0.5)
            ax_timeseries.set_xlabel('Time (seconds)')
            ax_timeseries.set_ylabel('Amplitude')
            ax_timeseries.set_title(f'Component {idx} Time Course')

            # Add labels
            # Assuming you have component labels or types in pipeline.flags['ic']
            comp_info = pipeline.flags['ic'].loc[idx]
            label_text = (
                f"Type: {comp_info['ic_type']}\n"
                f"Confidence: {comp_info['confidence']:.3f}"
            )

            fig.suptitle(
                label_text,
                fontsize=14,
                fontweight='bold',
                color='red' if comp_info['ic_type'] in ['muscle', 'eog'] else 'black'
            )

            # Save the figure
            pdf.savefig(fig)
            plt.close(fig)

        # Plot overlay of raw and cleaned data for the removed components
        fig_overlay = pipeline.ica2.plot_overlay(
            pipeline.raw,
            exclude=removed_ics,
            show=False
        )
        pdf.savefig(fig_overlay)
        plt.close(fig_overlay)

        # Plot the bad channels
        # Collect bad channels
        noisy_channels = pipeline.flags['ch']['noisy']
        bridged_channels = pipeline.flags['ch']['bridged']
        rank_deficient_channels = pipeline.flags['ch']['rank']
        uncorrelated_channels = pipeline.flags['ch']['uncorrelated']

        # Combine all bad channels into a single list
        bad_channels = []
        bad_channels.extend(noisy_channels)
        bad_channels.extend(bridged_channels)
        bad_channels.extend(rank_deficient_channels)
        bad_channels.extend(uncorrelated_channels)
        # Remove duplicates while preserving order
        bad_channels = list(dict.fromkeys(bad_channels))
        bad_channels = [str(channel) for channel in bad_channels]

        if bad_channels:
            # Get data for the bad channels
            data, times = cleaned_raw.get_data(picks=bad_channels, return_times=True)

            # Create a figure with appropriate layout
            fig_bad_ch = plt.figure(constrained_layout=True, figsize=(12, 6))
            ax = fig_bad_ch.add_subplot(111)

            # Plot each bad channel with offset
            for idx, channel in enumerate(bad_channels):
                offset = idx * np.ptp(data) * 0.1  # Offset for clarity
                ax.plot(times, data[idx] + offset, label=channel)

            # Customize the plot
            ax.set_title("Bad Channels")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.legend(loc="upper right")

            # Save the figure
            pdf.savefig(fig_bad_ch)
            plt.close(fig_bad_ch)

    

def process_resting_eyesopen(autoclean_dict: dict) -> None:
    """Process resting state eyes-open data."""

    # Initialize metadata tracking
    handle_metadata(autoclean_dict, mode='save')

    # Import and save raw EEG data
    raw = import_raw_eeg(autoclean_dict)
    save_raw_to_set(raw, autoclean_dict, 'post_import')

    # Run preprocessing pipeline and save intermediate result
    raw = pre_pipeline_processing(raw, autoclean_dict)
    save_raw_to_set(raw, autoclean_dict, 'post_prepipeline')

    # Create BIDS-compliant paths and filenames
    raw, autoclean_dict = create_bids_path(raw, autoclean_dict)

    raw = clean_bad_channels(raw)

    # Run PyLossless pipeline and save result
    pipeline = run_pylossless(autoclean_dict)
    save_raw_to_set(raw, autoclean_dict, 'post_pylossless')

    # Artifact Rejection
    pipeline, autoclean_dict = clean_artifacts_continuous(pipeline, autoclean_dict)

    console.print("[green]✓ Completed[/green]")

def process_chirp_default(raw: mne.io.Raw) -> None:
    """Process chirp default data."""
    pass

def process_assr_default(raw: mne.io.Raw) -> None:
    """Process assr default data."""
    pass

def extract_task_dict(full_dict, task_name):
    task_dict = {
        'task_config': full_dict['tasks'][task_name],
        'stage_files': full_dict['stage_files']
    }
    return task_dict

def entrypoint(
    unprocessed_file: Union[str, Path], 
    task: str
) -> None:
    """
    Main entry point for the autoclean pipeline.
    
    Args:
        unprocessed_file: Path to raw EEG data file
        eeg_system: Name of the EEG system used
        task: Task/experiment name
        config_file: Path to pipeline configuration file
    """

    # Validate environment variables
    autoclean_dir, autoclean_config_file    = validate_environment_variables()

    # Validate autoclean configuration
    autoclean_dict                          = validate_autoclean_config(autoclean_config_file)

    # Validate task and EEG system
    task                                    = validate_task(task, get_available_tasks(autoclean_dict))
    eeg_system                              = validate_eeg_system(autoclean_dict['tasks'][task]['settings']['montage']['value'])

    # Validate input file
    validate_input_file(unprocessed_file)

    # Log pipeline start    
    log_pipeline_start(unprocessed_file, eeg_system, task, autoclean_config_file)
    
    # Prepare directories
    autoclean_dir, bids_dir, metadata_dir, clean_dir, stage_dir, debug_dir = prepare_directories(task)

    autoclean_dict = {
        'task': task,
        'eeg_system': eeg_system,
        'config_file': autoclean_config_file,
        'tasks': {
            task: autoclean_dict['tasks'][task]
        },
        'stage_files': autoclean_dict['stage_files'],
        'unprocessed_file': unprocessed_file,
        'autoclean_dir': autoclean_dir,
        'bids_dir': bids_dir,
        'metadata_dir': metadata_dir,
        'clean_dir': clean_dir,
        'debug_dir': debug_dir,
        'stage_dir': stage_dir
    }

    # Branch to task-specific and eeg system-specific processing
    if task == "rest_eyesopen":
        process_resting_eyesopen(autoclean_dict)
    
    elif task == "chirp_default":
        if eeg_system == "EGI_129":
            pass
        else:
            raise ValueError(f"Unsupported EEG system for task: {eeg_system}")

    elif task == "assr_default":
        if eeg_system == "EGI_129":
            pass
        else:
            raise ValueError(f"Unsupported EEG system for task: {eeg_system}")

    else:
        raise ValueError(f"Unsupported task: {task}")
    
    log_pipeline_completion()

def prepare_directories(task: str) -> tuple[Path, Path, Path, Path, Path]:
    """Create and return required autoclean pipeline directories."""
    logger.info(f"Setting up pipeline directories for task: {task}")
    console.print("[bold blue]Setting up autoclean pipeline directories...[/bold blue]")
    
    autoclean_dir = Path(os.getenv("AUTOCLEAN_DIR"))
    dirs = {
        "bids": autoclean_dir / task / "bids",
        "metadata": autoclean_dir / task / "metadata", 
        "clean": autoclean_dir / task / "postcomps",
        "debug": autoclean_dir / task / "debug",
        "stage": autoclean_dir / task / "stage"
    }
    
    for name, dir in track(dirs.items(), description="Creating directories"):
        logger.debug(f"Creating directory: {dir}")
        dir.mkdir(parents=True, exist_ok=True)
    
    # Create and display directory table
    table_data = [[name, str(path)] for name, path in dirs.items()]
    table_data.insert(0, ["root", str(autoclean_dir)])
    
    console.print("\n[bold]autoclean Pipeline Directories:[/bold]")
    table = Table(title="Directory Structure", show_header=True, header_style="bold cyan", show_lines=True)
    table.add_column("Type", style="bold green")
    table.add_column("Path", style="white")
    for row in table_data:
        table.add_row(*row)
    console.print(Panel.fit(table))
    
    logger.info("Directory setup completed")
    console.print("[bold green]✓ Directory setup complete![/bold green]")
        
    return autoclean_dir, dirs["bids"], dirs["metadata"], dirs["clean"], dirs["stage"], dirs["debug"]

def validate_environment_variables() -> tuple[str, str]:
    """Check required environment variables are set and return their values.
    
    Returns:
        tuple[str, str]: AUTOCLEAN_DIR and AUTOCLEAN_CONFIG paths
    
    Raises:
        ValueError: If required environment variables are not set
    """
    autoclean_dir = os.getenv("AUTOCLEAN_DIR")
    if not autoclean_dir:
        logger.error("AUTOCLEAN_DIR environment variable is not set")
        raise ValueError("AUTOCLEAN_DIR environment variable (pipeline output root directory) is not set.")

    autoclean_config = os.getenv("AUTOCLEAN_CONFIG") 
    if not autoclean_config:
        logger.error("AUTOCLEAN_CONFIG environment variable is not set")
        raise ValueError("AUTOCLEAN_CONFIG environment variable (path to autoclean configuration file) is not set.")
        
    return autoclean_dir, autoclean_config

def validate_input_file(unprocessed_file: Union[str, Path]) -> None:
    """Check if the input file exists and is a valid EEG file."""
    logger.info(f"Validating input file: {unprocessed_file}")
    console.print(f"[cyan]Validating input file:[/cyan] {unprocessed_file}")

    file_path = Path(unprocessed_file)
    if not file_path.exists():
        error_msg = f"Input file does not exist: {unprocessed_file}"
        logger.error(error_msg)
        console.print(f"[red]✗ {error_msg}[/red]")
        raise FileNotFoundError(error_msg)

    logger.info(f"Input file validation successful: {unprocessed_file}")
    console.print(f"[green]✓ Input file exists and is accessible[/green]")

def validate_task(task: str, available_tasks: list[str]) -> str:
    if task in available_tasks:
        console.print(f"[green]✓ Valid task: {task}[/green]")
        return task
    else:
        console.print(f"[red]✗ Invalid task: {task}[/red]")
        console.print("[yellow]Allowed values are:[/yellow]")
        for t in available_tasks:
            console.print(f"  • [cyan]{t}[/cyan]")
        raise ValueError(f'Invalid task: {task}. Allowed values are: {available_tasks}')

def validate_eeg_system(eeg_system: str) -> str:
    """Validate that the EEG system montage is supported by MNE.
    
    Args:
        eeg_system: Name of the EEG system montage
        
    Returns:
        str: Validated EEG system montage name
        
    Raises:
        ValueError: If EEG system montage is not supported
    """
    VALID_MONTAGES = {
        # Standard system montages
        'standard_1005': '10-05 system (343+3 locations)',
        'standard_1020': '10-20 system (94+3 locations)', 
        'standard_alphabetic': 'LETTER-NUMBER combinations (65+3 locations)',
        'standard_postfixed': '10-20 system with postfixes (100+3 locations)',
        'standard_prefixed': '10-20 system with prefixes (74+3 locations)',
        'standard_primed': '10-20 system with prime marks (100+3 locations)',
        
        # BioSemi montages
        'biosemi16': 'BioSemi 16 electrodes (16+3 locations)',
        'biosemi32': 'BioSemi 32 electrodes (32+3 locations)', 
        'biosemi64': 'BioSemi 64 electrodes (64+3 locations)',
        'biosemi128': 'BioSemi 128 electrodes (128+3 locations)',
        'biosemi160': 'BioSemi 160 electrodes (160+3 locations)',
        'biosemi256': 'BioSemi 256 electrodes (256+3 locations)',
        
        # EasyCap montages
        'easycap-M1': 'EasyCap 10-05 names (74 locations)',
        'easycap-M10': 'EasyCap numbered (61 locations)',
        
        # EGI/GSN montages
        'EGI_256': 'Geodesic Sensor Net (256 locations)',
        'GSN-HydroCel-32': 'HydroCel GSN with Cz (33+3 locations)',
        'GSN-HydroCel-64_1.0': 'HydroCel GSN (64+3 locations)', 
        'GSN-HydroCel-65_1.0': 'HydroCel GSN with Cz (65+3 locations)',
        'GSN-HydroCel-128': 'HydroCel GSN (128+3 locations)',
        'GSN-HydroCel-129': 'HydroCel GSN with Cz (129+3 locations)',
        'GSN-HydroCel-256': 'HydroCel GSN (256 locations)',
        'GSN-HydroCel-257': 'HydroCel GSN with Cz (257+3 locations)',
        
        # MGH montages
        'mgh60': 'MGH 60-channel cap (60+3 locations)',
        'mgh70': 'MGH 70-channel BrainVision (70+3 locations)',
        
        # fNIRS montages
        'artinis-octamon': 'Artinis OctaMon fNIRS (8 sources, 2 detectors)',
        'artinis-brite23': 'Artinis Brite23 fNIRS (11 sources, 7 detectors)'
    }

    if eeg_system in VALID_MONTAGES:
        console.print(f"[green]✓ Valid EEG system: {eeg_system}[/green]")
        console.print(f"[cyan]Description: {VALID_MONTAGES[eeg_system]}[/cyan]")
        return eeg_system
    else:
        console.print(f"[red]✗ Invalid EEG system: {eeg_system}[/red]")
        console.print("[yellow]Supported montages are:[/yellow]")
        for system, desc in VALID_MONTAGES.items():
            console.print(f"  • [cyan]{system}[/cyan]: {desc}")
        raise ValueError(f'Invalid EEG system: {eeg_system}. Must be one of the supported MNE montages.')

def validate_autoclean_config(config_file: Union[str, Path]) -> dict:
    """Validate the autoclean configuration file.
    
    Args:
        config_file: Path to YAML configuration file
        
    Returns:
        tuple containing:
            - list[str]: Available task names
            - list[str]: Available EEG system names 
            - dict[str, str]: Mapping of tasks to their lossless config files
            
    Raises:
        FileNotFoundError: If config file does not exist
        ValueError: If config file is not a valid YAML file
    """
    config_path = Path(config_file)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    try:
        autoclean_dict = load_config(config_path)
                
    except Exception as e:
        logger.error(f"Invalid YAML configuration file: {str(e)}")
        raise ValueError(f"Invalid YAML configuration file: {str(e)}")

    return autoclean_dict

def main() -> None:
    logger.info("Initializing autoclean pipeline")
    console.print("[bold]Initializing autoclean Pipeline[/bold]")
    
    # Initialize required variables before calling entrypoint
    unprocessed_file = Path("/Users/ernie/Documents/GitHub/spg_analysis_redo/dataset_raw/0006_rest.raw")  
    task = "rest_eyesopen"

    try:
        entrypoint(unprocessed_file, task)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
