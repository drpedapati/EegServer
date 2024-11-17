# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "autoreject",
#     "mne",
#     "mne-bids",
#     "numpy",
#     "pandas",
#     "pathlib",
#     "rich",
#     "pylossless @ git+https://github.com/drpedapati/pylossless.git",
#     "python-dotenv",
#     "openneuro-py",
#     "eeglabio",
#     "torch",
#     "pybv",
#     "pyyaml",
#     "PySide6",
#     "mne-qt-browser",
#     "mne-icalabel[all]"
# ]
# ///

import mne
import mne_bids as mb
import pylossless as ll
from mne_icalabel.gui import label_ica_components
import dotenv
from rich.console import Console
from pathlib import Path
import os
from mne.preprocessing import ICA

from mne_icalabel import label_components

console = Console()

dotenv.load_dotenv(override=True)
# Add debug mode flag
DEBUG_MODE = True


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

    bids_path = mb.BIDSPath(root=str(bids_dir))
    
    # Add required BIDS entities
    bids_path = bids_path.update(
        subject='400257',      # Required
        datatype='eeg',    # Specify the data type (eeg, meg, ieeg, etc.)
        task='rest',       # Task name if applicable
    )
    
    
    breakpoint()
    
    raw = mb.read_raw_bids(bids_path, verbose="ERROR", extra_params={"preload": True})
    ica_path = "/Volumes/bigdrive/data/Proj_SPG601/EegServer/autoclean/rest_eyesopen/bids/derivatives/pylossless/sub-400257/eeg/sub-400257_task-rest_ica2_ica.fif"
    ica = mne.preprocessing.read_ica(ica_path)
    ica.plot_sources(raw, show_scrollbars=False, show=True)
    filt_raw = raw.copy().filter(l_freq=1, h_freq=100)
    filt_raw.set_eeg_reference("average")
    labels = ic_labels["labels"]
    exclude_idx = [
        idx for idx, label in enumerate(labels) if label not in ["brain", "other"]
    ]
    print(f"Excluding these ICA components: {exclude_idx}")
    pipeline = ll.LosslessPipeline(config_file)
        
    import pandas as pd

    # Read the IC labels TSV file
    ic_labels_path = "/Volumes/bigdrive/data/Proj_SPG601/EegServer/autoclean/rest_eyesopen/bids/derivatives/pylossless/sub-400257/eeg/sub-400257_task-rest_iclabels.tsv"
    ic_labels_df = pd.read_csv(ic_labels_path, sep='\t')

    # Find indices where confidence > 0.3 and ic_type is not brain or other
    mask = (ic_labels_df['confidence'] > 0.3) & (~ic_labels_df['ic_type'].isin(['brain', 'other']))
    high_conf_idx = ic_labels_df[mask].index.tolist()
    print(f"Components with confidence > 0.3 and not brain/other: {high_conf_idx}")
    ica.plot_overlay(filt_raw, exclude=high_conf_idx)
    # ica.apply() changes the Raw object in-place, so let's make a copy first:
    reconst_raw = filt_raw.copy()
    ica.apply(reconst_raw, exclude=exclude_idx)
    reconst_raw.plot(show_scrollbars=False, show=True)
    filt_raw.plot(show_scrollbars=False, show=True)

if __name__ ==  "__main__":
    main()