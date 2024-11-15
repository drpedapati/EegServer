# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "autoreject",
#     "mne",
#     "mne-bids",
#     "numpy",
#     "pandas",
#     "pathlib",
#     "rich",
#     "pylossless @ /media/bigdrive/data/Proj_SPG601/EegServer/pylossless",
#     "python-dotenv",
#     "openneuro-py",
#     "eeglabio",
#     "torch",
#     "pybv",
#     "pyyaml"
# ]
# ///

import mne
import mne_bids as mb
import pylossless as ll

import dotenv
from rich.console import Console
from pathlib import Path
import os

console = Console()

dotenv.load_dotenv()

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
    pipeline = ll.LosslessPipeline(config_file)

    raw = mb.read_raw_bids(bids_path, verbose="ERROR", extra_params={"preload": True})


if __name__ == "__main__":
    main()