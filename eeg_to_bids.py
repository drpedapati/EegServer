#!/usr/bin/env python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mne",
#     "mne-bids",
#     "pandas",
#     "pathlib",
#     "rich",
#     "pybv",
# ]
# ///

#!/usr/bin/env python

import argparse
import sys
import json
import pandas as pd
import mne
from mne_bids import BIDSPath, write_raw_bids, update_sidecar_json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
import shutil
from mne.io.constants import FIFF

console = Console()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert EEG data files into BIDS format."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--metadata_csv", help="CSV file with BIDS metadata and file paths."
    )
    group.add_argument("--eeg_file", help="Path to a single EEG data file to convert.")
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for BIDS dataset."
    )
    parser.add_argument("--study_name", default="EEG Study", help="Name of the study.")
    parser.add_argument(
        "--line_freq", type=float, default=60.0, help="Power line frequency in Hz."
    )
    parser.add_argument(
        "--task", default="resting", help="Task name for the EEG recording."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing BIDS dataset."
    )
    parser.add_argument(
        "--generate_template",
        action="store_true",
        help="Generate a sample metadata template CSV file and exit.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up the BIDS dataset generated in the output directory.",
    )
    return parser.parse_args()


def generate_template_csv(output_path):
    """
    Generates a sample template CSV file for the EEG to BIDS conversion script.

    Parameters:
    - output_path (str or Path): The directory where the template CSV will be saved.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define the required columns
    columns = [
        "file_path",  # Full path to the EEG data file (FIF format)
        "participant_id",  # Participant identifier (without 'sub-' prefix)
        "session",  # Session identifier (optional)
        "task",  # Task name
        "run",  # Run number (optional)
        "age",  # Age of the participant in years
        "sex",  # Biological sex ('M', 'F', or 'O')
        "group",  # Participant group
        # Add any additional BIDS metadata fields as needed
    ]

    # Create sample data
    sample_data = [
        {
            "file_path": "/path/to/participant1_session1_rest.fif",
            "participant_id": "001",
            "session": "01",
            "task": "rest",
            "run": "01",
            "age": 25,
            "sex": "M",
            "group": "control",
        },
        {
            "file_path": "/path/to/participant2_session1_task.fif",
            "participant_id": "002",
            "session": "01",
            "task": "task",
            "run": "01",
            "age": 30,
            "sex": "F",
            "group": "patient",
        },
        {
            "file_path": "/path/to/participant3_session2_rest.fif",
            "participant_id": "003",
            "session": "02",
            "task": "rest",
            "run": "02",
            "age": 28,
            "sex": "O",
            "group": "control",
        },
    ]

    # Create DataFrame and save as CSV
    df = pd.DataFrame(sample_data, columns=columns)
    csv_file = output_path / "metadata_template.csv"
    df.to_csv(csv_file, index=False)

    console.print(
        f"[green]Sample template CSV file has been created at: {csv_file}[/green]"
    )


def create_dataset_description(output_path, study_name):
    dataset_description = {
        "Name": study_name,
        "BIDSVersion": "1.6.0",
        "DatasetType": "raw",
    }
    with open(output_path / "dataset_description.json", "w") as f:
        json.dump(dataset_description, f, indent=4)
    console.print("[green]Created dataset_description.json[/green]")


def create_participants_json(output_path):
    participants_json = {
        "participant_id": {"Description": "Unique participant identifier"},
        "bids_path": {"Description": "Path to the BIDS file"},
        "file_hash": {"Description": "Hash of the original file"},
        "file_name": {"Description": "Name of the original file"},
        "eegid": {"Description": "Original participant identifier"},
        "age": {"Description": "Age of the participant", "Units": "years"},
        "sex": {
            "Description": "Biological sex of the participant",
            "Levels": {
                "M": "Male",
                "F": "Female",
                "O": "Other",
                "n/a": "Not available",
            },
        },
        "group": {"Description": "Participant group", "Levels": {}},
    }
    with open(output_path / "participants.json", "w") as f:
        json.dump(participants_json, f, indent=4)
    console.print("[green]Created participants.json[/green]")
    
    

def sanitize_participant_id(filename):
    """
    Sanitizes the participant ID extracted from the filename to comply with BIDS conventions.

    Parameters:
    - filename (str): The filename to sanitize.

    Returns:
    - str: A sanitized participant ID.
    """
    import hashlib

    def filename_to_number(filename, max_value=1000000):
        # Create a hash of the filename
        hash_object = hashlib.md5(filename.encode())
        # Get the first 8 bytes of the hash as an integer
        hash_int = int.from_bytes(hash_object.digest()[:8], "big")
        # Use modulo to get a number within the desired range
        return hash_int % max_value

    basename = Path(filename).stem
    participant_id = filename_to_number(basename)
    console.print(f"Unique Number for {basename}: {participant_id}")

    return participant_id


def convert_single_eeg_to_bids(
    file_path,
    output_dir,
    task="rest",
    participant_id=None,
    line_freq=60.0,
    overwrite=False,
    events=None,
    event_id=None,
    study_name="EEG Study"):
    """
    Converts a single EEG data file into BIDS format with default/dummy metadata.

    Parameters:
    - file_path (str or Path): Path to the EEG data file.
    - output_dir (str or Path): Directory where the BIDS dataset will be created.
    - task (str, optional): Task name. Defaults to 'resting'.
    - participant_id (str, optional): Participant ID. Defaults to sanitized basename of the file.
    - line_freq (float, optional): Power line frequency in Hz. Defaults to 60.0.
    - overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
    - study_name (str, optional): Name of the study. Defaults to "EEG Study".
    """
    import hashlib

    console = Console()
    bids_root = Path(output_dir)
    bids_root.mkdir(parents=True, exist_ok=True)

    # Sanitize and set participant ID
    if participant_id is None:
        participant_id = sanitize_participant_id(file_path)
    # subject_id = participant_id.zfill(5)
    subject_id = str(participant_id)

    # Default metadata
    session = None
    run = None
    age = "n/a"
    sex = "n/a"
    group = "n/a"

    bids_path = BIDSPath(
        subject=subject_id,
        session=session,
        task=task,
        run=run,
        datatype="eeg",
        root=bids_root,
        suffix="eeg",
    )

    fif_file = Path(file_path)
    if not fif_file.exists():
        console.print(f"[red]File {fif_file} does not exist. Exiting.[/red]")
        sys.exit(1)

    # Read the raw data
    try:
        raw = mne.io.read_raw_fif(fif_file, preload=False, verbose=False)
        file_hash = hashlib.sha256(fif_file.read_bytes()).hexdigest()
        file_name = fif_file.name
    except Exception as e:
        console.print(f"[red]Failed to read {fif_file}: {e}[/red]")
        sys.exit(1)

    # Prepare additional metadata
    raw.info["subject_info"] = {"id": int(subject_id), "age": None, "sex": sex}

    raw.info["line_freq"] = line_freq

    # Prepare unit for BIDS
    for ch in raw.info["chs"]:
        ch["unit"] = FIFF.FIFF_UNIT_V  # Assuming units are in Volts

    # Additional BIDS parameters
    bids_kwargs = {
        "raw": raw,
        "bids_path": bids_path,
        "overwrite": overwrite,
        "verbose": False,
        "format": "EEGLAB",
        "events": events,
        "event_id": event_id,
    }

    # Write BIDS data
    try:
        write_raw_bids(**bids_kwargs)
        console.print(f"[green]Converted {fif_file.name} to BIDS format.[/green]")
        entries = {"Manufacturer": "Unknown", "PowerLineFrequency": line_freq}
        sidecar_path = bids_path.copy().update(extension=".json")
        update_sidecar_json(bids_path=sidecar_path, entries=entries)
    except Exception as e:
        console.print(f"[red]Failed to write BIDS for {fif_file.name}: {e}[/red]")
        sys.exit(1)

    # Update participants.tsv
    participants_file = bids_root / "participants.tsv"
    if not participants_file.exists():
        participants_df = pd.DataFrame(
            columns=["participant_id", "age", "sex", "group"]
        )
    else:
        participants_df = pd.read_csv(participants_file, sep="\t")

    new_entry = {
        "participant_id": f"sub-{subject_id}",
        "bids_path": bids_path,
        "age": age,
        "sex": sex,
        "group": group,
        "eegid": fif_file.stem,
        "file_name": file_name,
        "file_hash": file_hash,
    }

    participants_df = participants_df._append(new_entry, ignore_index=True)
    participants_df.drop_duplicates(subset="participant_id", keep="last", inplace=True)
    participants_df.to_csv(participants_file, sep="\t", index=False, na_rep="n/a")

    # Create dataset_description.json if it doesn't exist
    dataset_description_file = bids_root / "dataset_description.json"
    if not dataset_description_file.exists():
        create_dataset_description(bids_root, study_name=study_name)

    # Create participants.json if it doesn't exist
    participants_json_file = bids_root / "participants.json"
    if not participants_json_file.exists():
        create_participants_json(bids_root)

    return bids_path


def cleanup_bids_dataset(output_dir):
    """
    Removes the BIDS dataset directory and its contents.

    Parameters:
    - output_dir (str or Path): The BIDS dataset directory to remove.
    """
    output_dir = Path(output_dir)
    if output_dir.exists() and output_dir.is_dir():
        try:
            shutil.rmtree(output_dir)
            console.print(f"[green]Cleaned up BIDS dataset at {output_dir}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to clean up {output_dir}: {e}[/red]")
    else:
        console.print(
            f"[yellow]No BIDS dataset found at {output_dir} to clean up.[/yellow]"
        )


def process_metadata_row(row, bids_root, line_freq=60):
    fif_file = Path(row["file_path"])
    if not fif_file.exists():
        console.print(f"[red]File {fif_file} does not exist. Skipping.[/red]")
        return

    subject_id = str(row["participant_id"]).zfill(5)
    session = str(row.get("session", "")).strip() or None
    task = str(row.get("task", "rest")).strip()
    run = str(row.get("run", "")).strip() or None
    age = row.get("age", "n/a")
    sex = str(row.get("sex", "n/a")).upper()
    group = row.get("group", "n/a")

    bids_path = BIDSPath(
        subject=subject_id,
        session=session,
        task=task,
        run=run,
        datatype="eeg",
        root=bids_root,
        suffix="eeg",
    )

    # Read the raw data
    try:
        raw = mne.io.read_raw_fif(fif_file, preload=False, verbose=False)
    except Exception as e:
        console.print(f"[red]Failed to read {fif_file}: {e}[/red]")
        return

    # Prepare additional metadata
    raw.info["subject_info"] = {
        "id": int(subject_id),
        "age": float(age) if age != "n/a" else None,
        "sex": sex,
    }

    raw.info["line_freq"] = line_freq

    # Prepare unit for BIDS
    for ch in raw.info["chs"]:
        ch["unit"] = FIFF.FIFF_UNIT_V  # Assuming units are in Volts

    # Additional BIDS parameters
    bids_kwargs = {
        "raw": raw,
        "bids_path": bids_path,
        "overwrite": True,
        "verbose": False,
        "format": "BrainVision",
        "events": None,
        "event_id": None,
    }

    # Write BIDS data
    try:
        write_raw_bids(**bids_kwargs)
        console.print(f"[green]Converted {fif_file.name} to BIDS format.[/green]")
        entries = {}
        entries["Manufacturer"] = "EGI"
        entries["PowerLineFrequency"] = line_freq
        sidecar_path = bids_path.copy().update(extension=".json")
        update_sidecar_json(bids_path=sidecar_path, entries=entries)

    except Exception as e:
        console.print(f"[red]Failed to write BIDS for {fif_file.name}: {e}[/red]")
        return

    # Update participants.tsv
    participants_file = bids_root / "participants.tsv"
    if not participants_file.exists():
        participants_df = pd.DataFrame(
            columns=["participant_id", "age", "sex", "group"]
        )
    else:
        participants_df = pd.read_csv(participants_file, sep="\t")

    new_entry = {
        "participant_id": f"sub-{subject_id}",
        "age": age,
        "sex": sex,
        "group": group,
    }

    participants_df = participants_df.append(new_entry, ignore_index=True)
    participants_df.drop_duplicates(subset="participant_id", keep="last", inplace=True)
    participants_df.to_csv(participants_file, sep="\t", index=False, na_rep="n/a")


def convert_eeg_to_bids(metadata_csv, output_dir, study_name, overwrite=False, line_freq=60.0):
    """
    Converts EEG data files into BIDS format based on a provided metadata CSV file.

    Parameters:
    - metadata_csv (str or Path): Path to the CSV file containing BIDS metadata and file paths.
    - output_dir (str or Path): Directory where the BIDS dataset will be created.
    - study_name (str, optional): Name of the study. Defaults to "EEG Study".
    - line_freq (float, optional): Power line frequency in Hz. Defaults to 60.0.
    - overwrite (bool, optional): Whether to overwrite the existing BIDS dataset if it exists. Defaults to False.

    Raises:
    - FileNotFoundError: If the metadata CSV file does not exist.
    - ValueError: If required columns are missing in the metadata CSV.
    - Exception: For other unforeseen errors during the conversion process.
    """
    console = Console()
    metadata_csv = Path(metadata_csv)
    output_dir = Path(output_dir)

    # Validate metadata CSV
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV file {metadata_csv} does not exist.")

    # Handle output directory
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory {output_dir} already exists. Set overwrite=True to overwrite."
            )
        else:
            console.print(
                f"[yellow]Overwriting existing BIDS dataset at {output_dir}[/yellow]"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]Created output directory at {output_dir}[/green]")

    # Read metadata
    metadata_df = pd.read_csv(metadata_csv)
    required_columns = {"file_path", "participant_id"}
    if not required_columns.issubset(metadata_df.columns):
        raise ValueError(f"Metadata CSV must contain columns: {required_columns}")

    # Create BIDS dataset_description.json and participants.json
    create_dataset_description(output_dir, study_name)
    create_participants_json(output_dir)

    console.print(Panel("[bold blue]Starting BIDS Conversion[/bold blue]"))

    # Process each row in the metadata CSV
    for _, row in track(
        metadata_df.iterrows(),
        total=metadata_df.shape[0],
        description="Processing files...",
    ):
        process_metadata_row(row, output_dir, line_freq=line_freq)

    console.print(Panel("[bold green]BIDS Conversion Complete[/bold green]"))


def main():
    args = parse_arguments()

    if args.generate_template:
        generate_template_csv(output_path=".")
        console.print(
            "[green]Template CSV file generated as 'metadata_template.csv'. Please modify it with your data and rerun the script.[/green]"
        )
        sys.exit(0)

    if args.cleanup:
        cleanup_bids_dataset(args.output_dir)
        sys.exit(0)

    if args.eeg_file:
        # Single EEG file conversion
        convert_single_eeg_to_bids(
            file_path=args.eeg_file,
            output_dir=args.output_dir,
            task=args.task,
            participant_id=None,  # Will be derived from filename
            line_freq=args.line_freq,
            overwrite=args.overwrite,
            study_name=args.study_name,
        )
    elif args.metadata_csv:
        # CSV-based conversion
        metadata_csv = Path(args.metadata_csv)
        output_dir = Path(args.output_dir)

        if not metadata_csv.exists():
            console.print(
                f"[red]Metadata CSV file {metadata_csv} does not exist.[/red]"
            )
            sys.exit(1)

        if output_dir.exists() and not args.overwrite:
            console.print(
                f"[red]Output directory {output_dir} already exists. Use --overwrite to overwrite.[/red]"
            )
            sys.exit(1)
        elif output_dir.exists() and args.overwrite:
            pass  # Optionally handle directory backup or cleanup

        output_dir.mkdir(parents=True, exist_ok=True)

        console.print(Panel("[bold blue]Starting BIDS Conversion[/bold blue]"))

        # Read metadata
        metadata_df = pd.read_csv(metadata_csv)
        required_columns = {"file_path", "participant_id"}
        if not required_columns.issubset(metadata_df.columns):
            console.print(
                f"[red]Metadata CSV must contain columns: {required_columns}[/red]"
            )
            sys.exit(1)

        # Create BIDS dataset_description.json and participants.json
        create_dataset_description(output_dir, args.study_name)
        create_participants_json(output_dir)

        # Process each row in the metadata CSV
        for _, row in track(
            metadata_df.iterrows(),
            total=metadata_df.shape[0],
            description="Processing files...",
        ):
            process_metadata_row(row, output_dir, line_freq=args.line_freq)

        console.print(Panel("[bold green]BIDS Conversion Complete[/bold green]"))
    else:
        console.print(
            "[red]Please provide either a metadata CSV file using '--metadata_csv' or a single EEG file using '--eeg_file'.[/red]"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
