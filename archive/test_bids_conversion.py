# test_bids_conversion.py

import os
import sys
from pathlib import Path
from eeg_to_bids import convert_single_eeg_to_bids
from rich.console import Console

console = Console()

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test BIDS conversion of a single FIF file.')
    parser.add_argument('fif_file_path', help='Path to the FIF file to convert.')
    parser.add_argument('bids_dir', help='Path to the BIDS directory where the BIDS structure will be formed.')
    parser.add_argument('--task', default='resting', help='Task name for the EEG recording. Default is "resting".')
    parser.add_argument('--line_freq', type=float, default=60.0, help='Power line frequency in Hz. Default is 60.0.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing BIDS data if it exists.')
    args = parser.parse_args()

    # Ensure the FIF file exists
    fif_file_path = Path(args.fif_file_path)
    if not fif_file_path.exists():
        console.print(f"[red]File not found: {fif_file_path}[/red]")
        sys.exit(1)

    # Create BIDS directory if it doesn't exist
    bids_dir = Path(args.bids_dir).resolve()
    bids_dir.mkdir(parents=True, exist_ok=True)

    # Convert the FIF file to BIDS format
    try:
        bids_path = convert_single_eeg_to_bids(
            file_path=str(fif_file_path),
            output_dir=str(bids_dir),
            task=args.task,
            participant_id=None,  # Participant ID will be derived from filename
            line_freq=args.line_freq,
            overwrite=args.overwrite,
            study_name='EEG Study'
        )
        console.print("[green]BIDS conversion successful.[/green]")
        console.print(f"BIDS data saved at: {bids_path}")
        console.print(f"BIDS data saved at: {bids_path.root}")
        
        # Print all available properties of bids_path in a table
        from rich.table import Table
        import json
        
        table = Table(title="BIDS Path Properties")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")
        
        properties = [
            "subject", "session", "task", "acquisition", "run", "processing",
            "recording", "space", "suffix", "datatype", "extension", "check",
            "basename", "directory", "fpath", "entities"
        ]
        
        bids_properties = {}
        for prop in properties:
            value = getattr(bids_path, prop, "N/A")
            if prop == "entities":
                value = str(value)  # Convert dictionary to string
            table.add_row(prop, str(value))
            bids_properties[prop] = str(value)
        
        console.print(table)
        
        # Save the information to a JSON file
        json_filename = f"{fif_file_path.stem}.json"
        with open(json_filename, 'w') as json_file:
            json.dump(bids_properties, json_file, indent=4)
        
        console.print(f"[green]BIDS properties saved to: {json_filename}[/green]")
    except Exception as e:
        console.print(f"[red]Error during BIDS conversion: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()
