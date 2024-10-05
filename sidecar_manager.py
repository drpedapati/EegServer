# sidecar_manager.py
import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

SIDECAR_DIR = os.getenv("SIDECAR_DIR")

class SidecarManager:
    def __init__(self, sidecar_dir=SIDECAR_DIR):
        self.sidecar_dir = sidecar_dir
        if not os.path.exists(self.sidecar_dir):
            os.makedirs(self.sidecar_dir)

    def _get_sidecar_path(self, file_path):
        """Generate the sidecar file path based on the original file name."""
        file_name = os.path.basename(file_path)
        file_basename, _ = os.path.splitext(file_name)
        sidecar_name = f"{file_basename}.json"
        return os.path.join(self.sidecar_dir, sidecar_name)

    def create_sidecar(self, file_path):
        """Create a new sidecar JSON file for the given file."""
        sidecar_path = self._get_sidecar_path(file_path)

        sidecar_data = {
            "original_file": os.path.basename(file_path),
            "original_path": os.path.dirname(file_path),
            "fullpath_file": file_path,
            "creation_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "processed": False,
            "ext": os.path.splitext(file_path)[1],
            "basename": os.path.splitext(os.path.basename(file_path))[0],
            "notes": ""
        }

        # Write the sidecar JSON file
        with open(sidecar_path, 'w') as json_file:
            json.dump(sidecar_data, json_file, indent=4)

        print(f"Created sidecar JSON for {sidecar_data['original_file']} at {sidecar_path}")

    def load_sidecar(self, file_path):
        """Load the sidecar JSON data for the given file."""
        sidecar_path = self._get_sidecar_path(file_path)

        if os.path.exists(sidecar_path):
            with open(sidecar_path, 'r') as json_file:
                return json.load(json_file)
        else:
            raise FileNotFoundError(f"No sidecar file found for {file_path}")

    def update_sidecar(self, file_path, updates):
        """Update the sidecar JSON file with new data or changes."""
        sidecar_path = self._get_sidecar_path(file_path)

        if os.path.exists(sidecar_path):
            # Load existing data
            sidecar_data = self.load_sidecar(file_path)

            # Update fields
            sidecar_data.update(updates)

            # Write back updated JSON
            with open(sidecar_path, 'w') as json_file:
                json.dump(sidecar_data, json_file, indent=4)

            print(f"Updated sidecar JSON for {sidecar_data['original_file']} with {updates}")
        else:
            raise FileNotFoundError(f"No sidecar file found to update for {file_path}")

    def add_field_to_sidecar(self, file_path, field_name, field_value):
        """Add a new field to the sidecar JSON file."""
        sidecar_path = self._get_sidecar_path(file_path)

        if os.path.exists(sidecar_path):
            # Load existing data
            sidecar_data = self.load_sidecar(file_path)

            # Add the new field
            if field_name not in sidecar_data:
                sidecar_data[field_name] = field_value

                # Write back updated JSON
                with open(sidecar_path, 'w') as json_file:
                    json.dump(sidecar_data, json_file, indent=4)

                print(f"Added field '{field_name}' to sidecar JSON for {sidecar_data['original_file']}")
            else:
                print(f"Field '{field_name}' already exists in the sidecar for {sidecar_data['original_file']}")
        else:
            raise FileNotFoundError(f"No sidecar file found to add field for {file_path}")

    def mark_processed(self, file_path, processed=True):
        """Mark the sidecar JSON as processed or unprocessed."""
        self.update_sidecar(file_path, {"processed": processed})
        
    def clean_up_raw(self, file_path):
        """Clean up the raw file and corresponding sidecar JSON."""
        sidecar_path = self._get_sidecar_path(file_path)

        if os.path.exists(sidecar_path):
            os.remove(sidecar_path)
            print(f"Deleted sidecar JSON for {os.path.basename(file_path)}")
        else:
            print(f"No sidecar file found to clean up for {file_path}")
