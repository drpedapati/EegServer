# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "pathlib",
#     "scipy",
#     "reportlab",
#     "rich",
#     "seaborn"
# ]
# ///


# Example usage:
# batch_results = batch_process_eeg_files('/path/to/input/dir', '/path/to/output/dir')
# print(batch_results)

import os
from peakFinderFunction_uv import batch_process_eeg_files

base_path = '/home/ernie/srv/Analysis/Pedapati_projects/AnalysisProjects/AdolescentData/EegServer/autoclean/rest_eyesopen/results'

# Path to Power Spectrogram results
base_path = '/Users/ernie/Documents/GitHub/spg_analysis_redo/results/'
output_dir = os.path.join(base_path, 'peakFinderFunction')
spg_input_dir = os.path.join(base_path, 'eeg_htpCalcRestPower')

# Pediatric FOOOF
try:
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process the files
    results = batch_process_eeg_files(spg_input_dir, output_dir, 
                                    file_filter='eeg_epo_eeg_htpCalcRestPower_spectro.csv')
    print(f"Successfully processed files. Results saved to {output_dir}")
    
except FileNotFoundError as e:
    print(f"Error: Input directory or files not found - {e}")
except PermissionError as e:
    print(f"Error: Permission denied when accessing directories - {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    print("Traceback:")
    import traceback
    traceback.print_exc()

# Adolescent FOOOF
#batch_process_eeg_files(adol_input_dir, output_dir)

# Adult FOOOF
#batch_process_eeg_files(adult_input_dir, output_dir)

