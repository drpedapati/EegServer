# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pyqt5",
#     "mne",
#     "matplotlib",
# ]
# ///
# ==============================================================================
# Demonstration of Using the New MNE Browser Based on PyQtGraph
# ==============================================================================
# This section of the code demonstrates the use of the new MNE browser for plotting 2D data.
# The new backend, based on pyqtgraph and Qt's Graphics View Framework, offers an alternative
# for visualizing EEG data interactively. This development was initiated during the Google Summer
# of Code 2021.
#
# To utilize this new browser, ensure that you have the latest version of MNE and the mne-qt-browser
# package installed. You can install or update these packages using pip:
# pip install -U mne mne-qt-browser

import mne
import matplotlib

# Specify the path to the resting state EEG data file
# Parameters:
#   eeg_file: String indicating the full path to the EEG data file in EEGLAB .set format.

eeg_file = '/Users/ernie/Documents/GitHub/EegServer/paradigms/resting_eyesopen/processed/fif_EGI128/S01_IMPORT/0006_rest_raw.fif'
try:
    raw = mne.io.read_raw_fif(eeg_file)
except Exception as e:
    print(f"Error reading epochs: {e}")

# create epochs
epochs = mne.make_fixed_length_epochs(raw, duration=2)
epochs.plot(block=True)

# For updated installation: pip install -U mne-qt-browser
# More information available at: https://github.com/mne-tools/mne-qt-browser
matplotlib.use('Qt5Agg')  # Switching backend to Qt5 for compatibility
raw.plot(block=True)
