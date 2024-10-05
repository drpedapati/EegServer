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
from pathlib import Path

# Specify the path to the resting state EEG data file
# Parameters:
#   eeg_file: String indicating the full path to the EEG data file in EEGLAB .set format.

base_dir = Path('/Users/ernie/Documents/GitHub/EegServer/')

raw_file = base_dir / 'autoclean/rest_eyesopen/debug/0354_rest_raw.fif'
stage1 = base_dir / 'autoclean/rest_eyesopen/debug/0354_rest_raw_preproc.fif'
stage2 = base_dir / 'autoclean/rest_eyesopen/debug/0354_rest_raw_pipeline.fif'
stage3 = base_dir / 'autoclean/rest_eyesopen/debug/0354_rest_postcomp_raw.fif'
stage4 = base_dir / 'autoclean/rest_eyesopen/debug/0354_rest_postcomp_epo.fif'

raw = mne.io.read_raw_fif(stage2, verbose=True)
raw.info['bads']
raw.plot(block=True)

#stage1eeg = mne.read_epochs(stage4, verbose=True)
#stage1eeg.plot(block=True)

# stage2eeg = mne.read_epochs(stage2, verbose=True)
# stage2eeg.plot(block=False)

# stage3eeg = mne.read_epochs(stage3, verbose=True)
# stage3eeg.plot(block=False)

# stage4eeg = mne.read_epochs(stage4, verbose=True)
# stage4eeg.plot(block=False)

# For updated installation: pip install -U mne-qt-browser
# More information available at: https://github.com/mne-tools/mne-qt-browser
# matplotlib.use('Qt5Agg')  # Switching backend to Qt5 for compatibility
