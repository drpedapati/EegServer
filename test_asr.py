# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "asrpy",
#     "mne",
# ]
# ///

import mne
import asrpy

unprocessed_file = "/Users/ernie/Documents/GitHub/EegServer/unprocessed/0006_chirp.raw"
raw = mne.io.read_raw_egi(input_fname=unprocessed_file, preload=False, events_as_annotations=True)
raw.crop(tmin=0, tmax=45)
raw2 = raw.copy()
import asrpy

asr2 = asrpy.ASR(sfreq=raw2.info["sfreq"], cutoff=20)
asr2.fit(raw2)
raw2 = asr2.transform(raw2)

print(raw.info)
print(raw2.info)

raw.plot()
raw2.plot()