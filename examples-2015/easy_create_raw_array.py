import mne
import numpy as np
data = np.zeros((30, 1000))
ch_names = [str(x) for x in range(30)]
info = mne.create_info(ch_names, 1000., 'eeg')
raw = mne.io.RawArray(data, info)

