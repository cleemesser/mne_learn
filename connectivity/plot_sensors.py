# try to adapt this script for more generic EEG use
# wonder if I can get labels hovering

# does not work yet
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import mne
from mne.viz import plot_alignment
from mayavi import mlab

print(__doc__)

FILENAME = 'bects_raw.fif'

data_path = mne.datasets.sample.data_path()
subjects_dir = data_path + '/subjects'
trans = mne.read_trans(data_path + '/MEG/sample/sample_audvis_raw-trans.fif')
print('trans:', trans)
raw = mne.io.read_raw_fif(FILENAME)
# fig = plot_alignment(raw.info, trans=None, subject='sample', dig=False,
#                      eeg=['original', 'projected'], meg=[],
#                      coord_frame='head', subjects_dir=subjects_dir)

fig = plot_alignment(raw.info, trans=None, subject='sample', dig=False,
                     eeg=['projected'], meg=[],
                     coord_frame='head', subjects_dir=subjects_dir)

for node in nodes_shown:
    x, y, z = sens_loc[node]
    mlab.text3d(x, y, z, raw.ch_names[picks[node]], scale=0.005,
                color=(0, 0, 0))

mlab.view(135, 80)
# TODO: fix trans for this example 
print('clearly need to fix the transformation to head coordinates w/ a rotation in the sagital plane')
