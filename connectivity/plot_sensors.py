# try to adapt this script for more generic EEG use
# wonder if I can get labels hovering

# does not work yet
# original Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)
# note: running this with "vglrun code" under turboVNC works
# should see if it also works with emacs
#%%
try:
    get_ipython().run_line_magic('gui','qt5')
except:
    print('running as script')
import os
import mne
import numpy as np
from mne.viz import plot_alignment
from mayavi import mlab

print(__doc__)
print(f'working dir: {os.getcwd()}')
os.getcwd()
#%%
FILENAME = 'bects_raw.fif'

data_path = mne.datasets.sample.data_path()
subjects_dir = data_path + '/subjects'
trans = mne.read_trans(data_path + '/MEG/sample/sample_audvis_raw-trans.fif')
print('trans:', trans)
raw = mne.io.read_raw_fif(FILENAME)
# fig = plot_alignment(raw.info, trans=None, subject='sample', dig=False,
#                      eeg=['original', 'projected'], meg=[],
#                      coord_frame='head', subjects_dir=subjects_dir)
#%%
fig = plot_alignment(raw.info, trans=None, subject='sample', dig=False,
                     eeg=['original', 'projected'], meg=[],
                     coord_frame='head', subjects_dir=subjects_dir)
#%%
chs = raw.info['chs']
print('len(chs)', len(chs))
chs[0]['loc']
print(chs[0].keys())
'coord frame: ', chs[0]['coord_frame']
#%%
nodes_shown = [ii for ii in range(len(chs)) if not np.isnan(chs[ii]['loc'][0]) ]
nodes_shown
#%%
sens_loc = [chs[ii]['loc'][:3] for ii in nodes_shown]
sens_loc = np.array(sens_loc)
sens_loc = 1.3 * sens_loc  # scale it out 
sens_loc
#%%
for node in nodes_shown:
    x, y, z = sens_loc[node]
    mlab.text3d(x, y, z, chs[node]['ch_name'], scale=0.005,
                color=(0, 0, 0))
    print(f'''plotting {chs[node]['ch_name']} {x, y, z}''')
#%% try the mlab interaction
mlab.view(135, 80)
# TODO: fix trans for this example 
print('clearly need to fix the transformation to head coordinates w/ a rotation in the sagital plane')
