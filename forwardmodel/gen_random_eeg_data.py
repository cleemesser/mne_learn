# -*- coding: utf-8 -*-
#
# get_ipython().run_line_magic('matplotlib', 'inline')
# %matplotlib qt5
print(
    'this file relies upon already having the precomputed "sample_forward_model-fwd.fif'
)
import mne
import numpy as np

import matplotlib.pyplot as plt
import h5py
import eeghdf
import eegvis

import eegvis.stacklineplot as slplot
from mne.datasets import sample

print("mn version:", mne.__version__)


data_path = sample.data_path()
raw_fname = data_path + "/MEG/sample/sample_audvis_raw.fif"
info = mne.io.read_info(raw_fname)
fwd = mne.read_forward_solution("sample_forward_model-fwd.fif")


# In[4]:


fwd_fixed = mne.convert_forward_solution(
    fwd, surf_ori=True, force_fixed=True, use_cps=True
)


# In[5]:


leadfield = fwd_fixed["sol"]["data"]
print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)

leadfield = fwd_fixed["sol"]["data"]
print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)

n_dipoles = leadfield.shape[1]
n_sensors = leadfield.shape[0]
vertices = [src_hemi["vertno"] for src_hemi in fwd_fixed["src"]]  # might need these
fs_gen = 200
DURATION = 10.0  # seconds
time_step = 1.0 / fs_gen  # sample freq = 200 was 0.5
n_times = int(DURATION * fs_gen)  # try 10 s of generation, truncate to integer

# generate random source dipole data
z = (
    np.dot(np.random.randn(n_dipoles, n_sensors), np.random.randn(n_sensors, n_times))
    * 1e-9
)

# determine vertex number of ??? something in the fwd_fixed solution
# vertices = [src_hemi['vertno'] for src_hemi in fwd_fixed['src']]
# len(vertices)
# vertices

srcest = mne.SourceEstimate(z, vertices, tmin=0.0, tstep=time_step)
gen_eeg = mne.apply_forward(fwd_fixed, srcest, info)  # / np.sum(z, axis=1)

gen_eeg.data.shape

fig = gen_eeg.plot(exclude=(), time_unit="s")

picks = mne.pick_types(gen_eeg.info, meg=False, eeg=True, eog=False)
gen_eeg.plot(spatial_colors=True, gfp=True, picks=picks, time_unit="s")
gen_eeg.plot_topomap(time_unit="s")

# plot 10s epochs (multiples in DE)
from pylab import rcParams

rcParams["figure.figsize"] = 20, 10


signals = gen_eeg.data
electrode_labels = list(range(n_sensors))
ch0, ch1 = (0, 19)
DE = 1  # how many 10s epochs to display
epoch = 0
ptepoch = 10 * int(fs_gen)
dp = 0  # int(0.5*ptepoch) # offset
slplot.stackplot(
    signals[ch0:ch1, epoch * ptepoch + dp : (epoch + DE) * ptepoch + dp],
    seconds=DE * 10.0,
    ylabels=electrode_labels[ch0:ch1],
    yscale=0.3,
)
