# -*- coding: utf-8 -*-
# 
# get_ipython().run_line_magic('matplotlib', 'inline')
# %matplotlib qt5

# note: could the sample example use the 'mgh60' montage (older cap 60 channels + 3)


print('this file relies upon already having the precomputed "sample_forward_model-fwd.fif')
import mne
import numpy as np

import matplotlib.pyplot as plt
import h5py
import eeghdf
import eegvis

import eegvis.stacklineplot as slplot
from mne.datasets import sample

# plot 10s epochs (multiples in DE)
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10


print('mn version:', mne.__version__)


data_path = sample.data_path()
subjects_dir = data_path + '/subjects'

raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
info = mne.io.read_info(raw_fname) 
fwd = mne.read_forward_solution("sample_forward_model-fwd.fif")


# In[4]:


fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True,use_cps=True)


# In[5]:


leadfield = fwd_fixed['sol']['data']
print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)

leadfield = fwd_fixed['sol']['data']
print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)

n_dipoles = leadfield.shape[1]
n_sensors = leadfield.shape[0]
vertices = [src_hemi['vertno'] for src_hemi in fwd_fixed['src']] # might need these
fs_gen = 200
DURATION= 10.0 # seconds
time_step = 1.0/fs_gen # sample freq = 200 was 0.5
n_times = int(DURATION * fs_gen)  # try 10 s of generation, truncate to integer

# generate random source dipole data
z = np.dot(np.random.randn(n_dipoles, n_sensors), np.random.randn(n_sensors, n_times)) * 1e-9

t = np.arange(0.0,10.0,step=time_step)

# from what I can tell, vertex 841 is in the left occipital region
# but this is not the same as the z[841] value oh well
stimulate_sites = [841, 1170, 1329] # these seem to be in the central region dorsally in lh


# notes on these somewhat arbitarary values: 
# a sine wave with a coef
# of 5*1e-8 can be seen on the brain activity view but not really n
# the EEG sensor view. If make this 10 times bigger at 5*1e-7
# then it sin wave shows up on multiple electrodes
# at 1e-6 it shows up everywere. Note these particular locations are near or in sulci
source_scale = 5e-7
z[stimulate_sites,:] = z[stimulate_sites,:] +  source_scale*np.sin(2*np.pi*10.0*t)

# determine vertex number of ??? something in the fwd_fixed solution
# vertices = [src_hemi['vertno'] for src_hemi in fwd_fixed['src']]
#len(vertices)
#vertices

# SourceEstimate(data, vertices=None, tmin=None, tstep=None,
#               subject=None, verbose=None
# data: array of shape (n_dipoles, n_times)
#       | 2-tuple (kernel, sens_data)
# vertices: list of two arrays with vertex numbers corresponding to the data 

srcest = mne.SourceEstimate(z, vertices, tmin=0., tstep=time_step)
gen_eeg = mne.apply_forward(fwd_fixed, srcest, info)# / np.sum(z, axis=1)

gen_eeg.data.shape

fig = gen_eeg.plot(exclude=(), time_unit='s')

picks = mne.pick_types(gen_eeg.info, meg=False, eeg=True, eog=False)
gen_eeg.plot(spatial_colors=True, gfp=True, picks=picks, time_unit='s')
gen_eeg.plot_topomap(time_unit='s')



signals = gen_eeg.data
electrode_labels = list(range(n_sensors))
ch0, ch1 = (0,59)
DE = 1 # how many 10s epochs to display
epoch = 0; ptepoch = 10*int(fs_gen)
dp = 0 # int(0.5*ptepoch) # offset 
slplot.stackplot(signals[ch0:ch1,epoch*ptepoch+dp:(epoch+DE)*ptepoch+dp],seconds=DE*10.0, ylabels=electrode_labels[ch0:ch1], yscale=0.3)

gen_eeg.plot_sensors(show_names=True)

## this requires 3D graphics
import numpy as np  # noqa
from mayavi import mlab  # noqa
from surfer import Brain  # noqa


def show_src_grid():
    brain = Brain('sample', 'lh', 'inflated', subjects_dir=subjects_dir)
    surf = brain.geo['lh']

    src = fwd['src'] # get the source space from stored forward solution
    vertidx = np.where(src[0]['inuse'])[0]

    mlab.points3d(surf.x[vertidx], surf.y[vertidx],
                  surf.z[vertidx], color=(1, 1, 0), scale_factor=1.5)

def show_vertices_on_brain(vertidx=[], scale_factor=1.5):
    if not vertidx:
        print('need to specify a list of vertices')
        return
    brain = Brain('sample', 'lh', 'inflated', subjects_dir=subjects_dir)
    surf = brain.geo['lh']

    src = fwd['src'] # get the source space from stored forward solution


    mlab.points3d(surf.x[vertidx], surf.y[vertidx],
                  surf.z[vertidx], color=(1, 1, 0), scale_factor=scale_factor)
    
# try to view the srcest activity overtime (use the control bar to increment the time)
srcest.plot(subject='sample', surface='inflated', hemi='lh',  subjects_dir=subjects_dir, time_viewer=True, alpha=0.7)     
