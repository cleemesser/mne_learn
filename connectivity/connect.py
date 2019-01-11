# Author: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

import mne
from mne import io
from mne.connectivity import spectral_connectivity, seed_target_indices
from mne.datasets import sample
from mne.time_frequency import AverageTFR

raw_fname = 'bects_raw.fif'
#event_fname = 'bects-eve.fif'

# Setup for reading the raw data
raw = io.read_raw_fif(raw_fname, preload=True)


# events shape n_events, 3)
# The first column specifies the sample number of each event, the second column is ignored, and the third column provides the event "value" this is an int which may have been stored in a stimulus channel so might be significant as a label or if it is changing or increasing (see mne.find_events)
# arbitrarily try two event_ids 1 and 2 as the third item value
event_list = [[int(393.5*200), 0, 1], 
              [int(399.5*200), 1, 2],
              [int(410*200), 2, 1]]
              
events = np.array(event_list)


# Add a bad channel
# raw.info['bads'] += ['MEG 2443']

# Pick MEG gradiometers
picks = mne.pick_types(raw.info,meg=False, eeg=True, stim=False, eog=True)

# for testing lets make this smaller
# picks = picks[3:7]

# Create epochs
ok_event_id, tmin, tmax = [1,2], -0.2, 2.0   # time before and after to include in data to process
epochs = mne.Epochs(raw, events,ok_event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), #reject=dict(grad=4000e-13, eog=150e-6),
                    preload=True)
# to see these epochs do
# epochs.plot()

# Use 'MEG 2343' as seed
seed_ch = 'C3'
picks_ch_names = [raw.ch_names[i] for i in picks]

# Create seed-target indices for connectivity computation
seed = picks_ch_names.index(seed_ch)
targets = np.arange(len(picks))
indices = seed_target_indices(seed, targets)

# Define wavelet frequencies and number of cycles
cwt_freqs = np.arange(7, 30, 2)
cwt_n_cycles = cwt_freqs / 7.

# Run the connectivity analysis using 2 parallel jobs
sfreq = raw.info['sfreq']  # the sampling frequency

# original epcoh          
# con, freqs, times, _, _ = spectral_connectivity(
#     epochs,
#     # indices=indices, # why not do all of the channels :-)
#     method='wpli2_debiased', mode='cwt_morlet', sfreq=sfreq,
#     cwt_freqs=cwt_freqs, cwt_n_cycles=cwt_n_cycles, n_jobs=1)

# con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
#     epochs,
#     # indices=indices, # why not do all of the channels :-)
#     method='wpli2_debiased', mode='cwt_morlet', sfreq=sfreq,
#     cwt_freqs=cwt_freqs, cwt_n_cycles=cwt_n_cycles, n_jobs=1)

con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
    epochs,
    # indices=indices, # why not do all of the channels :-)
    method='pli', mode='multitaper', sfreq=sfreq,
    fmin=12.0, fmax=30.0, faverage=True, n_jobs=1)

# Mark the seed channel with a value of 1.0, so we can see it in the plot
#con[np.where(indices[1] == seed)] = 1.0

# Show topography of connectivity from seed
title = 'WPLI2 - Visual - Seed %s' % seed_ch

layout = mne.find_layout(epochs.info, 'eeg')  # use full layout

# have to figure this out
#tfr = AverageTFR(epochs.info, con, times, freqs, len(epochs))
#tfr.plot_topo(fig_facecolor='w', font_color='k', border='k')

# get rid of extra dimension
con = con[:, :, 0] # make it 2D example: shape  is (25,25)

# remove EOG Pg1 and Pg2

ch_names = epochs.ch_names
idx = [ch_names.index(name) for name in ch_names if not name[:2]=='Pg' ]
con = con[idx][:, idx]

# Now, visualize the connectivity in 3D
from mayavi import mlab  # noqa

mlab.figure(size=(600, 600), bgcolor=(0.5, 0.5, 0.5))

# Plot the sensor locations
sens_loc = [raw.info['chs'][picks[i]]['loc'][:3] for i in idx]
sens_loc = np.array(sens_loc)

pts = mlab.points3d(sens_loc[:, 0], sens_loc[:, 1], sens_loc[:, 2],
                    color=(1, 1, 1), opacity=1, scale_factor=0.005)

# Get the strongest connections
# n_con = 20  # show up to 20 connections
# min_dist = 0.05  # exclude sensors that are less than 5cm apart
# threshold = np.sort(con, axis=None)[-n_con]
# ii, jj = np.where(con >= threshold)
n_con = 23   # show up to 10 connections maybe choose 23
min_dist = 0.05  # exclude sensors that are less than 5cm apart
threshold = np.sort(con, axis=None)[-n_con]
ii, jj = np.where(con >= threshold)


# Remove close connections
con_nodes = list()
con_val = list()
for i, j in zip(ii, jj):
    if linalg.norm(sens_loc[i] - sens_loc[j]) > min_dist:
        con_nodes.append((i, j))
        print('include:', con[i,j])
        con_val.append(con[i, j])

con_val = np.array(con_val)

# Show the connections as tubes between sensors
vmax = np.max(con_val)
vmin = np.min(con_val)
for val, nodes in zip(con_val, con_nodes):
    x1, y1, z1 = sens_loc[nodes[0]]
    x2, y2, z2 = sens_loc[nodes[1]]
    points = mlab.plot3d([x1, x2], [y1, y2], [z1, z2], [val, val],
                         vmin=vmin, vmax=vmax, tube_radius=0.001,
                         colormap='RdBu')
    points.module_manager.scalar_lut_manager.reverse_lut = True


mlab.scalarbar(points, title='Phase Lag Index (PLI)', nb_labels=4)

# Add the sensor names for the connections shown
nodes_shown = list(set([n[0] for n in con_nodes] +
                       [n[1] for n in con_nodes]))

for node in nodes_shown:
    x, y, z = sens_loc[node]
    mlab.text3d(x, y, z, raw.ch_names[picks[node]], scale=0.005,
                color=(0, 0, 0))

view = (-88.7, 40.8, 0.76, np.array([-3.9e-4, -8.5e-3, -1e-2]))
mlab.view(*view)

seed_ch_num = int(np.where(indices[1] == seed)[0])
plt.imshow(con)
plt.title(f'connectivity matrix (seed {seed_ch} is channel {seed_ch_num})')
plt.show()

