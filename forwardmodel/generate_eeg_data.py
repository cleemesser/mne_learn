# coding: utf-8

# ### Estimate EEG Data without Artifacts

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


import mne
import numpy as np
from mne.datasets import sample
import matplotlib.pyplot as plt
import h5py


# #### Load Forward Model

# In[3]:


data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
info = mne.io.read_info(raw_fname) 
fwd = mne.read_forward_solution("sample_forward_model")


# In[4]:


fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                         use_cps=True)


# In[5]:


leadfield = fwd_fixed['sol']['data']
print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)


# In[9]:


n_dipoles = leadfield.shape[1]
vertices = [src_hemi['vertno'] for src_hemi in fwd_fixed['src']]
# stc = mne.SourceEstimate(1e-9 * np.eye(n_dipoles), vertices, tmin=0., tstep=1)
# leadfield = mne.apply_forward(fwd_fixed, stc, info).data / 1e-9


# #### Use Forward Model to Estimate Data

# In[74]:


n_sensors = 60
n_times = 36000
time_step = .5


# Random source

# In[75]:


z = np.dot(np.random.randn(n_dipoles, n_sensors), np.random.randn(n_sensors, n_times)) * 1e-9


# In[76]:


# n_dipoles = leadfield.shape[1]
vertices = [src_hemi['vertno'] for src_hemi in fwd_fixed['src']]
# z = np.random.randn(n_dipoles, n_dipoles) * 1e-9
stc = mne.SourceEstimate(z, vertices, tmin=0., tstep=time_step)
leadfield_data = mne.apply_forward(fwd_fixed, stc, info).data # / np.sum(z, axis=1)


# In[13]:


leadfield_data.shape


# In[77]:


leadfield = mne.apply_forward(fwd_fixed, stc, info)


# In[97]:


from pylab import rcParams
rcParams['figure.figsize'] = 20, 10


# In[98]:


stacklineplot.show_epoch_centered(x, 0, epoch_width_sec=300, chstart=0, chstop=10, fs=.5, ylabels=range(60), yscale=2.0)
plt.title('Sample Data');


# In[99]:


stacklineplot.show_epoch_centered(x, 100, epoch_width_sec=300, chstart=0, chstop=10, fs=.5, ylabels=range(60), yscale=2.0)
plt.title('Sample Data');


# In[16]:


fig = leadfield.plot(exclude=(), time_unit='s')


# In[17]:


picks = mne.pick_types(leadfield.info, meg=False, eeg=True, eog=False)
leadfield.plot(spatial_colors=True, gfp=True, picks=picks, time_unit='s')
leadfield.plot_topomap(time_unit='s')


# In[18]:


import stacklineplot 


# In[32]:


leadfield.data.shape
