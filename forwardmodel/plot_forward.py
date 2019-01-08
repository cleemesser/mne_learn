# -*- coding: utf-8 -*-

# In[1]:


## get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# from mayavi import mlab
# mlab.init_notebook()
## does not work with vglrun jupyter notebook
## does run with jupyter notebook but then 


# In[3]:


# %load plotforward.py
# going through https://martinos.org/mne/dev/auto_tutorials/plot_forward.html
# conda activate mne 
# the sample data relies upon environment variable
# MNE_DATASETS_SAMPLE_PATH ->
# downloading sample data now

import mne
from mne.datasets import sample
data_path = sample.data_path() # original version
# MNE_DATASETS_SAMPLE_PATH = os.environ['MNE_DATASETS_SAMPLE_PATH']
#MNE_DATASETS_SAMPLE_PATH = '/usr/local/MNE-C/data'
#data_path = MNE_DATA_ROOT + '/MNE-sample-data'


# In[4]:


# the raw file containing the channel location + types
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
# The paths to Freesurfer reconstructions
# subjects_dir = data_path + '/subjects' # subjects_dir = '/usr/local/freesurfer/stable5/subjects'
freesurfer_path = '/usr/local/freesurfer/stable5'
subjects_dir = data_path + '/subjects' # use the sample data as a source
# subjects_dir = freesurfer_path + '/subjects'
subject = 'sample'  # use the downloaded sample
#subject = 'bert'  # use the bert example inclued with freesurfer


# In[5]:


mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir,
                 brain_surfaces='white', orientation='coronal')


# In[6]:


########

# The transformation file obtained by coregistration
trans = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'

info = mne.io.read_info(raw_fname)
# Here we look at the dense head, which isn't used for BEM computations but
# is useful for coregistration.


# In[7]:


info


# In[ ]:


mne.viz.plot_alignment(info, trans, subject=subject, dig=True,
                       meg=['helmet', 'sensors'], subjects_dir=subjects_dir,
                       surfaces='head-dense')


# In[ ]:

# convenience script: mne_setup_forward_model
# https://martinos.org/mne/stable/manual/c_reference.html#mne-setup-forward-model


src = mne.setup_source_space(subject, spacing='oct6',
                             subjects_dir=subjects_dir, add_dist=False)
print(src)


mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir,
                 brain_surfaces='white', src=src, orientation='coronal')


import numpy as np  # noqa
from mayavi import mlab  # noqa
from surfer import Brain  # noqa

brain = Brain('sample', 'lh', 'inflated', subjects_dir=subjects_dir)
surf = brain.geo['lh']

vertidx = np.where(src[0]['inuse'])[0]

mlab.points3d(surf.x[vertidx], surf.y[vertidx],
              surf.z[vertidx], color=(1, 1, 0), scale_factor=1.5)
