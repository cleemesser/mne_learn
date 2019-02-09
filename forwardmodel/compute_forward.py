
import mne
from mne.datasets import sample
data_path = sample.data_path() # original version

raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
subjects_dir = data_path + '/subjects' # use the sample data as a source
subject = 'sample'  # use the downloaded sample
# The transformation file obtained by coregistration
trans = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'


# setup source space
src_oct6 = mne.setup_source_space(subject, spacing='oct6',
                             subjects_dir=subjects_dir, add_dist=False)

src_oct5 = setup_source_space(subject, 'sample', spacing='oct5', subjects_dir=subject_dir)

src = src_oct6
print(src)

# MEG
conductivity = (0.3,)  # for single layer (MEG)
# EEG
#conductivity = (0.3, 0.006, 0.3)  # for three layers
conductivity3 = (0.3, 0.010, 0.3)  # for three layers

model = mne.make_bem_model(subject=subject, ico=4,
                           conductivity=conductivity,
                           subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)


fwd = mne.make_forward_solution(raw_fname, trans=trans, src=src, bem=bem,
                                meg=True, eeg=False, mindist=5.0, n_jobs=2)
print(fwd)

leadfield = fwd['sol']['data']
print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)

