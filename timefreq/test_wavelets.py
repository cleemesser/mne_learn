# -*- coding: utf-8 -*-
# test PyWavelets
#%%
import numpy as np
import matplotlib.pyplot as plt

import pywt
import pywt.data

import eeghdf
import mne
import mne.io

#%%
f0 = 0.3
t = np.arange(0,10,0.1)
s1 = np.sin(2*np.pi*f0*t)
#wtype = 'db2'
wtype = 'sym3'
ws1a, ws1d = pywt.dwt(s1, wtype)
#%%
fig = plt.figure()
plt.plot(t,s1)
plt.show()
#%%
fig2 = plt.figure()
plt.plot(ws1a)
plt.figure()
plt.plot(ws1d)

#%%
# invert
recon = pywt.idwt(ws1a, ws1d, wtype)

plt.plot(recon)

#%%
error = np.sqrt((recon - s1)**2)
plt.plot(error)
#%%
f'error.max(): {error.max()}'


#%%
#raw = mne.io.read_raw_fif('connectivity/bects_raw.fif', preload=True)
print('will need to change this filename to appropriate one')
FILENAME = r'C:/Users/clee/code/eegml/eeg-hdfstorage/data/spasms.eeghdf'
hf = eeghdf.Eeghdf(FILENAME)
#%%

arr = hf.phys_signals[5, 3000:3000+10*200]

#%%
plt.plot(arr)
plt.show()

#%%
print(f'find transformd and coef')
ch5A, ch5D = pywt.dwt(arr, wtype)

#%%
plt.subplot(2,1,1)
plt.plot(ch5A)
plt.subplot(2,1,2)
plt.plot(ch5D)



#%%
print('now try inverting, and checker error again')
recon_arr = pywt.idwt(ch5A, ch5D, wtype)
error2 = np.sqrt((recon_arr - arr)**2)
plt.subplot(3,1,1)
plt.plot(arr)
plt.subplot(3,1,2)
plt.plot(recon_arr)
plt.subplot(3,1,3)
plt.plot(error)
plt.show()
#%%
