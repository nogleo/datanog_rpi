#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 22:36:54 2020

@author: nog
"""
from skimage.restoration import (denoise_wavelet, estimate_sigma)
import pandas as pd
import pywt
from matplotlib.pyplot import plot, subplot, figure, semilogy, loglog, semilogx
import matplotlib.pyplot as plt
import numpy as np
import scipy
from mpl_toolkits import mplot3d
from scipy.signal import filtfilt, butter, buttord, spectrogram

%matplotlib inline
fs = 3330
data = np.load('log_21.npz', allow_pickle=True)
ang = data['arr_0']
gyr0 = data['arr_1']
acc0 = data['arr_2']

data = np.load('./raw/raw_21.npy')
ang = data[:,0]
gyr0 = data[:,1:4]
acc0 = data[:,4:7]

N = len(ang)

t = np.linspace(0.0, N/fs, N)

# %%
# def PSD(data):
#     _l = data.shape[0]plt.ion()
#     if len(data.shape) == 1:
#         data = data.reshape(_l, 1)
#     _n = data.shape[1]
#     d_f = np.zeros(data.shape)
#     d_p = np.zeros(data.shape)
#     for i in range(_n):
#         d_f[:, i] = scipy.fft.fft(data[:, i], _l)
#         d_p[:, i] = d_nd('Noisy', 'Denoised')f[:, i] * np.conj(d_f[:, i]) / _l
#     return d_p[1:int(_l/2)+1]


def PSD(_data):
    _f, _dout = scipy.signal.welch(_data, nperseg=fs, fs=fs, axis=0, scaling='spectrum', average='mean',window='hann',)
    return _dout


def spect(timeDataP, fsMeasure):
    f, t, Sxx = spectrogram(timeDataP, fsMeasure, window='blackmanharris', nperseg=256*2, noverlap=0.9, mode='magnitude')
    plt.pcolormesh(t, f, 20*np.log10(abs(Sxx)), shading='gouraud', cmap=plt.cm.viridis)
    plt.ylim((0, 850))
    plt.colorbar()
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def pulldata(_data):
    
    _l = _data.shape[0]
    if len(_data.shape) == 1:
        _data = _data.reshape(_l, 1)
    _n = _data.shape[1]
    _output = np.zeros(_data.shape)
    for _i in range(_n):
        _aux = _data[:, _i]
        _sig = estimate_sigma(_aux)
        _output[:, _i] = _aux
        _output[:, _i] = denoise_wavelet(_aux, method='VisuShrink', mode='soft', sigma =_sig/2, rescale_sigma=False)
    return _output

# %%
gyr0f = pulldata(gyr0)
acc0f = pulldata(acc0)
angf = pulldata(ang).reshape(N)

vel0 = scipy.integrate.cumtrapz(acc0f, axis=0)

g = np.linalg.norm(acc0,axis=1)
G = np.linalg.norm(acc0f,axis=1)

rot0 = np.array([])        
rot0 =  scipy.integrate.cumtrapz(gyr0f, axis=0)
spect(acc0f[:, 0], fs)
spect(acc0f[:, 1], fs)
spect(acc0f[:, 2], fs)
# spect(acc0[:, 0], fs)
# spect(acc0[:, 1], fs)
# spect(acc0[:, 2], fs)
spect(gyr0f[:, 0], fs)
spect(gyr0f[:, 1], fs)
spect(gyr0f[:, 2], fs)
spect(angf, fs)
# spect(acc1[:, 0], 1660)
# spect(acc1[:, 1], 1660)
# spect(acc1[:, 2], 1660)
# spect(gyr1[:, 0], 1660)
# spect(gyr1[:, 1], 1660)
# spect(gyr1[:, 2], 1660)

fig = figure(1)
subplot(3, 1, 1)
plot(t, acc0[:, 0])
subplot(3, 1, 2)
plot(t, acc0[:, 1])
subplot(3, 1, 3)
plot(t, acc0[:, 2])
subplot(3, 1, 1)
plot(t,  acc0f[:, 0])
subplot(3, 1, 2)
plot(t,  acc0f[:, 1])
subplot(3, 1, 3)
plot(t,  acc0f[:, 2])
plt.show()


figure(2)
subplot(3, 1, 1)
plot(t,  gyr0[:, 0])
subplot(3, 1, 2)
plot(t,  gyr0[:, 1])
subplot(3, 1, 3)
plot(t,  gyr0[:, 2])
subplot(3, 1, 1)
plot(t,  gyr0f[:, 0])
subplot(3, 1, 2)
plot(t,  gyr0f[:, 1])
subplot(3, 1, 3)
plot(t,  gyr0f[:, 2])

figure(3)
plot(t,  ang)
plot(t,  angf)


# figure(4)
# subplot(3, 1, 1)
# plot(acc1[:, 0])
# subplot(3, 1, 2)
# plot(acc1[:, 1])
# subplot(3, 1, 3)
# plot(acc1[:, 2])

# figure(5)
# subplot(3, 1, 1)
# plot(gyr1[:, 0])
# subplot(3, 1, 2)
# plot(gyr1[:, 1])
# subplot(3, 1, 3)
# plot(gyr1[:, 2])

# %%

plt.figure()
plt.plot(acc0[2000:3000,0])
plt.plot(acc0f[2000:3000,0])



#%%
dt = 1/fs
f = range(334)
f = range(int(fs/2)+1)
Acc0f = PSD(acc0f)
Acc0 = PSD(acc0)
Gyr0 = PSD(gyr0)
Ang = PSD(ang)
Gyr0f = PSD(gyr0f)
Angf = PSD(angf)
# Acc1 = PSD(acc1)
# Gyr1 = PSD(gyr1)
axxis = ['x','y','z']
for _i in range(3):
    figure()
    loglog(f, Acc0[:, _i])
    loglog(f, Acc0f[:, _i])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Acceleration [m/s²]')
    plt.title('Axis {}'.format(axxis[_i]))
    figure()
    loglog(f, Gyr0[:, _i])
    loglog(f, Gyr0f[:, _i])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Angular Rate [deg/s]')
    plt.title('Axis {}'.format(axxis[_i]))

figure()
loglog(f, Ang)
loglog(f, Angf)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Angle [deg]')
plt.title('Rotor')
# figure()
# loglog(f, Acc1)
# figure()
# loglog(f, Gyr1)


# #%%

# from scipy import signal
# import matplotlib.pyplot as plt

# b, a = signal.butter(20,1, btype='hp', fs=1660)
# w, h = signal.freqs(b, a)
# plt.semilogx(w, 20 * np.log10(abs(h)))
# plt.title('Chebyshev Type I frequency response (rp=5)')
# plt.xlabel('Frequency [radians / second]')
# plt.ylabel('Amplitude [dB]')
# plt.margins(0, 0.1)
# plt.grid(which='both', axis='both')
# plt.axvline(100, color='green') # cutoff frequency
# plt.axhline(-5, color='green') # rp
# plt.show()
# accf = signal.filtfilt(b,a,acc0f,axis=0)
# Accf = PSD(accf)
#%%
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(acc0[:,0], acc0[:,1], acc0[:,2], marker='o')
ax.scatter3D(acc0f[:,0], acc0f[:,1], acc0f[:,2], marker='^', color='red')
plt.show()

#%%

fig, ax = plt.subplots(3,1, figsize=(8,6))

ax[0].plot(t[3000:5000], acc0f[3000:5000,:])

ax[1].plot(t[3000:5000], gyr0f[3000:5000,:])
ax[2].plot(t[3000:5000], ang[3000:5000])

fig.tight_layout()
plt.show()