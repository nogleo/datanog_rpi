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
from scipy.signal import filtfilt, butter, buttord, spectrogram
n, wn = buttord([1.5/1660, 720/1660], [0.1/1660, 750/1660], 0.0175, 1, fs=1660)
#df = pd.read_csv('log_7.csv', delimiter=',', header=None)
# data = []
# data = df.values
fs = 1660
data = np.load('log_11.npy')
b, a = scipy.signal.bessel(2, 2/fs, btype='high')
N = len(data)

t = np.linspace(0.0, N/fs, N)


def PSD(data):
    _l = data.shape[0]
    if len(data.shape) == 1:
        data = data.reshape(_l, 1)
    _n = data.shape[1]
    d_f = np.zeros(data.shape)
    d_p = np.zeros(data.shape)
    for i in range(_n):
        d_f[:, i] = scipy.fft.fft(data[:, i], _l)
        d_p[:, i] = d_f[:, i] * np.conj(d_f[:, i]) / _l
    return d_p[1:int(_l/2)+1]


def spect(timeDataP, fsMeasure):
    f, t, Sxx = spectrogram(timeDataP, fsMeasure, window='blackmanharris', nperseg=256*1, noverlap=0.9, mode='magnitude')
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
        _aux = scipy.signal.filtfilt(b, a, _data[:, _i], axis=0)
        _aux = _data[:, _i]
        _sig = estimate_sigma(_aux)
        _output[:, _i] = denoise_wavelet(_aux, method='VisuShrink', mode='hard', sigma =_sig/4, rescale_sigma=True)
        _output[:, _i] = _aux
    return _output


#data = pulldata(data)

ang = data[:, 0]
gyr0 = data[:, 1:4]
acc0 = data[:, 4:7]
gyr1 = data[:, 7:10]
acc1 = data[:, 10:13]

a0 = np.sqrt(acc0[:, 0]**2 + acc0[:, 1]**2 + acc0[:, 2]**2)
a1 = np.sqrt(acc1[:, 0]**2 + acc1[:, 1]**2 + acc1[:, 2]**2)
figure()
plot(a0)
figure()
plot(a1)

cA, cD = pywt.dwt(a0, wavelet='bior6.8', mode='symmetric')
ccc = pywt.wavedec(a0, 'bior5.5', level=5)

C = PSD(cA)

# spect(acc0[:, 0], 1660)
# spect(acc0[:, 1], 1660)
# spect(acc0[:, 2], 1660)
# spect(gyr0[:, 0], 1660)
# spect(gyr0[:, 1], 1660)
# spect(gyr0[:, 2], 1660)
# spect(ang, 1660)
# spect(acc1[:, 0], 1660)
# spect(acc1[:, 1], 1660)
# spect(acc1[:, 2], 1660)
# spect(gyr1[:, 0], 1660)
# spect(gyr1[:, 1], 1660)
# spect(gyr1[:, 2], 1660)

figure(1)
subplot(3, 1, 1)
plot(acc0[:, 0])
subplot(3, 1, 2)
plot(acc0[:, 1])
subplot(3, 1, 3)
plot(acc0[:, 2])

figure(2)
subplot(3, 1, 1)
plot(gyr0[:, 0])
subplot(3, 1, 2)
plot(gyr0[:, 1])
subplot(3, 1, 3)
plot(gyr0[:, 2])

figure(3)
plot(ang)


figure(4)
subplot(3, 1, 1)
plot(acc1[:, 0])
subplot(3, 1, 2)
plot(acc1[:, 1])
subplot(3, 1, 3)
plot(acc1[:, 2])

figure(5)
subplot(3, 1, 1)
plot(gyr1[:, 0])
subplot(3, 1, 2)
plot(gyr1[:, 1])
subplot(3, 1, 3)
plot(gyr1[:, 2])

# %%





N = len(data)
dt = 1/fs
f = (fs/N) * np.arange(1, N/2 + 1)
Acc0 = PSD(acc0)
Gyr0 = PSD(gyr0)
Ang = PSD(ang)
Acc1 = PSD(acc1)
Gyr1 = PSD(gyr1)
figure()
loglog(f, Acc0)
figure()
loglog(f, Gyr0)

figure()
loglog(f, Ang)

figure()
loglog(f, Acc1)
figure()
loglog(f, Gyr1)


#%%

