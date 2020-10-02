#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 11:26:30 2020

@author: nog
"""
import numpy as np
from numpy.linalg import inv, norm
from matplotlib.pyplot import plot

caldata = np.load('cal_7.npy')



gy0 = caldata[:, 0:3]
ac0 = caldata[:, 3:6]
acc = caldata[0:6000:10, 3:6]
gyr = caldata[0:6000:10, 0:3]
k = np.zeros((3, 3))
b = np.zeros((3, 1))
Ti = np.ones((3, 3))
for _i in range(3):
    _a_up = np.mean(acc[acc[:, _i] > 1800, _i])
    _a_down = np.mean(acc[acc[:, _i] < -1800, _i])
    Ti[_i, _i-2] = np.mean(acc[acc[:, _i] > 1800, _i-2])/np.mean(acc[acc[:, _i] > 1800, _i])
    Ti[_i, _i-1] = np.mean(acc[acc[:, _i] > 1800, _i-1])/np.mean(acc[acc[:, _i] > 1800, _i])
    k[_i, _i] = (_a_up - _a_down)/(2*9.81)
    b[_i] = (_a_up + _a_down)/2

kT = inv(k.dot(inv(Ti)))
b_g = np.mean(gyr, axis=0)
k_g = 1/(np.max(90/((gy0 - b_g)*1/1660), axis=0))
acc_c = np.zeros(ac0.shape)
gyr_c = np.zeros(gy0.shape)
G = np.zeros(len(ac0))
g_c = np.zeros(len(ac0))
g = np.zeros(len(ac0))
for _i in range(len(ac0)):
    g[_i] = np.linalg.norm(ac0[_i])
    acc_c[_i, :] = kT@((ac0[_i, :]-b.transpose()).transpose()).reshape(3)
    gyr_c[_i,:] = k_g*(gy0[_i] - b_g)
    g_c[_i] = np.linalg.norm(acc_c[_i])




a_m = []
G = []
for _i in range(int(len(g_c)/100)):
    a_m.append(np.mean(g_c[_i*100:(_i+1)*100]))
    G.append(((np.mean(g_c[_i*100:(_i+1)*100]))**2+(9.81)**2)**2)


data = np.load('../data/log_10.npy')
Acc = data[:, 4:7]
Gyr = data[:, 1:4]
Acc = (kT@(Acc.T-b)).T
Gyr = k_g*(Gyr-b_g)
