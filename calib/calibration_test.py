#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 09:51:07 2020

@author: nog
"""
import numpy as np
import autograd
import autograd.numpy as nap
from numpy.linalg import inv, norm
import scipy.optimize as op


data = np.load('imu1rawdata.npy')

gravity = 9.81
fs = 3330
dt = 1/fs
nsamp = int(5/dt)
gsamp = int(1/dt)

accdata = data[0:6*nsamp, 3:6]


k = np.zeros((3, 3))
b = np.zeros((3, 1))
Ti = np.ones((3, 3))
for i in range(3):
    aup = np.mean(accdata[accdata[:, i] > 1800, :])
    adown = np.mean(accdata[accdata[:, i] < -1800, :])
    Ti[i, i-2] = np.arctan(np.mean(accdata[accdata[:, i] > 1800, i-2])/np.mean(accdata[accdata[:, i] > 1800, i]))
    Ti[i, i-1] = np.arctan(np.mean(accdata[accdata[:, i] > 1800, i-1])/np.mean(accdata[accdata[:, i] > 1800, i]))
    k[i, i] = (aup - adown)/(2*gravity)
    b[i] = (aup + adown)/2
kT = inv(k.dot(inv(Ti)))
aux = np.zeros((6, 3))
for i in range(6):
    for j in range(3):
        aux[i, j] = np.mean(accdata[i*nsamp:(i+1)*nsamp, j])

accm = aux

# %%
def accObj(X):
        NS = nap.array([[X[0], X[6], X[7]], [X[8], X[1], X[9]], [X[10], X[11], X[2]]])
        b = nap.array([[X[3]], [X[4]], [X[5]]])
        sum = 0
        for u in accm:
            sum += (gravity - nap.linalg.norm(NS@(u-b.T).T))**2

        return sum
    
param = np.append(np.append(np.append(kT.diagonal(), b.T), kT[np.tril(kT, -1) != 0]), kT[np.triu(kT, 1) != 0])
jac = autograd.jacobian(accObj)
hes = autograd.hessian(accObj)
res = op.minimize(accObj, param, method='trust-ncg', jac=jac, hess=hes)

res.x
  
    


for i in range(6):
            for j in range(3):
                aux[i, j] = np.mean(acc[i*nsamp:(i+1)*nsamp, j])