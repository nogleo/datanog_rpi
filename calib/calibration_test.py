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
import scipy.integrate as intg
from matplotlib.pyplot import plot


data = np.load('imu01rawdata.npy')
# param = np.load('imu01rawdata.npy')

gravity = 1
fs = 3330
dt = 1/fs
nsamp = int(5/dt)
gsamp = int(2/dt)

accdata = data[0:6*nsamp, 3:6]
gyrdata = data[:, 0:3]

k = np.zeros((3, 3))
b = np.zeros((3))
Ti = np.ones((3, 3))
accm = np.zeros((6, 3))
for i in range(6):
    for j in range(3):
        accm [i, j] = np.mean(accdata[i*nsamp:(i+1)*nsamp, j])



# %%        
for i in range(3):
    _max = accm[:,i].max(0)
    _min = accm[:,i].min(0)
    k[i, i] = (_max - _min)/ (2*gravity)
    b[i] = (_max + _min)/2    
    Ti[i, i-2] = np.arctan(accm[accm[:,i].argmax(0),i-2] / _max)
    Ti[i, i-1] = np.arctan(accm[accm[:,i].argmax(0),i-1] / _max)

kT = inv(k.dot(inv(Ti)))
 
# %%
def accObj(X):
        NS = nap.array([[X[0], X[6], X[7]], [X[8], X[1], X[9]], [X[10], X[11], X[2]]])
        b = nap.array([X[3], X[4], X[5]])
        sum = 0
        for u in accm:
            sum += (gravity - nap.linalg.norm(NS@(u-b).T))**2

        return sum

def transf(data, X):
        NS = nap.array([[X[0], X[6], X[7]], [X[8], X[1], X[9]], [X[10], X[11], X[2]]])
        b = nap.array([X[3], X[4], X[5]])
        return (NS@(data-b).T).T
    # %%
param = np.append(np.append(np.append(kT.diagonal(), b.T), kT[np.tril(kT, -1) != 0]), kT[np.triu(kT, 1) != 0])
jac = autograd.jacobian(accObj)
hes = autograd.hessian(accObj)
res = op.minimize(accObj, param, method='trust-ncg', jac=jac, hess=hes)

dparam = param - res.x

accc = transf(accdata, res.x)  


# %%
rotation = 90

gyr_s = gyrdata[0:6*nsamp,:]
gyr_d = gyrdata[6*nsamp:,:] 
b = np.mean(gyr_s, axis=0)
gyr_r = gyrdata[6*nsamp:,:] - b
# %%
ang = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        ang[i, j] = np.abs(intg.trapz(gyr_r[gsamp*i:gsamp*(i+1), j], dx=dt))

n = ang.argmax(axis=0)

rates = np.zeros((gsamp,3))
for i in range(3):
    rates[:,i] = gyr_d[gsamp*n[i]:gsamp*(n[i]+1), i]
# %%
k = np.zeros((3,3))
k[:,0] = ang[:,ang[0].argmax()]
k[:,1] = ang[:,ang[1].argmax()]
k[:,2] = ang[:,ang[2].argmax()]


kT = np.diag([90,90,90])@inv(k)

# %%
    
def gyrObj(Y):
    NS = nap.array([[Y[0], Y[6], Y[7]], [Y[8], Y[1], Y[9]], [Y[10], Y[11], Y[2]]])
    b = nap.array([Y[3], Y[4], Y[5]])
    sum = 0
    for u in rates:
        sum += NS@(u-b).T*dt
       
    
    return (90 - nap.abs(sum)).sum()**2

# %%
param = np.append(np.append(np.append(kT.diagonal(), b.T), kT[np.tril(kT, -1) != 0]), kT[np.triu(kT, 1) != 0])
jac = autograd.jacobian(gyrObj)
hes = autograd.hessian(gyrObj)
res = op.minimize(gyrObj, param, method='trust-ncg', jac=jac, hess=hes)

dparam = param - res.x
