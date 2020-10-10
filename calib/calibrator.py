from collections import deque
import autograd.numpy as np
import scipy.optimize as op
import autograd
from numpy.linalg import norm, inv


class CALIBRIZZATTI:
    def __init__(self):
        self.__name__ = "CALIBRIZZATTI"
        self.freq = 1660
        self.dt = 1/self.freq
        
    def calibrate(self, _filename):
        self._nsamp = 10000
        self._gsamps = 2000
        _data = np.load(_filename)
        self.acc_raw = _data[0:6*self._nsamp,3:6]
        self.gyr_raw = _data[:,0:3]
        acc_p = self.calibacc(self.acc_raw)
        np.save('acc_param.npy', acc_p)
        gyr_p = self.calibgyr(self.gyr_raw)
        np.save('gyr_param.npy', gyr_p)
               
        return True
    
    def calibacc(self, _accdata):
        _k = np.zeros((3, 3))
        _b = np.zeros((3, 1))
        _Ti = np.ones((3, 3))
        for _i in range(3):
            _a_up = np.mean(norm(_accdata[_accdata[:, _i] > 1800, :],axis=1))
            _a_down = np.mean(norm(_accdata[_accdata[:, _i] < -1800, :], axis=1))
            _Ti[_i, _i-2] = np.arctan(np.mean(_accdata[_accdata[:, _i] > 1800, _i-2])/np.mean(_accdata[_accdata[:, _i] > 1800, _i]))
            _Ti[_i, _i-1] = np.arctan(np.mean(_accdata[_accdata[:, _i] > 1800, _i-1])/np.mean(_accdata[_accdata[:, _i] > 1800, _i]))
            _k[_i, _i] = (_a_up - _a_down)/(2*9.81)
            _b[_i] = (_a_up + _a_down)/2
        _kT = inv(_k.dot(inv(_Ti)))
        _param = np.append(np.append(np.append(_kT.diagonal(), _b.T), _kT[np.tril(_kT, -1) != 0]), _kT[np.triu(_kT, 1) != 0])
        _acc_m = []
        for _i in range(6):
            _acc_m.append(np.mean(_accdata[_i*self._nsamp:(_i+1)*self._nsamp,:], axis=0))
                    
        _opt = self.optacc(_param, _acc_m, 9.81)
        return _opt
    
        
    def calibgyr(self, _gyrdata):
        _k = np.zeros((3, 3))
        _b = np.mean(_gyrdata[0:6*self._nsamp], axis=0).T
        _kT = np.diag(90/((_gyrdata[6*self._nsamp:]-_b)*self.dt).sum(0))        
        _param = np.append(np.append(np.append(_kT.diagonal(), _b.T), _kT[np.tril(_kT, -1) != 0]), _kT[np.triu(_kT, 1) != 0])
        _opt = self.optgyr(_param, _gyrdata[6*self._nsamp:-1,:], 90)
        return _opt
    
    def transl(self, _data, X):
        _data_out = np.zeros(_data.shape)
        _NS = np.array([[X[0], X[6], X[7]], [X[8], X[1], X[9]], [X[10], X[11], X[2]]])
        _b = np.array([[X[3]], [X[4]], [X[5]]])
        
        for _i in range(len(_data)):
            _data_out[_i] = (_NS@(_data[_i]-_b.T).T).reshape(3,)
        
        return _data_out

    
    def accObj(self, X):
        _NS = np.array([[X[0], X[6], X[7]], [X[8], X[1], X[9]], [X[10], X[11], X[2]]])
        _b = np.array([[X[3]], [X[4]], [X[5]]])
        _sum = 0
        for u in self._datopt:
            _sum += (self._G - np.linalg.norm(_NS@(u-_b.T).T))**2

        return _sum
    
    def gyrObj(self, X):
        _NS = np.array([[X[0], X[6], X[7]], [X[8], X[1], X[9]], [X[10], X[11], X[2]]])
        _b = np.array([[X[3]], [X[4]], [X[5]]])
        
       
        _sum = norm(_NS@((self._datopt-_b)*self.dt).T)
        return (self._G - _sum)**2
    
    def optacc(self, _X, _datopt, _G):
        self._datopt = _datopt
        self._G = _G
        _jac = autograd.jacobian(self.accObj)
        _hes = autograd.hessian(self.accObj)
        _res = op.minimize(self.accObj, _X, method='trust-ncg', jac=_jac, hess=_hes)
        return _res.x
    
    def optgyr(self, _X, _datopt, _G):
        self._datopt = _datopt
        self._G = _G
        _jac = autograd.jacobian(self.gyrObj)
        _hes = autograd.hessian(self.gyrObj)
        _res = op.minimize(self.accObj, _X, method='trust-ncg', jac=_jac, hess=_hes)
        return _res.x