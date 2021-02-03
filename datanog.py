import os, gc
from struct import unpack
#import RPi.GPIO as GPIO
import time
from collections import deque
import numpy as np
import autograd.numpy as nap
import scipy.optimize as op
import scipy.integrate as intg
import autograd
from numpy.linalg import norm, inv
import smbus
import sched, time
import json

bus = smbus.SMBus(1)
'''GPIO.setmode(GPIO.BCM)
GPIO.setup(18,GPIO.OUT)
GPIO.setup(23,GPIO.OUT)
GPIO.setup(24,GPIO.OUT)
GPIO.output(18,1)
GPIO.output(23,1)
GPIO.output(24,1)

GPIO.setup(7,GPIO.IN)
GPIO.setup(11,GPIO.IN)
GPIO.setup(13,GPIO.IN)
GPIO.setup(15,GPIO.IN)'''

ODR_POWER_DOWN = 0
ODR_12_5_HZ = 1
ODR_26_HZ = 2
ODR_52_HZ = 3
ODR_104_HZ = 4
ODR_208_HZ = 5
ODR_416_HZ = 6
ODR_833_HZ = 7
ODR_1_66_KHZ = 8
ODR_3_33_KHZ = 9
ODR_6_66_KHZ = 10

ACC_SCALE_2G = 0
ACC_SCALE_4G = 2
ACC_SCALE_8G = 3
ACC_SCALE_16G = 1

GYRO_SCALE_250DPS = 0
GYRO_SCALE_500DPS = 1
GYRO_SCALE_1000DPS = 2
GYRO_SCALE_2000DPS = 3

class DATANOG:
    def __init__(self):
        self.__name__ = "DATANOG"
        self.gravity = 9.81
        self.rotation = 90
        self.sampfreq = 3330
        self.dt = 1/self.sampfreq
        self.led = (18, 23, 24)
        self._sensors = ['AS5600', 'LSM6DS3-0', 'LSM6DS3-1', 'ADS1015']

        # device address for LSM6DS3.1 AS5600 LSM6DS3.2 ADC
        self.bus_addr = self.i2cscan()
        self.settings()
    
    def i2cscan(self):
        bus_addr = []
        
        for device in range(128):
            try:
                bus.read_byte(device)
                bus_addr.append(device)
            except: # exception if read_byte fails
                pass
        
        return bus_addr
    
    def settings(self):
         # linear acceleration sensitivity
        self.ang_sensitivity = 0.087890625  #deg/LSB
        self.accodr = ODR_3_33_KHZ
        self.accscale = ACC_SCALE_16G
        self.gyroodr = ODR_3_33_KHZ
        self.gyroscale = GYRO_SCALE_2000DPS
        

        # ODR
        mask = (self.accodr << 4) | (self.accscale << 2) #| current_reg_data
        bus.write_byte_data(self.bus_addr[-1], 0x10, mask)
        if len(self.bus_addr) > 2:
            bus.write_byte_data(self.bus_addr[-2], 0x10, mask)
        
        
        # Scale
        mask = (self.gyroodr << 4) | (self.gyroscale << 2) 
        bus.write_byte_data(self.bus_addr[-1], 0x11, mask)
        if len(self.bus_addr) > 2:
            bus.write_byte_data(self.bus_addr[-2], 0x11, mask)
        
        # Block Data Read
        bus.write_byte_data(self.bus_addr[-1], 0x12, 0x44)
        if len(self.bus_addr) > 2:
            bus.write_byte_data(self.bus_addr[-2], 0x12, 0x44)
            bus.write_
            
        # ADC config    
        #bus.write_i2c_block_data(self.bus_addr[-3], 0x01, [0x2, 0x43])
    
    

        
    def log(self, _data):
        gc.collect()      
        _filename = 'log_'+str(len(os.listdir('data')))
        _sensname = 'imu00'
        param = np.load('./sensors/'+_sensname+'.npz')
        acc_p = param['arr_1']
        gyr_p = param['arr_0']       
        os.chdir('data')
        _ang = []
        _gyr0 = []
        _acc0 = []
        
        for i in range(len(_data)):
            _aux = np.array(unpack('>H',bytearray(_data[i][0:2]))+unpack('<hhhhhh',bytearray(_data[i][2:14])))
            _ang.append(_aux[0]*self.ang_sensitivity)
            _gyr0.append(self.transl(_aux[1:4], gyr_p))
            _acc0.append(self.transl(_aux[4:7], acc_p))   
        np.savez(_filename, _ang, _gyr0, _acc0)
        os.chdir('..')
        gc.collect()
        print(_filename)
    
    
    def lograw(self, _data):
        gc.collect()      
        _filename = 'raw_'+str(len(os.listdir('data')))
        os.chdir('data/raw')
        _aux = []
        
        for i in range(len(_data)):
            _aux.append(unpack('>H',bytearray(_data[i][0:2]))+unpack('<hhhhhh',bytearray(_data[i][2:14])))
               
        np.save(_filename, _aux)
        os.chdir('../..')
        gc.collect()
        print(_filename)
    
    def logdata(self, _data):
        gc.collect()      
        _filename = 'log_'+str(len(os.listdir('data')))+'.npy'
        os.chdir('data')
        _file = []
        for i in range(len(_data)):
            _aux = np.array(unpack('>H',bytearray(_data[i][0:2]))+unpack('<hhhhhh',bytearray(_data[i][2:14]))+unpack('<hhhhhh',bytearray(_data[i][14:26]))+unpack('>h',bytearray(_data[i][26:28])))
            _file.append(_aux)   
        np.save(_filename, _file)
        os.chdir('..')
        gc.collect()
        print(_filename)

   
    def pull(self):
        return bus.read_i2c_block_data(self.bus_addr[-1],0x22,12)#+bus.read_i2c_block_data(self.bus_addr[-2],0x22,12)+bus.read_i2c_block_data(self.bus_addr[-3],0x0,2), bus.read_i2c_block_data(0x36,0xE,2)+
    
    def pullcalib(self, _addr):
        return bus.read_i2c_block_data(_addr,0x22,12)
    
    def calibrate(self):
        _sensname = input('Connnect sensor and name it: ')
        _sensor = {'name': _sensname}
        self._caldata = []
        _addr = self.i2cscan()
        print('Iniciando 6 pos calibration')
        self._nsamp = int(input('Number of Samples/Position: ') or 5/self.dt)
        for _n in range(6):
            input('Position {}'.format(_n+1))
            i=0
            tf = time.perf_counter()
            while i<self._nsamp:
                ti=time.perf_counter()
                if ti-tf>=self.dt:
                    tf = ti
                    i+=1
                    self._caldata.append(self.pullcalib(_addr[-1]))
        self._gsamps = int(input('Number of Samples/Rotation: ') or 2/self.dt)
        for _n in range(0,6,2):
            input('Rotate 90 deg around axis {}-{}'.format(_n+1,_n+2))
            i=0
            tf = time.perf_counter()
            while i<self._gsamps:
                ti=time.perf_counter()
                if ti-tf>=self.dt:
                    tf = ti
                    i+=1
                    self._caldata.append(self.pullcalib(_addr[-1]))
        
        self._aux = []
        print('Data collection done...')
        for _d in self._caldata:
            self._aux.append(unpack('<hhhhhh',bytearray(_d)))
        _data = np.array(self._aux)
        self.acc_raw = _data[0:6*self._nsamp,3:6]
        self.gyr_raw = _data[:,0:3]
        np.save('./sensors/'+_sensor['name']+'rawdata.npy', _data)
        print(_sensor['name']+'rawdata saved')
        print('Calculating calibration parameters. Wait...')
        gc.collect()
        _sensor['acc_p'] = self.calibacc(self.acc_raw)
        gc.collect()
        _sensor['gyr_p'] = self.calibgyr(self.gyr_raw)        
        np.savez('./sensors/'+_sensor['name'], _sensor['gyr_p'], _sensor['acc_p'])
       
        os.chdir('..')
        gc.collect()
        return _sensor
    
    def calibacc(self, _accdata):
        _k = np.zeros((3, 3))
        _b = np.zeros((3))
        _Ti = np.ones((3, 3))
        
        self.acc_m = np.zeros((6, 3))
        for _i in range(6):
            for _j in range(3):
                self.acc_m[_i, _j] = np.mean(_accdata[_i*self._nsamp:(_i+1)*self._nsamp, _j])

        
        for _i in range(3):
            _max = self.acc_m[:,_i].max(0)
            _min = self.acc_m[:,_i].min(0)
            _k[_i, _i] = (_max - _min)/ (2*self.gravity)
            _b[_i] = (_max + _min)/2    
            _Ti[_i, _i-2] = np.arctan(self.acc_m[self.acc_m[:,_i].argmax(0),_i-2] / _max)
            _Ti[_i, _i-1] = np.arctan(self.acc_m[self.acc_m[:,_i].argmax(0),_i-1] / _max)
        _kT = inv(_k.dot(inv(_Ti)))
        _param = np.append(np.append(np.append(_kT.diagonal(), _b.T), _kT[np.tril(_kT, -1) != 0]), _kT[np.triu(_kT, 1) != 0])
        _jac = autograd.jacobian(self.accObj)
        _hes = autograd.hessian(self.accObj)
        _res = op.minimize(self.accObj, _param, method='trust-ncg', jac=_jac, hess=_hes)
        return _res.x
  
    
    def accObj(self, X):
        _NS = nap.array([[X[0], X[6], X[7]], [X[8], X[1], X[9]], [X[10], X[11], X[2]]])
        _b = nap.array([X[3], X[4], X[5]])
        _sum = 0
        for u in self.acc_m:
            _sum += (self.gravity - nap.linalg.norm(_NS@(u-_b).T))**2

        return _sum
        
    def calibgyr(self, _gyrdata):
        _gyr_s = _gyrdata[0:6*self._nsamp,:]
        _b = np.mean(_gyr_s, axis=0).T
        _gyr_d = _gyrdata[6*self._nsamp:,:] 
        _gyr_r = _gyr_d - _b
        _ang = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                _ang[i, j] = np.abs(intg.trapz(_gyr_r[self._gsamps*i:self._gsamps*(i+1), j], dx=self.dt))

        _n = _ang.argmax(axis=0)

        self.rates = np.zeros((self._gsamps,3))
        for i in range(3):
            self.rates[:,i] = _gyr_d[self._gsamps*_n[i]:self._gsamps*(_n[i]+1), i]

        _k = np.zeros((3,3))
        _k[:,0] = _ang[:,_ang[0].argmax()]
        _k[:,1] = _ang[:,_ang[1].argmax()]
        _k[:,2] = _ang[:,_ang[2].argmax()]

        _kT = np.diag([90,90,90])@inv(_k)
        
        _param = np.append(np.append(np.append(_kT.diagonal(), _b.T), _kT[np.tril(_kT, -1) != 0]), _kT[np.triu(_kT, 1) != 0])
        _jac = autograd.jacobian(self.gyrObj)
        _hes = autograd.hessian(self.gyrObj)
        _res = op.minimize(self.gyrObj, _param, method='trust-ncg', jac=_jac, hess=_hes)
        return _res.x
    
    def gyrObj(self,Y):
        _NS = nap.array([[Y[0], Y[6], Y[7]], [Y[8], Y[1], Y[9]], [Y[10], Y[11], Y[2]]])
        _b = nap.array([Y[3], Y[4], Y[5]])
        sum = 0
        for u in self.rates:
            sum += _NS@(u-_b).T*self.dt
       
    
        return (90 - nap.abs(sum)).sum()**2
    
    def transl(self, _data, X):
        _NS = np.array([[X[0], X[6], X[7]], [X[8], X[1], X[9]], [X[10], X[11], X[2]]])
        _b = X[3:6]
        _data_out = (_NS@(_data-_b))
        
        return _data_out

    
    
    
    