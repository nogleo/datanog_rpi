import os, gc
from struct import unpack
import RPi.GPIO as GPIO
import time
from collections import deque
import numpy as np
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
    def __init__(self, _base = 2, _scale = 0):
        self.__name__ = "DATANOG"
        self.sampfreq = 1660
        self.led = (18, 23, 24)

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
        self.acc_sensitivity = .061 * 1e-3 * 9.81 # mg/LSB
        self.gyro_sensitivity = 4.375 * 1e-3 # mdps/LSB
        self.ang_sensitivity = 0.087890625  #deg/LSB
        self.accodr = ODR_1_66_KHZ
        self.accscale = ACC_SCALE_16G
        self.gyroodr = ODR_1_66_KHZ
        self.gyroscale = GYRO_SCALE_1000DPS
        # this will be a constant multiplier for the output data
        self._accscale = [1, 8, 2, 4]
        self.a_scaler = self._accscale[self.accscale]
        self._gyroscale = [2, 4, 8, 16]
        self.g_scaler = self._gyroscale[self.gyroscale]

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
    
    def led(self, _led = 0, _n = 1, _dur = 300, _blink = True):
        if _blink == False:
            GPIO.output(self.led[_led], not GPIO.input(self.led[_led]))
        else:
            for _i in range(_n):
                GPIO.output(self.led[_led], 1)
                time.sleep(_dur/1000)
                GPIO.output(self.led[_led], 0)
                time.sleep(_dur/1000)


    
    
    
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

    '''
    def get(self):
        data0 = deque()
        t0 = time.perf_counter()
        tf = time.perf_counter()
        while #wait button press#:
            ti=time.perf_counter()
            if ti-tf>=1/1660:
                tf = ti
                data0.append(self.pull())
    
        t1 = time.perf_counter()
        print(t1-t0)
        dn.logdata(data0)
        '''

       
    
    def pull(self):
        return bus.read_i2c_block_data(0x36,0xE,2)+bus.read_i2c_block_data(self.bus_addr[-1],0x22,12)#+bus.read_i2c_block_data(self.bus_addr[-2],0x22,12)+bus.read_i2c_block_data(self.bus_addr[-3],0x0,2)
    
    def pullcalib(self, _addr):
        return bus.read_i2c_block_data(_addr,0x22,12)
    
    def calibrate(self):
        _sensname = input('Connnect sensor and name it: ')
        _sensor = {'name': _sensname}
        self._caldata = []
        _addr = self.i2cscan()
        print('Iniciando 6 pos calibration')
        _nsamp = int(input('Number of Samples/Position: ') or 10000)
        for _n in range(6):
            input('Position {}'.format(_n+1))
            i=0
            tf = time.perf_counter()
            while i<_nsamp:
                ti=time.perf_counter()
                if ti-tf>=1/1660:
                    tf = ti
                    i+=1
                    self._caldata.append(self.pullcalib(_addr[-1]))
        
        self._aux = []
        for _d in self._caldata:
            self._aux.append(unpack('<hhhhhh',bytearray(_d)))
        _data = np.array(self._aux)
        self.acc_raw = _data[:,3:6]
        self.gyr_raw = _data[:,0:3]
        _sensor['acc_p'] = self.calibacc(self.acc_raw)
        _sensor['gyr_p'] = self.calibgyr(self.gyr_raw)
        
        os.chdir('sensors')
        os.mkdir(_sensor['name'])
        os.chdir(_sensor['name'])
        np.savez('gyr_param.npz', _sensor['gyr_p'])
        np.savez('acc_param.npz', _sensor['acc_p'][0], _sensor['acc_p'][1])
        
        os.chdir('..')
        os.chdir('..')
        
        return _sensor
    
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
        
        return _kT, _b
    
        
    def calibgyr(self, _gyrdata):
        _k = np.zeros((3, 3))
        _b = np.mean(_gyrdata, axis=0)
        
        return _b