import os, gc
from struct import unpack


import numpy as np
import smbus
import sched, time

bus = smbus.SMBus(1)


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

    
   

       
    #TODO: add ADC pull
    def pull(self):
        return bus.read_i2c_block_data(0x36,0xE,2)+bus.read_i2c_block_data(self.bus_addr[-1],0x22,12)+bus.read_i2c_block_data(self.bus_addr[-2],0x22,12)+bus.read_i2c_block_data(self.bus_addr[-3],0x0,2)
    
    def pullcalib(self, _addr):
        return bus.read_i2c_block_data(_addr,0x22,12)
    



