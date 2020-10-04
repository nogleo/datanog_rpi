import datanog as nog
import os
import time
from collections import deque
import numpy as np
from struct import unpack
dn = nog.DATANOG()

caldata= deque()
ext = ''
addr = dn.i2cscan()
while ext != '0':
    ext = input('Positin and Press button: ')
    if ext == '1':
        _count = 100000*30
    else:
        _count = 100000*3
    i=0
    tf = time.perf_counter()
    while i<_count:
        ti=time.perf_counter()
        if ti-tf>=1/1660:
            tf = ti
            i+=1
            caldata.append(dn.pullcalib(addr[-1]))


file = []
_filename = 'cal_'+str(len(os.listdir('calib')))+'.npy'
os.chdir('calib')
for i in range(len(caldata)):
    file.append(unpack('<hhhhhh',bytearray(caldata[i][0:12])))    
np.save(_filename,file)
os.chdir('..')
print(_filename)