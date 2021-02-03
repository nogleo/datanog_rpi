
import datanog as nog
import time
from collections import deque
dn = nog.DATANOG()
fs = 3333
dt = 1/fs

i=0
data0 = deque()
t0=tf = time.perf_counter()
while i<10000:
    ti=time.perf_counter()
    if ti-tf>=dt:
        tf = ti
        i+=1
        data0.append(dn.pull())
    
t1 = time.perf_counter()
print(t1-t0)
dn.lograw(data0)
dn.log(data0)


 