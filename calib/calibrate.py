# %%
import calibrator
import os
riza = calibrator.CALIBRIZZATTI()
print(os.listdir())
res = riza.calibrate("rawdata.npy")

