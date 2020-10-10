# %%
import autograd.numpy as np
from numpy.linalg import inv, norm
import scipy.optimize as op
import matplotlib.pyplot as plt
import autograd
# %% Carregando dados
caldata = np.load("calib/rawdata.npy")
nsamp = 10000
gsamp = 2000

# %% Disovendo arquivo de medição
gy0 = caldata[:, 0:3]       # Serando as rotações
ac0 = caldata[:, 3:6]       # Separando as acelerações
acc = caldata[0:6*nsamp,3:6]
gyr = caldata[:,0:3]
k = np.zeros((3, 3))        # Criado a matriz dos fatores de escala
b = np.zeros((3, 1))        # Criando o vetor de Bias (offset)
Ti = np.ones((3, 3))        # Criando a matriz de desalinhamento do sensor
_aux = np.zeros((6, 3))

# %% Definindo variáveis para cálculo
gravity = 9.81          # Gravidade usada na otimização
rotation = 90           # Rotação usada na otimização
NSamplesMesure = len(acc)//6    # Número de amostras medidas em cada posição
# %% Realizando a média no tempo de cada posição medida
for _i in range(6):
    for _j in range(3):
        _aux[_i, _j] = np.mean(acc[_i*nsamp:(_i+1)*nsamp, _j])

acc_m = _aux

# %% Filtrando amplitudes das acelerações e calculando k, b e Ti
for _i in range(3):
    _a_up = np.mean(acc[acc[:, _i] > 1800, _i])
    _a_down = np.mean(acc[acc[:, _i] < -1800, _i])
    Ti[_i, _i-2] = np.arctan(np.mean(acc[acc[:, _i] > 1800, _i-2]) / \
        np.mean(acc[acc[:, _i] > 1800, _i]))
    Ti[_i, _i-1] = np.arctan(np.mean(acc[acc[:, _i] > 1800, _i-1]) / \
        np.mean(acc[acc[:, _i] > 1800, _i]))
    k[_i, _i] = (_a_up - _a_down)/(2*9.81)
    b[_i] = (_a_up + _a_down)/2


kT = inv(k.dot(inv(Ti)))

# %% Organizando vetor a ser otimizado x = [Sx,Sy,Sz,bx,by,bz,phiyx,phizx,phizy]
x = np.append(np.append(np.append(kT.diagonal(), b.T), kT[np.tril(kT, -1) != 0]), kT[np.triu(kT, 1) != 0])

# %% Função objetiva a ser otimizada


def funcObj(X):
    _NS = np.array([[X[0], X[6], X[7]], [X[8], X[1], X[9]], [X[10], X[11], X[2]]])
    _b = np.array([[X[3]], [X[4]], [X[5]]])
    G = gravity

    summ = 0

    for u in acc_m:
        summ += (G - np.linalg.norm(_NS@(u-_b.T).T))**2

    return summ

# %%
def transcal(_data, X):
    _data_out = np.zeros(_data.shape)
    _NS = np.array([[X[0], 0, 0], [X[6], X[1], 0], [X[7], X[8], X[2]]])
    _b = np.array([[X[3]], [X[4]], [X[5]]])
    
    for _i in range(len(_data)):
        _data_out[_i] = (_NS@(_data[_i]-_b.T).T).reshape(3,)
    
    return _data_out

# %%
jacF = autograd.jacobian(funcObj)
hesF = autograd.hessian(funcObj)

# %% Realizando a otimização
resultado = op.minimize(funcObj, x, method='trust-ncg', jac=jacF, hess=hesF)

# %% Mostrnado o resultado de otimização

print(resultado)

ACC = {'precal': x, 'param': resultado.x}

# %%
acc_cal = transcal(ac0, resultado.x)
accc_cal = transcal(ac0, x)
# %%
plt.figure()
plt.plot(acc_cal-accc_cal)
# %%
plt.figure()
plt.plot(acc_cal)

# %%
plt.figure()
plt.plot(accc_cal)

# %%
gyr_s = gyr[0:6*nsamp, :]
b = gyr_s.mean(0)
gyr_r = gyr[6*nsamp:,:]-b


# %%
_aux = np.zeros((3, 3))
for _i in range(3):
    for _j in range(3):
        _aux[_i, _j] = np.abs(np.sum((gyr_r[_i*gsamp:(_i+1)*gsamp, _j])/1660))

k = np.zeros((3,3))
k[:,0] = _aux[:,_aux[0].argmax()]
k[:,1] = _aux[:,_aux[1].argmax()]
k[:,2] = _aux[:,_aux[2].argmax()]

k = np.diag([90,90,90])@inv(k)
# %%
y = np.append(np.append(np.append(k.diagonal(), b.T), k[np.tril(k, -1) != 0]), k[np.triu(k, 1) != 0])
#y = np.append(k.diagonal(), b.T)
gyr_r = gyr_r+b
# %%
def gyrObj(Y):
    _NS = np.array([[Y[0], Y[6], Y[7]], [Y[8], Y[1], Y[9]], [Y[10], Y[11], Y[2]]])
    _b = np.array([[Y[3]], [Y[4]], [Y[5]]])
    _R = rotation
    
    _sum = 0
    for _u in gyr_r:
        _sum +=((_NS@(_u).T).reshape(3,)/1660)
        
   
    
    return (np.sum(_R - np.abs(_sum)))**2
    
# %%

jacG = autograd.jacobian(gyrObj)
hesG = autograd.hessian(gyrObj)

# %%

res_gyr = op.minimize(gyrObj, y, method='trust-ncg', jac=jacG, hess=hesG)
print(res_gyr)

# %%
def transgyr(_data, Y):
    _data_out = np.zeros(_data.shape)
    _NS = np.array([[Y[0], Y[6], Y[7]], [Y[8], Y[1], Y[9]], [Y[10], Y[11], Y[2]]])
    #_NS = np.diag([Y[0], Y[1], Y[2]])
    _b = np.array([[Y[3]], [Y[4]], [Y[5]]])
    
    for _i in range(len(_data)):
        _data_out[_i] = (_NS@(_data[_i]-_b.T).T).reshape(3,)
    return _data_out
        
#%%
gyr_pc = transgyr(gyr,y)
plt.plot(gyr_pc)

#%%
gyr_cc = transgyr(gyr, res_gyr.x)
plt.plot(gyr_cc)

# %%

plt.plot(gyr_cc - gyr_pc)
# %%
