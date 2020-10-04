import numpy as np
from numpy.linalg import inv
import scipy.optimize as op

# %% Carregando dados
caldata = np.load("calib/cal_1.npy")

# %% Definindo variáveis para cálculo
gravity = 9.81              # Gravidade usada na otimização
NSamplesMesure = 300000     # Número de amostras medidas em cada posição

# %% Disovendo arquivo de medição
gy0 = caldata[:, 0:3]       # Serando as rotações
ac0 = caldata[:, 3:6]       # Separando as acelerações
acc = caldata[0:-1, 3:6]
gyr = caldata[0:-1, 0:3]
k = np.zeros((3, 3))        # Criado a matriz dos fatores de escala
b = np.zeros((3, 1))        # Criando o vetor de Bias (offset)
Ti = np.ones((3, 3))        # Criando a matriz de desalinhamento do sensor
_aux = np.zeros((6, 3))

# %% Realizando a média no tempo de cada posição medida
for _i in range(6):
    for _j in range(3):
        _aux[_i, _j] = np.mean(
            acc[_i*NSamplesMesure:(_i+1)*NSamplesMesure, _j])

acc_m = _aux

# %% Filtrando amplitudes das acelerações e calculando k, b e Ti
for _i in range(3):
    _a_up = np.mean(acc[acc[:, _i] > 1800, _i])
    _a_down = np.mean(acc[acc[:, _i] < -1800, _i])
    Ti[_i, _i-2] = np.mean(acc[acc[:, _i] > 1800, _i-2]) / \
        np.mean(acc[acc[:, _i] > 1800, _i])
    Ti[_i, _i-1] = np.mean(acc[acc[:, _i] > 1800, _i-1]) / \
        np.mean(acc[acc[:, _i] > 1800, _i])
    k[_i, _i] = (_a_up - _a_down)/(2*9.81)
    b[_i] = (_a_up + _a_down)/2


kT = inv(k.dot(inv(Ti)))

# %% Organizando vetor a ser otimizado x = [Sx,Sy,Sz,bx,by,bz,phiyx,phizx,phizy]
x = np.append(np.append(kT.diagonal(), b.T), kT[np.tril(kT, -1) != 0])

# %% Função objetiva a ser otimizada


def funcObj(X):
    N_S = np.array([[X[0], 0, 0], [X[6], X[1], 0], [X[7], X[8], X[2]]])

    b = np.array([[X[3]], [X[4]], [X[5]]])

    G = gravity

    summ = 0

    for u in acc_m:
        summ += (G - np.linalg.norm(N_S@(u-b.T).T))**2

    return summ


# %% Realizando a otimização
resultado = op.minimize(funcObj, x, method='SLSQP')

# %% Mostrnado o resultado de otimização

print(resultado)