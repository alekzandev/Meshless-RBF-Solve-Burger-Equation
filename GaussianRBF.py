#%%
import numpy as np
import matplotlib.pyplot as plt
from Halton_Points import haltonPoints
from augmented_system import aug_matrix
from scipy import linalg

# %%
def distanceMatrix2D(n):
    M = haltonPoints(2, n)
    my_matrix = list()
    for j in M:
        for l in M:
            aux = j-l
            my_matrix.append(linalg.norm(aux, ord=2))

    my_matrix = np.array(my_matrix)
    return my_matrix.reshape(n,n)

# %%
def GaussianRBF(n, e=1):
     M = distanceMatrix2D(n)
     return np.exp(-(e*M)**2)

# %%
def test_func(x,y):
    return (x+y)/2

# %%
n = 256
m = 3
MHp = haltonPoints(2, n)
x, y = MHp[:, 0], MHp[:, 1]

A = GaussianRBF(n)
B = aug_matrix(A, MHp, n, m)
F = np.hstack((test_func(x, y), np.zeros((m))))
C = linalg.solve(B, F)

# %%
n_test = 1
M_test = np.random.random(n_test*2).reshape((n_test,2))

# %%
M_dist = list()
for i in M_test:
    for j in MHp:
        M_dist.append(linalg.norm(i-j, ord = 2))

M_dist = np.array(M_dist)
res = np.dot(C[:-m], np.exp(-(M_dist)**2)) + np.dot(np.array([1, M_test[0][0], M_test[0][1]]), C[-m:])
res - test_func(M_test[:,0], M_test[:,1])[0]
