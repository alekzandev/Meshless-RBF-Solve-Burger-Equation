#%%
import numpy as np
import matplotlib.pyplot as plt
from Halton_Points import haltonPoints
from augmented_system import aug_matrix
from scipy import linalg

# %%
def distanceMatrix2D(M):
    n = M.shape[0]
    my_matrix = list()
    for j in M:
        for l in M:
            aux = j-l
            my_matrix.append(linalg.norm(aux, ord=2))

    my_matrix = np.array(my_matrix)
    return my_matrix.reshape(n,n)

def GaussianRBF(M, e=1):
     return np.exp(-(e*M)**2)

def test_func(x,y):
    return (x+y)/2

# %%
n = 512
m = 3
M = haltonPoints(2, n)
x, y = M[:, 0], M[:, 1]

A = GaussianRBF(distanceMatrix2D(M))
B = aug_matrix(A, M, n, m)
F = np.hstack((test_func(x, y), np.zeros((m))))
C = linalg.solve(B, F)

# %%
error = list()
for _ in range (int(1e3)):
    n_test = 1
    M_test = np.random.random(n_test*2).reshape((n_test,2))

    M_dist_tot = list()
    for p_test in M_test:
        M_dist = list()
        for p_col in M:
            M_dist.append(linalg.norm(p_test-p_col, ord = 2))
        M_dist_tot.append(M_dist)
        

    M_dist = np.array(M_dist)
    res = np.dot(C[:-m], np.exp(-(M_dist)**2)) + np.dot(np.array([1, M_test[0][0], M_test[0][1]]), C[-m:])
    error.append(abs(res - test_func(M_test[:,0], M_test[:,1])[0]))

error = np.array(error)
plt.hist(error)
# %%
