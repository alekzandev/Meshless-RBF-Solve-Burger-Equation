#%%
import numpy as np
import matplotlib.pyplot as plt
from Halton_Points import haltonPoints
from augmented_system import aug_matrix

#%%
n = 256
MHp = haltonPoints(2, n)
x, y = MHp[:, 0], MHp[:, 1]


# %%
def distanceMatrix2D(n):
    M = haltonPoints(2, n)
    my_matrix = list()
    for _,j in enumerate(M):
        for _,l in enumerate(M):
            aux = j-l
            my_matrix.append(np.linalg.norm(aux, ord=2))

    my_matrix = np.array(my_matrix)
    return my_matrix.reshape(n,n)

# %%
def GaussianRBF(n, e=1):
     M = distanceMatrix2D(n)
     return np.exp(-(e*M)**2)

# %%
