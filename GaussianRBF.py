#%%
import numpy as np
import matplotlib.pyplot as plt
from Halton_Points import haltonPoints

#%%

x = haltonPoints(2, 4)[:,0]
y = haltonPoints(2, 4)[:, 1]


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
def GaussianRBF(n, e):
     M = distanceMatrix2D(n)
     return np.exp(-(e*M)**2)

# %%
