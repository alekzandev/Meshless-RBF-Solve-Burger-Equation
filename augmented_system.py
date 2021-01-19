import numpy as np

def aug_matrix(A, Mp, n, m):
    P = np.hstack((np.ones((n,1)), Mp))
    aux = np.hstack((A, P))
    aux1 = np.hstack((P.transpose(), np.zeros((m,m)))) 

    return np.vstack((aux, aux1))
