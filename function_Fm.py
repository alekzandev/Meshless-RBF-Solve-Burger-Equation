import numpy as np

from expressions import *

Mb = np.array([
    [0., 0.],
    [0., 0.5],
    [0., 1.],
    [0.5, 1.],
    [1., 1.],
    [1., 0.5],
    [1., 0.],
    [0.5, 0.]
])


poly_b = np.array([[-1, -1, 1], [1/2, 3/2, -1], [3/2, 1/8, -3/8]])
x = np.array([0.16, .093])
npnts = 3

uh = assembled_matrix(Mb, npnts, 2, 1, poly_b=poly_b)
X0 = uh.X_0()


def Fm(uh, X0):
    for xn in uh.Mi:
        uh.x = xn
        yield uh.F_m(X0)[0], uh.J(X0)[0]


FJ = np.vstack(tuple(Fm(uh, X0)))
F, J = FJ[::2], FJ[1::2]

print(F)
