import numpy as np

from expressions import *
from halton_points import HaltonPoints

Mi = HaltonPoints(2, 5).haltonPoints()

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

my_obj = assembled_matrix(Mi, Mb, x, 2, 1, poly_b=poly_b)
#X0 = my_obj.X_0()
#print(my_obj.F_m(X0))
xn = Mi[0,:]
#my_obj.x = xn
#print(my_obj.F_m(X0))