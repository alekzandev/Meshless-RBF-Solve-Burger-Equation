import numpy as np
from expressions import *
from halton_points import HaltonPoints


class interpolate(object):
    def __init__(self, n, dm=2):
        self.n = n
        self.Mi = HaltonPoints(dm, self.n).haltonPoints()
        self.Mb = np.array([
            [0., 0.],
            [0., 0.5],
            [0., 1.],
            [0.5, 1.],
            [1., 1.],
            [1., 0.5],
            [1., 0.],
            [0.5, 0.]
        ])
        self.poly_b = np.array([[-1, -1, 1], [1/2, 3/2, -1], [3/2, 1/8, -3/8]])

    def F(self):
        for x in self.Mi:
            assem_matrix = assembled_matrix(self.Mi, self.Mb, 1, 0.1, x, poly_b=self.poly_b)
            X0 = assem_matrix.X_0()
            yield assem_matrix.F_m(X0)[0]

aa = interpolate(n=100)
mlist = tuple(aa.F())
print(np.vstack(mlist))