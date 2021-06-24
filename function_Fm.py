import numpy as np
from expressions import *
from halton_points import HaltonPoints


class f(object):
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
            yield assem_matrix.F_m(X0)


# Mi = HaltonPoints(2, 10).haltonPoints()  # np.hstack((x_i, y_i))
# Mb = np.array([
#     [0., 0.],
#     [0., 0.5],
#     [0., 1.],
#     [0.5, 1.],
#     [1., 1.],
#     [1., 0.5],
#     [1., 0.],
#     [0.5, 0.]
# ])

# x = np.array([1.1, 2.3])
# poly_b = np.array([[-1, -1, 1], [1/2, 3/2, -1], [3/2, 1/8, -3/8]])

# aa = assembled_matrix(Mi, Mb, 1, 0.1, x, poly_b=poly_b)
# X0 = aa.X_0()
# print(aa.F_m(X0))
aa = f(5)
print(list(f.F()))