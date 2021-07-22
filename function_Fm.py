import numpy as np

from explicit_RK import *
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
npnts = 1

uh = assembled_matrix(Mb, npnts, 2, 1, poly_b=poly_b)
X0 = uh.X_0()


def Fm(t, X0, uh):
    F = []
    for xn in uh.Mi:
        uh.x = xn
        # yield uh.F_m(X0)[0], uh.J()[0]
        F.append(uh.F_m(X0)[0])
    return np.vstack(F)

# F, J = Fm(uh, X0)
# F, J = np.vstack(F), np.vstack(J)
# print(J)


t0, te = 0, 1.
N = 10

system = explicit_RungeKutta(Fm, X0, t0, te, N, uh)
system.solve()
S = system.solution()
print(S)
# # t = S[:, 0]
# # y = S[:, 1]
# a = np.vstack(tuple(exact_sol(timegrid)))
# print(a)
# error = np.linalg.norm(np.vstack(y)-a, axis=-1)
# plt.plot(np.vstack(t), error)
# plt.show()
