import numpy as np
from Implement_Thesis import  *

t0, te = 0, 1.
tol_newton = 1e-9
tol_sol = 1e-5
N = 100

y0 = np.array([1])
def f(t, y): return -5*y

scalar = Gauss(f, y0, t0, te, N, tol_newton)
#scalar = SDIRK_tableau2s(f, y0, t0, te, N, tol_newton)

scalar.solve()
S = scalar.solution

t = S[:, 0]
y = S[:, 1]
a = np.exp((-5*t))
error = np.abs(y-a)

plt.plot(t, error)
plt.show();