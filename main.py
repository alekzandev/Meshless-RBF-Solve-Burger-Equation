import numdifftools as nd
import numpy as np
from scipy import linalg


def f(x, y):
    return x+y


def F(stageDer, t0, y0, s, m, A, f, c, h):
    stageDer_new = np.empty((s, m))

    for i in range(s):
        h_phi = [h * np.dot(A[i, :], stageDer.reshape(s, m)[:, j])
                 for j in range(m)]
        stageVal = y0 + np.array(h_phi)
        stageDer_new[i, :] = f(t0 + c[i] * h, stageVal)

    return stageDer - stageDer_new.reshape(-1)


def phi_newthonstep(t0, y0, initVal, J, luFactor, s, m, A, f, c, h):
    d = linalg.lu_solve(luFactor, F(
        initVal.flatten(), t0, y0, s, m, A, f, c, h))

    return initVal.flatten() + d, np.norm(d)


def phi_solve(t0, y0, initVal, J, M, s, m, h, A, c, tol):
    JJ = np.eye(s*m) - h * np.kron(A, J)
    luFactor = linalg.lu_factor(JJ)
    for i in range(M):
        initVal, norm_d = phi_newthonstep(
            t0, y0, initVal, J, luFactor, s, m, A, f, c, h)
        if norm_d < tol:
            print("Newton converged in {:,.0f} steps.".format(i))
            break
        elif i == M-1:
            raise ValueError("The Newton iteration did not converge.")

    return initVal


def phi(t0, y0, J, M, s, m, h, A, c, tol, b):
    M = 10
    stageDer = np.array(s*[f(t0, y0)])
    J = nd.Jacobian(f(t0, y0))
    stageVal = phi_solve(t0, y0, stageDer, J, M, s, m, h, A, c, tol)
    phi_val = [np.dot(b, stageVal.reshape(s, m)[:, j]) for j in range(m)]

    return np.array(phi_val)
