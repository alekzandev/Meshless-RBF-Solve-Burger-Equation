import numpy as np
import scipy as sc


class Onestepmethod(object):
    def __init__(self, f, y0, b, t0, te, N, tol):
        self.f = f
        self.y0 = y0.astype('float')
        self.t0 = t0
        self.interval = [t0, te]
        self.grid = np.linspace(t0, te, N+2)
        self.h = (te-t0)/(N+1)
        self.N = N
        self.tol = tol
        self.m = len(y0)
        self.s = len(b)

    def step(self):
        ti, yi = self.grid[0], self.y0
        tim1 = ti
        yield ti, yi

        for ti in self.grid[1:]:
            yi += self.h*self.phi(tim1, yi)
            tim1 = ti
            yield ti, yi

    def solve(self):
        self.solution = list(self.step())


class RungeKutta_implicit(Onestepmethod):

    def F(self, stageDer, t0, y0):
        stageDer_new = np.empty((self.s, self.m))
        for i in range(self.s):
            stageVal = y0 + np.array([self.h * np.dot(
                self.A[i, :], stageDer.reshape(self.s, self.m)[:, j]) for j in range(self.m)])
            stageDer_new[i, :] = self.f(t0 + self.c[i] * self.h, stageVal)
        return stageDer - stageDer_new.reshape(-1)

    def phi_newthonstep(self, t0, y0, initialVal, luFactor):
        d = sc.linalg.lu_solve(luFactor, -self.F(initialVal.flatten(), t0, y0))

        return initialVal.flatten() + d, np.norm(d)

    # def phi_solve(self, t0, y0, initialVal, J, M):
    #     JJ = np.eye(self.s * self.m) - self.h * np.kron(self.A, J)
    #     luFactor = sc.linalg.lu(JJ)
    #     for i in range(M):
    #         initialVal, norm_d = self.phi_newthonstep(
    #             t0, y0, initialVal, luFactor)

    #         if norm_d < self.tol:
    #             print('Newton converged in {:,.0f} steps.'.format(i))
    #             break
    #         elif i == M-1:
    #             raise ValueError("The Newton iteration did not converge.")
    #     return initialVal

    # def phi(self, t0, y0):
    #     '''
    #     Calculates the sum of b_j*Y_j in one step of the Runge-Kutta method with y_{n+1}= y_n + h*sum_{j=1}^s b_j *Y
    #     where j=1,2,...,s and s is the number of stages, b the notes and Y the stages values.

    #     Parameters:
    #     ----------------
    #     t0 = float, current timestep
    #     y0 = 1xm vector, the last solution y_n. Where m is the lenght of the IC y_0 of IVP.

    #     '''
    #     M = 10
    #     stageDer = np.array(self.s*[self.f(t0, y0)])  # Initial value Y'_0
    #     J = np.jacobian(self.f, t0, y0)


class SDIRK(RungeKutta_implicit):
    def phi_solve(self, t0, y0, initVal, J, M):
        JJ = np.eye(self.m) - self.h * self.A[0, 0] * J
        luFactor = sc.linalg.lu(JJ)


# f = np.sin
# y0 = np.array([9,2,3])
# b = np.array([5,2,3])
# t0 = 0
# te = 5
# N = 1000
# tol = 1e-9

# meth = Onestepmethod(f, y0, b, t0, te, N, tol)
# print(meth.step())
