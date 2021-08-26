import numpy as np


class explicit_RungeKutta(object):
    def __init__(self, f, u0, t0, te, N, uh):
        self.f = f
        self.u0 = u0
        self.dt = (te-t0)/N
        self.timegrid = np.linspace(t0, te, N+1)
        self.uh = uh

    def stage2(self, t):
        return self.u0 + self.dt/2 * self.f(t + self.dt/2, self.u0, self.uh)

    def stage3(self, t):
        ub = self.stage2(t)
        return self.u0 + self.dt/2 * self.f(t + self.dt/2, ub, self.uh)

    def stage4(self, t):
        ub = self.stage3(t)
        return self.u0 + self.dt * self.f(t + self.dt, ub, self.uh)

    def stagef(self, t):
        return self.u0\
            + self.dt/6 * (self.f(t, self.u0, self.uh)
                           + 2*self.f(t + self.dt/2, self.stage2(t), self.uh)
                           + 2*self.f(t + self.dt/2, self.stage3(t), self.uh)
                           + self.f(t + self.dt, self.stage4(t), self.uh))

    def step(self):
        u_n = self.u0
        yield u_n
        # print(self.timegrid)
        for t in self.timegrid[1:]:
            u_n = self.stagef(t)
            self.u0 = u_n
            yield u_n

    def solve(self):
        self.solution = np.array(tuple(self.step()))


# t0, te = 0, 1.
# tol_newton = 1e-9
# tol_sol = 1e-5
# N = 10000
# t = 0

# u0 = np.array([2., 3.])
# M = np.array([[0, 1], [-1, 0]])
# def f(t,y,r): return np.dot(M, y)
# system = explicit_RungeKutta(f, u0, t0, te, N, 2)

# timegrid  = np.linspace(t0, te, N+2)
# def exact_sol(timegrid):
#     for t in timegrid:
#         y1 = np.array([2*np.cos(t) + 3*np.sin(t)])
#         y2 = np.array([-2*np.sin(t)+ 3*np.cos(t)])
#         yield np.hstack((y1,y2))

# system.solve()
# S = system.solution
# a = np.vstack(tuple(exact_sol(timegrid)))
# error = np.linalg.norm(S-a, axis=-1)
# plt.plot(timegrid, error)
# plt.show()
