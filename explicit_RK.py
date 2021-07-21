import numpy as np


class explicit_RungeKutta(object):
    def __init__(self, f, u0, t0, te, N):
        self.f = f
        self.u0 = u0
        self.dt = (te-t0)/(N+2)
        self.timegrid = np.linspace(t0, te, N+2)

    def stage2(self, t):
        return self.u0 + self.dt/2 * self.f(t + self.dt/2, self.u0)

    def stage3(self, t):
        ub = self.stage2(t)
        return self.u0 + self.dt/2 * self.f(t + self.dt/2, ub)

    def stage4(self, t):
        ub = self.stage3(t)
        return self.u0 + self.dt * self.f(t + self.dt, ub)

    def stagef(self, t):
        return self.u0\
            + self.dt/6 * (self.f(t, self.u0)
                           + 2*self.f(t + self.dt/2, self.stage2(t))
                           + 2*self.f(t + self.dt/2, self.stage3(t))
                           + self.f(t + self.dt, self.stage4(t)))

    def solve(self):
        u_n = self.u0
        yield u_n
        for t in self.timegrid:
            u_n = self.stagef(t)
            self.u0 = u_n
            yield u_n
    
    def solution(self):
        return np.array(tuple(self.solve()))



t0, te = 0, 1.
tol_newton = 1e-9
tol_sol = 1e-5
N = 5
t = 0

u0 = np.array([2., 3.])
M = np.array([[0, 1], [-1, 0]])
def f(t,y): return np.dot(M, y)
system = explicit_RungeKutta(f, u0, t0, te, N)

timegrid  = np.linspace(t0, te, N+2)
def exact_sol(timegrid):
    for t in timegrid:
        y1 = np.array([2*np.cos(t) + 3*np.sin(t)])
        y2 = np.array([-2*np.sin(t)+ 3*np.cos(t)])
        yield np.hstack((y1,y2))

system.solve()
S = system.solution()
print(S)
# t = S[:, 0]
# y = S[:, 1]
a = np.vstack(tuple(exact_sol(timegrid)))
print(a)
# error = np.linalg.norm(np.vstack(y)-a, axis=-1)
# plt.plot(np.vstack(t), error)
# plt.show()
