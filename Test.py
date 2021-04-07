import numpy as np


class Onestepmethod(object):
    def __init__(self, f, x0, t0, te, N):
        self.f = f
        self.x0 = x0
        self.interval = [t0, te]
        self.grid = np.linspace(t0, te, N)
        self.h = (te - t0)/N

    def step(self):
        ti, ui = self.grid[0], self.x0
        yield ti, ui

        for t in self.grid[1:]:
            ui += self.h * self.phi(self.f, ui, ti)
            ti = t
            yield ti, ui

        return ui

    def solve(self):
        self.solution = np.array([tu for tu in self.step()])


class ExpEuler(Onestepmethod):
    def phi(self, f, u, t):
        return f(u, t)


class MidPointRule(Onestepmethod):
    def phi(self, f, u, t):
        return f(u + self.h/2 * f(u, t), t + self.h/2)


def f(x, t):
    return -0.5*x


euler = ExpEuler(f, 15., 0., 10., 20)
# euler.solve()

midpoint = MidPointRule(f, 15, 0., 10., 20)
print(midpoint.solve())
