import numpy as np


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
        self.solution = list(self.step)
