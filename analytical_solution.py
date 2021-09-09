import numpy as np


class exact_solution(object):
    def __init__(self, X, nu):
        self.nu = nu
        self.X = X

    def u(self, t):
        x = self.X[:, 0]
        y = self.X[:, 1]
        u = 3/4 - 1/4 * (1/(1 + np.exp((4*y - 4*x - t)/(self.nu*32))))
        v = 3/4 + 1/4 * (1/(1 + np.exp((4*y - 4*x - t)/(self.nu*32))))
        return np.hstack((u.reshape(-1, 1), v.reshape(-1, 1)))
