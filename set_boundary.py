import numpy as np


def R(t):
    return 0.5*np.sqrt((np.cos(5*t))**2 + (np.cos(2*t))**2 + (np.cos(t))**2)


def x(t):
    return R(t)*np.cos(t)


def y(t):
    return R(t)*np.sin(t)
