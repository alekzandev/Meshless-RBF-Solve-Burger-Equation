import numpy as np


class Gauss():
    A = np. array([
        [5/36, 2/9 - np.sqrt(15)/15, 5/36 - np.sqrt(15)/30],
        [5/36 + np.sqrt(15)/24, 2/9, 5/36 - np.sqrt(15)/24],
        [5/36 + np.sqrt(15)/30, 2/9 + np.sqrt(15)/15, 5/36]
    ])

    b = np.array([5/18, 4/9, 5/18])
    c = np.array([1/2-np.sqrt(15)/10, 1/2, 1/2 + np.sqrt(15)/10])


class SDIRK_tableau2s():
    p = (3 - np.sqrt(3))/6
    A = np.array([[p, 0], [1 - 2*p, p]])
    b = np.array([1/2, 1/2])
    c = np.array([p, 1-p])


class SDIRK_tableau5s():
    A = np.array([
        [1/4, 0, 0, 0, 0],
        [1/2, 1/4, 0, 0, 0],
        [17/50, -1/25, 1/4, 0, 0],
        [371/1360, -137/2720, 15/544, 1/4, 0],
        [25/24, -49/48, 125/16, -85/12, 1/4]
    ])
    b = np.array([25/24, -49/48, 125/16, -85/12, 1/4])
    c = np.array([1/4, 3/4, 11/20, 1/2, 1])
