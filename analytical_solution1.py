import numpy as np
from scipy import integrate, special


def u_analytical(x, t, nu=1):
    num = 2 * nu * (np.exp(-(x-1)**2/(4 * nu * t + 1e-40)) -
                    np.exp(-x**2/(4 * nu * t + 1e-40)))
    den = np.sqrt(np.pi * nu * t + 1e-40) * (special.erf((1-x) /
                                                         (2 * np.sqrt(nu * t + 1e-40))) - special.erf(x/(2 * np.sqrt(nu * t + 1e-40))))
    return num/den


def ki(x, i, nu):
    return np.exp(-(2 * np.pi * nu)**(-1)
                                   * (1-np.cos(np.pi * x)))*np.cos(i * np.pi * x)
    return 2 * integrate.quad(f(x, i, nu), 0, 1)[0]



def k0(x):
    return np.exp(-(2 * np.pi)**(-1) * (1-np.cos(np.pi * x)))


def u_analytical2(x, t, nu=1):
    num = 0
    for i in range(1, 500):
        num += integrate.quad(ki(x, i, nu=1),0,1, args=(1,))[0] * np.exp(-i**2 * np.pi **
                                     2 * nu * t) * i * np.sin(i * np.pi * x)
    subden = 0
    for i in range(1, 500):
        subden += ki(x, i, nu) * np.exp(-i**2 * np.pi **
                                        2 * nu * t) * np.cos(i * np.pi * x)
    den = integrate.quad(k0(x), 0, 1)[0] + subden

    return 2 * nu * np.pi * num/den


print(u_analytical2(1, 0))