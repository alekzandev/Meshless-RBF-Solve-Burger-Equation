from tableu_order import *


def test_scalar():
    t0, te = 0, 0.1
    tol_newthon = 1e-9
    tol_sol = 1e-5
    N = [2*n for n in range(10)]

    for method in [(Gauss, 6), (SDIRK_tableau2s, 3), (SDIRK_tableau5s, 4)]:
        stepsize = list()
        mean_error = list()

        for n in N:
            stepsize.append((te-t0)/(n+1))
            timeGrid = np.linspace(t0, te, n+2)
            expected = [(t, np.exp(-5*t)) for t in timeGrid]

    return expected


if __name__ == '__main__':
    print(test_scalar())
