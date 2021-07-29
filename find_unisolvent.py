import numpy as np

def find_points(f, tol, n):
    s = list()
    x = np.random.random(n)
    b = [[0], [1], [0], [1]]
    m = len(x)
    for i in x:
        for j in b:
            for k in j*m:
                r = f(i,k)
                if abs(r) < tol:
                    s.append([i,k])
    return s#np.vstack(s)

f = lambda x,y: x/2 +3*y/2 - 1
tol = 1e-4
n = 20
r = find_points(f,tol, n)
print(r)