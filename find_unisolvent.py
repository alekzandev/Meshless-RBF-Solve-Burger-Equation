import numpy as np

def find_points(f, tol, n):
    s = list()
    x = np.random.random(n)
    y = np.random.random(n)
    for i in x:
        for j in y:
            r = f(i,j)
            if abs(r) < tol:
                s.append([i,j])
    return np.vstack(s)

f = lambda x,y: -x -y + 1
tol = 1e-7
n = 5000
#r = find_points(f,tol, n)
print(f(0.76759738, 0.23240258))