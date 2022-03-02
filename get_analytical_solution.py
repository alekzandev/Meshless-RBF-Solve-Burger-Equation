# user/bin/python3
#%%
from analytical_solution import *
from halton_points import HaltonPoints

#%%
nu = 0.01
X = HaltonPoints(2, 500).haltonPoints()
U = exact_solution(X, nu)

# %%
U.u(0.1)

# %%
