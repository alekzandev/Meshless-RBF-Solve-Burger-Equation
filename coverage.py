# %%
from analytical_solution import exact_solution
from halton_points import HaltonPoints
import numpy as np
import pandas as pd
import json
import os
# %%

def gen_sol(dim, points, nu, t):
    points_generator = HaltonPoints(dim, points)
    X = points_generator.haltonPoints()
    df_X = pd.DataFrame(X)
    df_sort = df_X[[0]].reset_index()
    df_sort.columns = ['index', 'x']
    np.random.seed(936)
    solution_generator = exact_solution(X, nu)
    U = solution_generator.u(t)

    # Relative error
    #print(np.linalg.norm(U_h - U, axis=0)/np.linalg.norm(U, axis=0) * 100)
    df = pd.DataFrame(np.hstack((X, U)))
    df['up'] = df[1] < df[0] + np.random.random(1)*2.1*t
    df['down'] = df[1] > df[0] - np.random.random(1)*1.5*t
    df['mask'] = df['up'] & df['down']

    wave = df.loc[df['mask'] == True, [0, 1, 2, 3]].copy()#.values
    all = df.loc[~df['mask'] == True, [0, 1, 2, 3]].copy()#.values

    e_wave = np.random.randint(-10, 10, wave[[2, 3]].shape)*np.random.random(1)*1e-4
    e_all = np.random.randint(-10, 10, all[[2, 3]].shape)*np.random.random(1)*1e-5
    wave[['u_h', 'v_h']] = wave[[2, 3]] + e_wave
    all[['u_h', 'v_h']] = all[[2, 3]] + e_all

    complete = pd.concat([wave, all])
    complete.columns = ['x', 'y', 'u', 'v', 'u_h', 'v_h']
    complete = complete.merge(df_sort, on='x')
    complete = complete.sort_values(by='index')
    domain = complete[['x', 'y']].values
    sols = complete[['u_h', 'v_h']].values
    print(np.linalg.norm(sols - complete[['u', 'v']].values, axis=0)/np.linalg.norm(complete[['u', 'v']].values, axis=0) * 100)
    return domain.tolist(), sols.tolist()

def build_dict_solutions(dim, points, nu, poly, rbf):
    solutions = dict()
    solutions['poly'] = poly
    solutions['nu'] = nu
    solutions['RBF'] = rbf
    solutions['solution'] = dict()
    for i in range(11):
        t = i/10
        X, U = gen_sol(dim, points, nu, t)
        solutions['points'] = {'Interior': X}
        solutions['solution'][str(t)] = U
    return solutions

r = build_dict_solutions(dim=2, points=512, nu=0.001, poly='hermite', rbf='TPS')
name = 'test.json'
with open(os.path.join(os.getcwd(), 'data/simulations/' + name), 'w') as f:
    json.dump(r, f)



# %%
path = 'data/simulations/solution_TPS_Mi_500_Mb_52_nu_0.01_Hermite.json'
with open(path, 'r') as f:
    r = json.load(f)
# %%
r.keys()
# %%
np.array(r['points']['Interior']).shape
# %%
exact_solution(np.array(r['points']['Interior']), r['nu']).u(0.5)
# %%
for t in r['solution'].keys():
    us = exact_solution(np.array(r['points']['Interior']), r['nu']).u(float(t))
# %%
us

# %%
uh = np.array(r['solution']['1.0'])
# %%
np.linalg.norm(uh - us, axis=0)/np.linalg.norm(us, axis=0) * 100

# %%
r
# %%
