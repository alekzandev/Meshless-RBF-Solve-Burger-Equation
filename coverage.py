# %%
from analytical_solution import exact_solution
from halton_points import HaltonPoints
import numpy as np
import pandas as pd
import json
# %%

def gen_sol(dim, points, nu, t):
    points_generator = HaltonPoints(dim, points)
    X = points_generator.haltonPoints()
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
    
    return complete[['x', 'y']].values.tolist(), complete[['u_h', 'v_h']].values.tolist()

def build_dict_solutions(dim, points, nu, poly, rbf):  
    # solutions = {
    #     'poly': '',
    #     'nu': '',
    #     'rbf': '',
    #     'points': '',
    #     'solution': '',
    # }
    solutions = dict()
    solutions['poly'] = poly
    solutions['nu'] = nu
    solutions['RBF'] = rbf
    solutions['points'] = {
        'points': {
            'Interior': points
        }
    }
    solutions['solution'] = dict()
    for i in range(1,200):
        t = i/100
        solutions['solution'][str(t)] = gen_sol(dim, points, nu, t)
    return solutions

r = build_dict_solutions(dim=2, points=52, nu=0.01, poly='hermite', rbf='TPS')
name = 'test.json'
with open(name, 'w') as f:
    json.dump(r, f)



# %%
