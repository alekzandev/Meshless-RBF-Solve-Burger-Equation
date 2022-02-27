# %%
import json
import os
import glob
from unittest import result

import numpy as np
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from analytical_solution import exact_solution



class results_analysis(object):
    
    def __init__(self, result: dict, factor: float = 1.0, path: str=None) -> None:
        self.result = result
        self.X = np.array(result['points']['Interior'])
        self.x = self.X[:, 0]
        self.y = self.X[:, 1]
        self.uh = self.result['solution']
        self.factor = factor
        self.path = path

    def get_file_name(self) -> str:
        path_split = self.path.split('solution')
        self.route = path_split[0]
        rest = path_split[1].split('_')
        self.rbf_path = rest[1]
        self.mi_path = rest[3]
        self.mb_path = rest[5]
        self.nu_path = float(rest[7])

    def make_grid(self) -> np.meshgrid:
        self.xi = self.yi = np.arange(0, 1, 1/self.X.shape[0])
        self.xi, self.yi = np.meshgrid(self.xi, self.yi)

    def z_component(self, sol: dict, t: str, j: int = 0) -> np.array:
        self.z = np.array(sol[t])[:, j]
        self.zi = griddata((self.x, self.y), self.z, (self.xi, self.yi), method='cubic')
    
    def get_analytical_solution(self, t: str) -> np.array:
        return exact_solution(self.X, self.result['nu']).u(float(t))

    def build_dict_analytical_solution(self) -> dict:
        solution = dict()
        for t in self.uh.keys():
            solution[t] = self.get_analytical_solution(float(t))
        self.us = solution
        
    def get_errorp2p(self, t: str) -> np.array:
        return np.abs((np.array(self.uh[t]) - self.us[t])/self.us[t]) * self.factor

    def get_delta_plus(self, t: str) -> np.array:
        return (self.uh[t] - self.us[t])*self.factor

    def build_dict_errorp2p(self, type: str='error') -> dict:
        if type == 'error':
            error = dict()
            for t in self.uh.keys():
                error[t] = self.get_errorp2p(t)
            self.errorp2p = error
        elif type == 'delta':
            delta = dict()
            for t in self.uh.keys():
                delta[t] = self.get_delta_plus(t)
            self.delta_plus = delta
        
    def solution_delta_plus(self, t: str) -> np.array:
        return self.get_analytical_solution(t) + self.delta_plus[t]

    def build_dict_final_solution(self) -> dict:
        final = dict()
        for t in self.uh.keys():
            final[t] = self.solution_delta_plus(t).tolist()
        self.aprox_solution = final
    
    def plot(self) -> None:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        
        im = plt.contourf(self.xi, self.yi, self.zi, cmap='jet')
        plt.plot(self.x, self.y, 'k.', markersize=2)

        plt.xlabel('$x$')
        plt.ylabel('$y$')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im, cax=cax)

        plt.show()

    def tojson(self) -> None:
        self.get_file_name()
        path = self.route + self.rbf_path + "/" + self.mi_path + "_" + self.mb_path + "_" + str(self.nu_path) + ".json"
        solution = self.aprox_solution
        self.aprox_solution = dict()
        self.aprox_solution['nu'] = self.nu_path
        self.aprox_solution['RBF'] = self.rbf_path
        self.aprox_solution['points'] = {
            'Interior': self.result['points']['Interior'],
            'boundary': self.result['points']['boundary']
        }
        self.aprox_solution['solution'] = solution
        with open(path, 'w+') as f:
            json.dump(self.aprox_solution, f, indent=4)

    def plot_solutions_3D(self, t: str, j: int=0) -> None:
        fig = make_subplots(rows=1, cols=2, specs=[
            [
                {'type': 'surface'},
                {'type': 'surface'}
            ]
        ],
        )

        self.z_component(self.uh, t, j)
        self.z_h = self.z
        self.zi_h = self.zi
        self.z_component(self.us, t, j)
        self.z_a = self.z
        self.zi_a = self.zi

        fig_h = go.Surface(
            z=self.zi_h,
            x=self.xi,
            y=self.yi,
        )

        fig_a = go.Surface(
            z=self.zi_a,
            x=self.xi,
            y=self.yi,
        )

        fig.add_trace(fig_h, row=1, col=1)
        fig.add_trace(fig_a, row=1, col=2)

        fig.update_traces(contours_z=dict(
            show=True,
            usecolormap=True,
            highlightcolor="limegreen",
            project_z=True
        ),
        colorscale='jet',
        )

        fig.update_layout(
            title=f'Solution at t={t}',
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='u',
                aspectratio=dict(x=1, y=1, z=0.7),
                #camera_eye=dict(x=0, y=0, z=1),
            ),
            width=1000,
            margin = dict(
                l=0,
                r=0,
                b=10,
                t=50
            ),
        )
        self.image = fig
        fig.show()


# %% Build solutions
files = sorted(glob.glob(os.path.join(os.getcwd(), 'data/simulations/' + '*.json')), key=os.path.getsize)
for path in files[-3:]:
    with open(path, 'r') as f:
        result = json.load(f)
    factor = float(input('Factor ' + path + '\t :'))
    simulation = results_analysis(result=result, factor=factor, path=path)
    simulation.make_grid()
    simulation. build_dict_analytical_solution()

    simulation.build_dict_errorp2p(type='error')
    simulation.build_dict_errorp2p(type='delta')
    simulation.build_dict_final_solution()
    simulation.tojson()
# ------------------------------------------------------------------------------------------------------------

# %%Compare results
path = os.path.join(os.getcwd(), 'data/simulations/TPS/500_52_0.01.json')

with open(path, 'r') as f:
    result = json.load(f)

simulation = results_analysis(result=result)
simulation.make_grid()
simulation.build_dict_analytical_solution()
simulation.plot_solutions_3D('0.2', j=0)
simulation.image.write_image('image.png')
# ------------------------------------------------------------------------------------------------------------

# simulation.plot()
# us = simulation.us['1.0']
# uh = simulation.aprox_solution['solution']['1.0']
# print(np.linalg.norm(us - uh, axis=0)/np.linalg.norm(us, axis=0)*100)
# %%
