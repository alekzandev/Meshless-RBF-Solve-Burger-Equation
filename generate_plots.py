# %%
from distutils.command.build import build
import json
import os
import glob
from statistics import mode
from unittest import result

import numpy as np
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from analytical_solution import exact_solution

from PIL import Image



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

    # def plot_solutions_3D(self, t: str, j: int=0) -> None:
    #     fig = make_subplots(rows=1, cols=2, specs=[
    #         [
    #             {'type': 'surface'},
    #             {'type': 'surface'}
    #         ]
    #     ],
    #     )

    #     self.z_component(self.uh, t, j)
    #     self.z_h = self.z
    #     self.zi_h = self.zi
    #     self.z_component(self.us, t, j)
    #     self.z_a = self.z
    #     self.zi_a = self.zi

    #     fig_h = go.Surface(
    #         z=self.zi_h,
    #         x=self.xi,
    #         y=self.yi,
    #     )

    #     fig_a = go.Surface(
    #         z=self.zi_a,
    #         x=self.xi,
    #         y=self.yi,
    #     )

    #     fig.add_trace(fig_h, row=1, col=1)
    #     fig.add_trace(fig_a, row=1, col=2)

    #     fig.update_traces(contours_z=dict(
    #         show=True,
    #         usecolormap=True,
    #         highlightcolor="limegreen",
    #         project_z=True
    #     ),
    #     colorscale='jet',
    #     )

    #     fig.update_layout(
    #         title=f'Solution at t={t}',
    #         scene=dict(
    #             xaxis_title='x',
    #             yaxis_title='y',
    #             zaxis_title='u',
    #             aspectratio=dict(x=1, y=1, z=0.7),
    #             #camera_eye=dict(x=0, y=0, z=1),
    #         ),
    #         width=1000,
    #         margin = dict(
    #             l=0,
    #             r=0,
    #             b=10,
    #             t=50
    #         ),
    #     )
    #     self.image = fig
    #     fig.show()

    def plot_solutions_3D(self, t: str, j: int=0, type: str='numerical') -> None:
        fig = make_subplots(rows=1, cols=1, specs=[
            [
                {'type': 'surface'},
                #{'type': 'surface'}
            ]
        ],
        )
        self.type = type
        self.comp = j
        if self.comp == 0:
            label_ax = 'u'
            eye_y = 2.1
            eye_x = 2.5
            center_y = 0.2
        else:
            label_ax = 'v'
            eye_y = 2.75
            eye_x = 1.5
            center_y = 0.3
        if self.type == 'analytical':
            color_bar = False
            self.z_component(self.us, t, j)
            self.z_a = self.z
            self.zi_a = self.zi

            fig_a = go.Surface(
                z=self.zi_a,
                x=self.xi,
                y=self.yi,
            )
            fig.add_trace(fig_a, row=1, col=1)
        
        elif self.type == 'numerical':
            color_bar = True
            self.z_component(self.uh, t, j)
            self.z_h = self.z
            self.zi_h = self.zi
        
            fig_h = go.Surface(
                z=self.zi_h,
                x=self.xi,
                y=self.yi,
            )
            fig.add_trace(fig_h, row=1, col=1)
        
        
        fig.update_traces(contours_z=dict(
            show=True,
            usecolormap=True,
            highlightcolor="limegreen",
            project_z=True,
        ),
        colorscale='jet',
        colorbar=dict(
            lenmode='fraction',
            len=0.8,
            orientation='v',
        ),
        showscale=color_bar,
        )

        fig.update_layout(
            #title=f't={t}',
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title=label_ax,
                aspectratio=dict(x=1.5, y=1.5, z=1.),
                camera_center = dict(x=0.4, y=center_y, z=0.1),
                camera_eye=dict(x=eye_x, y=eye_y, z=1.5),
            ),
            width=600,
            height=600,
            margin = dict(
                l=0,
                r=0,
                b=0,
                t=0,
                pad=50
            ),
            #plot_bgcolor='rgb(0,0,0,0)',
        )
        self.image = fig
        fig.show()

class build_images(results_analysis):

    def save_image(self, file: str, t: str) -> None:
        path = os.path.join(os.getcwd(), file)
        name = path.split('simulations')[-1].split('.json')[0].replace('/', '_')[1:].split('_')
        name = name[0] + '_' + name[1] + '_' + name[-1] + '_' + t + '_' + self.type + '_' + str(self.comp) +'.png'
        path_im = os.path.join(os.getcwd(), 'data/images', name)
        self.image.write_image(path_im)

# %% Build solutions
# files = sorted(glob.glob(os.path.join(os.getcwd(), 'data/simulations/' + '*.json')), key=os.path.getsize)
# for path in files[-3:]:
#     with open(path, 'r') as f:
#         result = json.load(f)
#     factor = float(input('Factor ' + path + '\t :'))
#     simulation = results_analysis(result=result, factor=factor, path=path)
#     simulation.make_grid()
#     simulation. build_dict_analytical_solution()

#     simulation.build_dict_errorp2p(type='error')
#     simulation.build_dict_errorp2p(type='delta')
#     simulation.build_dict_final_solution()
#     simulation.tojson()
# ------------------------------------------------------------------------------------------------------------

# %%Compare results

file = 'data/simulations/MQ/500_52_0.01.json'
path = os.path.join(os.getcwd(), file)
with open(path, 'r') as f:
    result = json.load(f)

simulation = build_images(result=result)
simulation.make_grid()
simulation.build_dict_analytical_solution()
timegrid = ['0.1', '0.5', '1.0']
typeu = ['analytical', 'numerical']
comp = [0, 1]

for t in timegrid:
    print(t)
    for typ in typeu:
        for j in comp:
            simulation.plot_solutions_3D(t, j, typ)
            simulation.save_image(file, t)
# ------------------------------------------------------------------------------------------------------------

#%%

file = 'data/simulations/MQ/500_52_0.01.json'
path = os.path.join(os.getcwd(), file)
with open(path, 'r') as f:
    result = json.load(f)

simulation = build_images(result=result)
simulation.make_grid()
simulation.build_dict_analytical_solution()
#%%

nu = '0.01'
comp = '0'
pol = 'MQ'

for t in ['0.1', '0.5', '1.0']:

    img_path = os.path.join(os.getcwd(), f'data/images/{pol}')

    # name_im1 = f'data/images/TPS_{pol}_{nu}_{t}_analytical_{comp}.png'
    # name_im2 = f'data/images/TPS_{pol}_{nu}_{t}_numerical_{comp}.png'
    name_im1 = f'data/images/{pol}_500_{nu}_{t}_analytical_{comp}.png'
    name_im2 = f'data/images/{pol}_500_{nu}_{t}_numerical_{comp}.png'

    image1 = Image.open(name_im1)
    image2 = Image.open(name_im2)
    new_image = Image.new('RGB', (2*image1.width, image1.height), (255, 255, 255))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (image1.width, 0))
    new_image.show()
    new_image.save(os.path.join(img_path, f'{nu}_{t}_{comp}.png'))



# %%

# simulation.plot()
# us = simulation.us['1.0']
# uh = simulation.aprox_solution['solution']['1.0']
# print(np.linalg.norm(us - uh, axis=0)/np.linalg.norm(us, axis=0)*100)
# %%
simulation.X
#exact_solution()
# %%
