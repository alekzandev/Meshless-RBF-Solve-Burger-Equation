#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
def R(t):
  return 0.5*np.sqrt((np.cos(5*t))**2 + (np.cos(2*t))**2 + (np.cos(t))**2)

def x(t):
  return R(t)*np.cos(t)

def y(t):
  return R(t)*np.sin(t)

#%%
n_boundary = 600
x_t = np.array([x(t) for t in range(0, n_boundary)])
y_t = np.array([y(t) for t in range(0, n_boundary)])
plt.scatter(x_t, y_t)