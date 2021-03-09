import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from mpl_toolkits.mplot3d import Axes3D

from Halton_Points import haltonPoints


def sol_burgers_2D_homogeneous(x, y, t, nu):
    first_comp = 3/4 - 1/4 * (1 / (1 + np.exp((4*y - 4*x - t) / (32*nu))))
    second_comp = 3/4 + 1/4 * (1 / (1 + np.exp((4*y - 4*x - t) / (32*nu))))

    return first_comp, second_comp


n = 60
colloc_points = haltonPoints(2, int(n/2))
x, y = colloc_points[:, 0], colloc_points[:, 1]
x, y = np.hstack((-x, x)), np.hstack((-y, y))
x, y = np.meshgrid(x, y)
t = 0.1
nu = 2

comp_x, comp_y = sol_burgers_2D_homogeneous(x, y, t, nu)


z = comp_x
fig = plt.figure()
ax = fig.gca(projection='3d')

# nodes = np.quantile(z, [i/6 for i in range(6)])
colors = ["red", "darkorange", "gold", "lightseagreen", "cyan", "blue"]
colors.reverse()
cmap = LinearSegmentedColormap.from_list("mycmap", colors)
surf = ax.plot_surface(x, y, z, cmap=cmap,
                       linewidth=0, antialiased=False)

#ax.plot_wireframe(x, y, z, rstride=10, cstride=10)
z_min = np.min(z)
z_max = np.max(z)
ax.set_zlim(z_min, z_max)
# ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()
