import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from mpl_toolkits.mplot3d import Axes3D

from halton_points import HaltonPoints


def sol_burgers_2D_homogeneous(X, t, nu=1, alpha=0.5):
    # first_comp = 3/4 - 1/4 * (1 / (1 + np.exp((4*y - 4*x - t) / (32*nu))))
    # second_comp = 3/4 + 1/4 * (1 / (1 + np.exp((4*y - 4*x - t) / (32*nu))))
    normx = np.linalg.norm(X, axis=1).reshape(-1, 1)
    first_comp = X[:, 0].reshape(-1, 1)/(t+alpha + (t+alpha)
                                         ** 2 * np.exp(normx/(4 * (alpha + t))))
    second_comp = X[:, 1].reshape(-1, 1)/(t+alpha + (t+alpha)
                                          ** 2 * np.exp(normx/(4 * (alpha + t))))

    return first_comp, second_comp


# n = 30
# interior_points = HaltonPoints(2, n).haltonPoints()
# x, y = interior_points[:, 0], interior_points[:, 1]
# x, y = np.hstack((-x, x)), np.hstack((-y, y))
#X, Y = np.meshgrid(x, y, sparse=True)
#t = 0.2

# comp_x, comp_y = sol_burgers_2D_homogeneous(interior_points, t)

#print(comp_x)
# z = comp_y
# fig = plt.figure()
# ax = fig.gca(projection='3d')

# # nodes = np.quantile(z, [i/6 for i in range(6)])
# colors = ["red", "darkorange", "gold", "lightseagreen", "cyan", "blue"]
# colors.reverse()
# cmap = LinearSegmentedColormap.from_list("mycmap", colors)
# surf = ax.plot_surface(x, y, z, cmap=cmap,
#                        linewidth=0, antialiased=False)

# #ax.plot_wireframe(x, y, z, rstride=10, cstride=10)
# z_min = np.min(z)
# z_max = np.max(z)
# ax.set_zlim(z_min, z_max)
# # ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# fig.colorbar(surf, shrink=0.5, aspect=10)
# plt.show()
