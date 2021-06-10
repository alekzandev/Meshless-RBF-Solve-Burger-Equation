# pylint: disable=E1101
# pylint: disable=all

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from Halton_Points import HaltonPoints


class terms_uh(object):
    '''
    x:  array
    beta: Natural number
    l: norm order (1,2,np.inf)
    '''

    def __init__(self, Mi, Mb, beta, c, x, nu=1, poly_b=np.array([1]),  rbf='TPS', l=2):
        self.Mi = Mi
        self.ni = Mi.shape[0]
        self.Mb = Mb
        self.nb = Mb.shape[0]
        self.d = Mi.shape[1]
        self.dm = poly_b.shape[0]
        self.beta = beta
        self.c = c
        self.x = x
        self.nu = nu
        self.rbf = rbf
        self.l = l

    def RBF(self, M):
        if self.rbf == 'TPS':
            return (-1)**(self.beta + 1) * M**(2*self.beta) * np.log(M+1e-20)
        elif self.rbf == 'MQ':
            return (-1)**self.beta * (self.c**2 + M ** 2)**self.beta
        elif self.rbf == 'radial_powers':
            return (-1)**int(self.beta/2) * self.norm_x(M)**self.beta
        elif self.rbf == 'rbf_ut':
            return (-1/self.c**3) * (np.exp(-self.c * self.norm_x(M)) + self.c * self.norm_x(M))

    def K2(self):
        return self.RBF(self.norm_x(self.matrix_K(self.Mb)))

    def poly_basis(self):
        return 1  # np.array([1])

    def O2(self):
        return np.zeros((self.dm, self.dm))

    def Q2(self):
        return np.ones((self.nb, self.dm)).reshape(-1, 1)

    def A(self):
        A1 = np.hstack((self.K2(), self.Q2()))
        A2 = np.hstack((self.Q2().transpose(), self.O2()))
        return np.vstack((A1, A2))

    def K1(self):
        return self.RBF(self.norm_x(self.matrix_K(self.Mi)))

    def M(self):
        return self.RBF(self.norm_x(self.matrix_M()))

    def Q1(self):
        return np.ones((self.ni, self.dm))

    def B(self):
        return np.matmul(np.hstack((self.M(), self.Q1())), np.linalg.inv(self.A()))

    def Sinv(self):
        MQ1 = np.hstack((self.M(), self.Q1()))
        return np.linalg.inv(self.K1() - np.matmul(np.matmul(MQ1, np.linalg.inv(self.A())), MQ1.transpose()))

    def C(self):
        return np.linalg.inv(self.A()) + np.matmul(np.matmul(self.B().transpose(), self.Sinv()), self.B())

    def lambda_m(self, phi_m):
        self.op = "lambda"
        return phi_m(self.norm_x(self.matrix_lamb_gamm_thet(self.Mi))).reshape(-1, 1)

    def gamma_m(self, phi_m):
        self.op = "gamma"
        if phi_m == self.laplacian_TPS:
            # M = self.matrix_lamb_gamm_thet(self.Mb)
            # M_norm = self.norm_x(M).reshape(-1,1)
            # # print (" M={} \n M_norm={}".format(M.shape, M_norm.shape))
            # print(M_norm)
            # return phi_m(M_norm)
            return phi_m(self.norm_x(self.matrix_lamb_gamm_thet(self.Mb)).reshape(-1, 1))
        elif phi_m == self.grad_TPS:
            return phi_m(self.norm_x(self.matrix_lamb_gamm_thet(self.Mb)))

    def theta_m(self):
        return np.ones((self.dm, 1))

    def a_m_op(self, lin_op):
        if lin_op == self.laplacian_TPS:
            return np.matmul(self.Sinv(), np.matmul(self.B(), np.vstack((self.gamma_m(lin_op), self.theta_m()))) - self.lambda_m(lin_op))
        elif lin_op == self.grad_TPS:
            gamma = self.gamma_m(lin_op)
            print(self.lambda_m(lin_op))
            am_x = np.matmul(self.Sinv(), np.matmul(self.B(), np.vstack((gamma[:,0].reshape(-1,1), self.theta_m()))) - self.lambda_m(lin_op))
            am_y = np.matmul(self.Sinv(), np.matmul(self.B(), np.vstack((gamma[:,1].reshape(-1,1), self.theta_m()))) - self.lambda_m(lin_op))
            return np.matmul(self.Sinv(), np.matmul(self.B(), np.vstack((self.gamma_m(lin_op), self.theta_m()))) - self.lambda_m(lin_op))

    def a_m(self):
        return self.a_m_op(self.RBF)

    def lap_am(self):
        return self.a_m_op(self.laplacian_TPS)

    def grad_am(self):
        return self.a_m_op(self.grad_TPS)

    def b_m(self):
        return np.matmul(self.C(), np.vstack((self.gamma_m(), self.theta_m()))) - np.matmul(self.B().transpose(), np.matmul(self.Sinv(), self.lambda_m()))


class operators(terms_uh):

    def norm_x(self, M):
        return np.linalg.norm(M, self.l, axis=-1)

    def matrix_K(self, M):
        n=M.shape[0]
        my_matrix=list()
        for j in M:
            for l in M:
                my_matrix.append(j-l)
        return np.array(my_matrix).reshape(n, n, self.d)

    def matrix_M(self):
        my_matrix=list()
        for r_i in self.Mi:
            for r_b in self.Mb:
                my_matrix.append(r_i-r_b)
        return np.array(my_matrix).reshape(self.ni, self.nb, self.d)

    def matrix_lamb_gamm_thet(self, M):
        return self.x - M


class assembled_matrix(operators):
    def grad_TPS(self, M):
        op=self.op
        if op == "lambda":
            comp_x=(M**(2*self.beta-1)).reshape(-1, 1) *\
                self.matrix_lamb_gamm_thet(self.Mi)[:, 0].reshape(-1, 1) *\
                (2*self.beta*np.log(M+1e-20)+1).reshape(-1, 1)
            comp_y=(M**(2*self.beta-1)).reshape(-1, 1) *\
                self.matrix_lamb_gamm_thet(self.Mi)[:, 1].reshape(-1, 1) *\
                (2*self.beta*np.log(M+1e-20)+1).reshape(-1, 1)
        elif op == "gamma":
            comp_x=(M**(2*self.beta-1)).reshape(-1, 1) *\
                self.matrix_lamb_gamm_thet(self.Mb)[:, 0].reshape(-1, 1) *\
                (2*self.beta*np.log(M+1e-20)+1).reshape(-1, 1)
            comp_y=(M**(2*self.beta-1)).reshape(-1, 1) *\
                self.matrix_lamb_gamm_thet(self.Mb)[:, 1].reshape(-1, 1) *\
                (2*self.beta*np.log(M+1e-20)+1).reshape(-1, 1)
        return np.hstack((comp_x, comp_y))

    def laplacian_TPS(self, M):
        return M**(2*self.beta-2) * (4*self.beta*(self.beta*np.log(M)+1))

    def F_m(self, init_val):
        return self.nu * self.lap_am().T


class exact_solution(object):
    def __init__(self, A=None, t=None, nu=None, radius=0.15, c_x=0.5, c_y=0.5):
        self.A=A
        self.x=A[:, 0]
        self.y=A[:, 1]
        self.t=t
        self.nu=nu
        self.radius=radius
        self.c_x=c_x
        self.c_y=c_y

    def domain(self):
        dist_from_center=np.sqrt(
            (self.x - self.c_x)**2 + (self.y - self.c_y)**2)
        mask=dist_from_center < self.radius
        # mask_x = np.logical_and(
        #     self.x > self.c_x - self.radius, self.x < self.c_x + self.radius)
        # A = self.A.copy()
        # A = A[mask]
        # mask_y = np.logical_and(
        #     A[:, 1] > self.c_y - self.radius, A[:, 1] < self.c_y + self.radius)
        return self.A[mask]

    def hopf_cole_transform(self):
        Omega=self.domain()
        x, y=Omega[:, 0], Omega[:, 1]
        # x, y = np.meshgrid(x, y)
        u=3/4 - 1/4 * \
            (1 / (1 + np.exp((4*y - 4*x - self.t) / (32*self.nu))))
        v=3/4 + 1/4 * \
            (1 / (1 + np.exp((4*y - 4*x - self.t) / (32*self.nu))))

        return u, v


x_i=np.array([2, 3, 2]).reshape(-1, 1)
y_i=np.array([1, 0, 3]).reshape(-1, 1)
Mi=HaltonPoints(2, 10).haltonPoints()  # np.hstack((x_i, y_i))
Mb=np.array([
    [0., 0.],
    [0., 0.5],
    [0., 1.],
    [0.5, 1.],
    [1., 1.],
    [1., 0.5],
    [1., 0.],
    [0.5, 0.]
])
x=np.array([1.1, 2.3])

M1=np.array([
    [
        [0, 0],
        [-1, -1],
        [0, -2]
    ],

    [
        [1, -1],
        [0, 0],
        [1, -3]
    ],

    [
        [0, 2],
        [-1, -3],
        [0, 0]
    ]
])

# print(implementation(Mi, Mb, 2, 0.3).K1())
init_val=np.array([0.4, -0.2])

# print(np.linalg.norm(x-Mb, axis=-1).reshape(-1,1))
amm=assembled_matrix(Mi, Mb, 2, 0.1, x).grad_am()
print(amm)

# print(Mi.shape)
# print(x)

# print(x-Mb)
# f =implementation(Mi, Mb, 2, 0.1, x)
# print(Mb)
# M = f.RBF(Mb)
# print(np.linalg.norm(M, 2, axis=-1))

# M2 = np.linalg.norm(Mb, 2, axis = -1)
# print(M2)
# M2 = f.RBF(M2)


# print(np.linalg.norm(x-Mb, axis=-1).reshape(-1, 1))
# u1, u2 = exact_solution(A=Mi, t=0.2, nu=1.1).hopf_cole_transform()
# solution_domain = exact_solution(
#     A=Mi, radius=0.2, t=0.2, nu=1.01, c_x=0.6, c_y=0.7)
# Omega = solution_domain.domain()
# u, v = solution_domain.hopf_cole_transform()

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# X, Y = Omega[:, 0], Omega[:, 1]
# Z = v.copy()

# print(X.shape, Y.shape, Z.shape)
# print(Omega)
# plt.scatter(Omega[:, 0], Omega[:, 1])
# plt.scatter(X, Y)
# plt.xlim(0, 1)
# plt.ylim(0, 1)


# ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
# cs = ax.contourf(X, Y, Z)
# ax.plot3D(X, Y, Z)
# ax.contour(cs, colors = 'k')
# ax.grid(c='k', ls='-', alpha=0.3)
# ax.clabel(cs, inline=True, fontsize=10)
# cset = ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
# cset = ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
# cset = ax.contourf(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)
# ax.plot_trisurf(x, y, Z, linewidth=0.2, antialiased=True, cmap=plt.cm.Spectral)
# ax.set_xlabel('X')
# #ax.set_xlim(-40, 40)
# ax.set_ylabel('Y')
# #ax.set_ylim(-40, 40)
# ax.set_zlabel('Z')
# #ax.set_zlim(-100, 100)

# plt.show()
