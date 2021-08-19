# pylint: disable=E1101
# pylint: disable=all

#import matplotlib.pyplot as plt
import numpy as np

from halton_points import HaltonPoints

#from matplotlib import cm
#from matplotlib.ticker import LinearLocator


class terms_uh(object):
    '''
    x:  array
    beta: Natural number
    l: norm order (1,2,np.inf)
    '''

    def __init__(self, Mb, npnts, c=1, beta=1, nu=0.1, mu=1., epsilon=4.5, poly_b=np.array([1]),  rbf='TPS', l=2):
        # np.random.seed(9936)
        self.Mb = Mb
        self.nb = Mb.shape[0]
        self.d = self.Mb.ndim
        self.x = np.empty((2, ))
        self.poly_b = poly_b
        self.dm = poly_b.shape[0]
        self.npnts = npnts
        self.Mi = HaltonPoints(self.d, self.npnts).haltonPoints()
        self.ni = self.Mi.shape[0]
        self.beta = beta
        self.c = c
        self.nu = nu
        self.mu = mu
        self.epsilon = epsilon
        self.rbf = rbf
        self.l = l
        self.alpha = 1

    def RBF(self, M):
        '''
        RBF
        '''
        if self.rbf == 'TPS':
            return (-1)**(self.beta + 1) * M**(2*self.beta) * np.log(M+1e-80)
        elif self.rbf == 'MQ':
            self.beta = 3/2
            return (-1)**(int(self.beta)+1) * (self.c**2 + self.epsilon**2 * M ** 2)**self.beta
        elif self.rbf == 'radial_powers':
            return (-1)**int(self.beta/2) * self.norm_x(M)**self.beta
        elif self.rbf == 'rbf_ut':
            return (-1/self.c**3) * (np.exp(-self.c * self.norm_x(M)) + self.c * self.norm_x(M))

    def K2(self):
        return self.RBF(self.norm_x(self.matrix_K(self.Mb))) + np.eye(self.nb)

    def q(self, x, y, i):
        # if i == 1:
        #     return 2*y
        # elif i == 2:
        #     return 4*x*y
        # elif i == 3:
        #     return 8*x*y**2 - 4*x
        
        if i == 1:
            return np.ones(x.shape)
        elif i == 2:
            return 2*x
        elif i == 3:
            return 4*x*y

    def lap_q(self, x, i):
        # if i == 1:
        #     return 0
        # elif i == 2:
        #     return 0
        # elif i == 3:
        #     return 16*x
        if i == 1:
            return 0
        elif i == 2:
            return 0
        elif i == 3:
            return 0

    def grad_q(self, x, y, i):
        # if i == 1:
        #     return np.hstack((0, 2)).reshape(1, -1)
        # elif i == 2:
        #     return np.hstack((4 * y, 4 * x)).reshape(1, -1)
        # elif i == 3:
        #     return np.hstack((8 * y**2 - 4, 16 * x * y)).reshape(1, -1)
        if i == 1:
            return np.hstack((0, 0)).reshape(1, -1)
        elif i == 2:
            return np.hstack((2, 0)).reshape(1, -1)
        elif i == 3:
            return np.hstack((4 * y, 4 * x)).reshape(1, -1)

    def poly_basis(self, M, i):
        # return M[:, 0].reshape(-1, 1) * self.poly_b[i, 0] + M[:, 1].reshape(-1, 1) * self.poly_b[i, 1] + self.poly_b[i, 2]
        return self.q(M[:, 0].reshape(-1, 1), M[:, 1].reshape(-1, 1), i)

    def O2(self):
        return np.zeros((self.dm, self.dm)) + 0.001*np.eye(self.dm)

    def O1(self):
        return np.zeros((self.dm, self.d))

    def G(self, t):
        alpha = self.alpha
        nX = np.linalg.norm(self.Mb, axis=1).reshape(-1, 1)
        return self.Mb/((t+alpha)+(t+alpha)**2 * np.exp(nX**2/(4*(alpha+t))))
        # u = (self.Mb[:,0] + self.Mb[:,1] - 2*self.Mb[:,0]*t)/(1-2*t**2)
        # v = (self.Mb[:,0] - self.Mb[:,1] - 2*self.Mb[:,1]*t)/(1-2*t**2)
        # return np.hstack((u.reshape(-1,1), v.reshape(-1,1)))

    def G_tilde(self, t):
        return np.vstack((self.G(t), self.O1()))

    def Q2(self):
        col1 = self.poly_basis(self.Mb, 1)
        col2 = self.poly_basis(self.Mb, 2)
        col3 = self.poly_basis(self.Mb, 3)
        # return np.ones((self.nb, self.dm))  # .reshape(-1, 1)
        return np.hstack((col1, col2, col3))

    def A(self):
        A1 = np.hstack((self.K2(), self.Q2()))
        A2 = np.hstack((self.Q2().transpose(), self.O2()))
        return np.vstack((A1, A2))

    def K1(self):
        return self.RBF(self.norm_x(self.matrix_K(self.Mi))) + np.eye(self.ni)

    def M(self):
        return self.RBF(self.norm_x(self.matrix_M()))

    def Q1(self):
        col1 = self.poly_basis(self.Mi, 1)
        col2 = self.poly_basis(self.Mi, 2)
        col3 = self.poly_basis(self.Mi, 3)
        # return np.ones((self.ni, self.dm))
        return np.hstack((col1, col2, col3))

    def ACaps(self):
        r1 = np.hstack((self.K1(), self.M(), self.Q1()))
        r2 = np.hstack((self.M().T, self.K2(), self.Q2()))
        r3 = np.hstack((self.Q1().T, self.Q2().T, self.O2()))
        return np.vstack((r1, r2, r3))

    def B(self):
        return np.matmul(np.hstack((self.M(), self.Q1())), np.linalg.inv(self.A()))

    def Sinv(self):
        MQ1 = np.hstack((self.M(), self.Q1()))
        return np.linalg.inv(self.K1() - np.matmul(np.matmul(MQ1, np.linalg.inv(self.A())), MQ1.transpose()))

    def C(self):
        return np.linalg.inv(self.A()) + np.matmul(np.matmul(self.B().transpose(), self.Sinv()), self.B())

    def lambda_m(self, phi_m):
        self.op = "lambda"
        if phi_m == self.laplacian_TPS or phi_m == self.RBF:
            return phi_m(self.norm_x(self.matrix_lamb_gamm_thet(self.Mi))).reshape(-1, 1)
        elif phi_m == self.grad_TPS:
            return phi_m(self.norm_x(self.matrix_lamb_gamm_thet(self.Mi)))

    def gamma_m(self, phi_m):
        self.op = "gamma"
        if phi_m == self.laplacian_TPS or phi_m == self.RBF:
            return phi_m(self.norm_x(self.matrix_lamb_gamm_thet(self.Mb)).reshape(-1, 1))
        elif phi_m == self.grad_TPS:
            return phi_m(self.norm_x(self.matrix_lamb_gamm_thet(self.Mb)))

    def theta_m(self, phi_m):
        if phi_m == self.RBF:
            row1 = self.q(self.x[0], self.x[1], 1)
            row2 = self.q(self.x[0], self.x[1], 2)
            row3 = self.q(self.x[0], self.x[1], 2)
            return np.vstack((row1, row2, row3))
        elif phi_m == self.laplacian_TPS:
            row1 = self.lap_q(self.x[0], 1)
            row2 = self.lap_q(self.x[0], 2)
            row3 = self.lap_q(self.x[0], 3)
            return np.vstack((row1, row2, row3))
        else:
            row1 = self.grad_q(self.x[0], self.x[1], 1)
            row2 = self.grad_q(self.x[0], self.x[1], 2)
            row3 = self.grad_q(self.x[0], self.x[1], 3)
            return np.vstack((row1, row2, row3))

            # row1 = self.x[0] * self.poly_b[0, 0] + self.x[1] * \
            #     self.poly_b[0, 1] + self.poly_b[0, 2]
            # row2 = self.x[0] * self.poly_b[1, 0] + self.x[1] * \
            #     self.poly_b[1, 1] + self.poly_b[1, 2]
            # row3 = self.x[0] * self.poly_b[2, 0] + self.x[1] * \
            #     self.poly_b[2, 1] + self.poly_b[2, 2]
            # return np.vstack((row1, row2, row3))
        # return np.ones((self.dm, 1))

    def a_m_op(self, lin_op):
        if lin_op == self.laplacian_TPS or lin_op == self.RBF:
            return -np.matmul(
                self.Sinv(),
                np.matmul(
                    self.B(),
                    np.vstack((self.gamma_m(lin_op), self.theta_m(lin_op)))
                ) - self.lambda_m(lin_op)
            )
        elif lin_op == self.grad_TPS:
            return -np.matmul(
                self.Sinv(),
                np.matmul(
                    self.B(),
                    np.vstack((self.gamma_m(lin_op), self.theta_m(lin_op)))
                ) - self.lambda_m(lin_op)
            )

    def b_m_op(self, lin_op):
        if lin_op == self.laplacian_TPS or lin_op == self.RBF:
            return np.matmul(
                self.C(),
                np.vstack((self.gamma_m(lin_op), self.theta_m(lin_op)))
            )\
                - np.matmul(self.B().transpose(),
                            np.matmul(
                    self.Sinv(),
                    self.lambda_m(lin_op)
                )
            )
        elif lin_op == self.grad_TPS:
            return np.matmul(
                self.C(),
                np.vstack((self.gamma_m(lin_op), self.theta_m(lin_op)))
            )\
                - np.matmul(self.B().transpose(),
                            np.matmul(
                    self.Sinv(),
                    self.lambda_m(lin_op)
                )
            )

    def a_m(self):
        return self.a_m_op(self.RBF)

    def lap_am(self):
        return self.a_m_op(self.laplacian_TPS)

    def grad_am(self):
        return self.a_m_op(self.grad_TPS)

    def b_m(self):
        return self.b_m_op(self.RBF)

    def lap_bm(self):
        return self.b_m_op(self.laplacian_TPS)

    def grad_bm(self):
        return self.b_m_op(self.grad_TPS)


class operators(terms_uh):

    def norm_x(self, M):
        return np.linalg.norm(M, self.l, axis=-1)

    def matrix_K(self, M):
        n = M.shape[0]
        my_matrix = list()
        for j in M:
            for l in M:
                my_matrix.append(j-l)
        return np.array(my_matrix).reshape(n, n, self.d)

    def matrix_M(self):
        my_matrix = list()
        for r_i in self.Mi:
            for r_b in self.Mb:
                my_matrix.append(r_i-r_b)
        return np.array(my_matrix).reshape(self.ni, self.nb, self.d)

    def matrix_lamb_gamm_thet(self, M):
        return self.x - M


class assembled_matrix(operators):
    def grad_TPS(self, M):
        '''
        RBF gradient
        '''
        op = self.op
        if self.rbf == 'TPS':
            if op == "lambda":
                comp_x = (M**(2*self.beta-1)).reshape(-1, 1) *\
                    self.matrix_lamb_gamm_thet(self.Mi)[:, 0].reshape(-1, 1) *\
                    (2*self.beta*np.log(M+1e-20)+1).reshape(-1, 1)
                comp_y = (M**(2*self.beta-1)).reshape(-1, 1) *\
                    self.matrix_lamb_gamm_thet(self.Mi)[:, 1].reshape(-1, 1) *\
                    (2*self.beta*np.log(M+1e-20)+1).reshape(-1, 1)
            elif op == "gamma":
                comp_x = (M**(2*self.beta-1)).reshape(-1, 1) *\
                    self.matrix_lamb_gamm_thet(self.Mb)[:, 0].reshape(-1, 1) *\
                    (2*self.beta*np.log(M+1e-20)+1).reshape(-1, 1)
                comp_y = (M**(2*self.beta-1)).reshape(-1, 1) *\
                    self.matrix_lamb_gamm_thet(self.Mb)[:, 1].reshape(-1, 1) *\
                    (2*self.beta*np.log(M+1e-20)+1).reshape(-1, 1)
            return np.hstack((comp_x, comp_y))

        elif self.rbf == 'MQ':
            if op == 'lambda':
                xy = self.matrix_lamb_gamm_thet(self.Mi)
                comp_x = 3 * self.epsilon**2 *\
                    xy[:, 0].reshape(-1, 1) * np.sqrt(self.epsilon **
                                                      2 * M**2 + 1).reshape(-1, 1)
                comp_y = 3 * self.epsilon**2 *\
                    xy[:, 1].reshape(-1, 1) * np.sqrt(self.epsilon **
                                                      2 * M**2 + 1).reshape(-1, 1)
            elif op == 'gamma':
                xy = self.matrix_lamb_gamm_thet(self.Mb)
                comp_x = 3 * self.epsilon**2 *\
                    xy[:, 0].reshape(-1, 1) * np.sqrt(self.epsilon **
                                                      2 * M**2 + 1).reshape(-1, 1)
                comp_y = 3 * self.epsilon**2 *\
                    xy[:, 1].reshape(-1, 1) * np.sqrt(self.epsilon **
                                                      2 * M**2 + 1).reshape(-1, 1)
            return np.hstack((comp_x, comp_y))

    def laplacian_TPS(self, M):
        '''
        RBF laplacian
        '''
        if self.rbf == 'TPS':
            return M**(2*self.beta-2) * (4*self.beta*(self.beta*np.log(M+1e-20)+1))
        elif self.rbf == 'MQ':
            return 3 * self.epsilon**2 * ((3 * self.epsilon**2 * M**2 + 2)/np.sqrt(self.epsilon**2 * M**2 + 1))

    def X_0(self):
        alpha = self.alpha
        # c1 = np.sin(np.pi * self.Mi[:,0]) * np.cos(np.pi * self.Mi[:,1])
        # c2 = np.sin(np.pi * self.Mi[:,1]) * np.cos(np.pi * self.Mi[:,0])
        n = np.linalg.norm(self.Mi, axis=-1)
        c1 = self.Mi[:, 0]/(alpha + (alpha ** 2) * np.exp((n ** 2)/(4*alpha)))
        c2 = self.Mi[:, 1]/(alpha + (alpha ** 2) * np.exp((n ** 2)/(4*alpha)))
        # c1 = self.Mi[:, 0] + self.Mi[:, 1]
        # c2 = self.Mi[:, 0] - self.Mi[:, 1]
        return np.hstack((c1.reshape(-1, 1), c2.reshape(-1, 1)))

    def F_m(self, X0, t):
        # return np.matmul(self.nu * self.lap_am().T - np.matmul(self.mu * self.a_m().T, np.matmul(X0, self.grad_am().T)), X0)
        return self.nu * (np.matmul(
            self.lap_am().T,
            X0
        )
            + np.matmul(
            self.lap_bm().T,
            self.G_tilde(t)
        )
        )\
            - self.mu * (
            np.matmul(
                np.matmul(
                    self.a_m().T,
                    X0
                )
                + np.matmul(
                    self.b_m().T,
                    self.G_tilde(t)
                ),
                np.matmul(
                    self.grad_am().T,
                    X0
                )
                + np.matmul(
                    self.grad_bm().T,
                    self.G_tilde(t)
                )
            )
        )
    # def F_m(self, X0):
    #     return np.matmul(self.nu * self.lap_am().T, X0)

    def F0(self):
        for x in self.Mi:
            yield self.F_m(self.X_0, x)
    # def J(self, X0):
    #     return np.matmul(
    #         self.nu * self.lap_am().T -
    #         np.matmul(
    #             self.a_m().T,
    #             np.matmul(
    #                 X0,
    #                 self.grad_am().T)),
    #         np.ones((self.ni, self.d))) - \
    #         np.matmul(
    #         np.matmul(
    #             self.a_m().T,
    #             np.matmul(
    #                 np.ones((self.ni, self.d)),
    #                 self.grad_am().T)),
    #         X0)

    def J(self):
        return self.nu * self.lap_am().T


class stabillity(terms_uh):
    def cond_num(self, A):
        _, s, __ = np.linalg.svd(A)
        return max(s)/min(s), np.linalg.det(A)

    def qX(self):
        M = np.vstack((self.Mi, self.Mb))
        dists = list()
        for ii, i in enumerate(M):
            for jj, j in enumerate(M):
                if ii != jj:
                    dists.append(np.linalg.norm(i-j))
        return 0.5*min(dists)

    def G_q(self):
        return self.qX()**(self.beta-(self.d-1)/2) * np.exp(-12.76*self.d/self.qX())


class initial_condition(assembled_matrix):
    def F0(self):
        for x in self.Mi:
            assem_matrix = self.assembled_matrix(x=x)
            X0 = assem_matrix.X_0()
            yield assem_matrix.F_m(X0)[0]


class exact_solution(object):
    def __init__(self, A=None, t=None, nu=None, radius=0.15, c_x=0.5, c_y=0.5):
        self.A = A
        self.x = A[:, 0]
        self.y = A[:, 1]
        self.t = t
        self.nu = nu
        self.radius = radius
        self.c_x = c_x
        self.c_y = c_y

    def domain(self):
        dist_from_center = np.sqrt(
            (self.x - self.c_x)**2 + (self.y - self.c_y)**2)
        mask = dist_from_center < self.radius
        # mask_x = np.logical_and(
        #     self.x > self.c_x - self.radius, self.x < self.c_x + self.radius)
        # A = self.A.copy()
        # A = A[mask]
        # mask_y = np.logical_and(
        #     A[:, 1] > self.c_y - self.radius, A[:, 1] < self.c_y + self.radius)
        return self.A[mask]

    def hopf_cole_transform(self):
        Omega = self.domain()
        x, y = Omega[:, 0], Omega[:, 1]
        # x, y = np.meshgrid(x, y)
        u = 3/4 - 1/4 * \
            (1 / (1 + np.exp((4*y - 4*x - self.t) / (32*self.nu))))
        v = 3/4 + 1/4 * \
            (1 / (1 + np.exp((4*y - 4*x - self.t) / (32*self.nu))))

        return u, v
