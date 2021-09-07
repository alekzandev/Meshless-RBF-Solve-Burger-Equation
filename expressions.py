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

    def __init__(self, Mb, npnts, c=1, beta=1, nu=0.1, mu=1., epsilon=1, poly_b=np.array([1]),  rbf='TPS', l=2):
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
        self.exact_solution = "1"

    def drop_to_zero(self, M, tol=1e-5):
        return np.where(abs(M) < tol, 0, M)

    def RBF(self, M):
        '''
        RBF
        '''
        if self.rbf == 'TPS':
            self.beta = 1
            return self.drop_to_zero((-1)**(self.beta + 1) * M**(2*self.beta) * np.log(M+1e-80))
        elif self.rbf == 'MQ':
            self.beta = 3/2
            return self.drop_to_zero((-1)**(int(self.beta)+1) * (self.c**2 + self.epsilon**2 * M ** 2)**self.beta)
        elif self.rbf == 'radial_powers':
            return self.drop_to_zero((-1)**int(self.beta/2) * self.norm_x(M)**self.beta)
        elif self.rbf == 'rbf_ut':
            return self.drop_to_zero((-1/self.c**3) * (np.exp(-self.c * self.norm_x(M)) + self.c * self.norm_x(M)))

    def O1(self):
        return np.zeros((self.dm, self.d))

    def O2(self):
        return np.zeros((self.dm, self.dm)) + 0.001*np.eye(self.dm)

    def K1(self):
        if self.rbf == 'TPS':
            self.lambda_K = 1
        elif self.rbf == 'MQ':
            self.lambda_K = 0
        return self.RBF(self.norm_x(self.matrix_K(self.Mi))) + self.lambda_K*np.eye(self.ni)

    def K2(self):
        if self.rbf == 'TPS':
            self.lambda_K = 1
        elif self.rbf == 'MQ':
            self.lambda_K = 0
        return self.RBF(self.norm_x(self.matrix_K(self.Mb))) + self.lambda_K*np.eye(self.nb)

    def M(self):
        return self.RBF(self.norm_x(self.matrix_M()))

    def q(self, x, y, i):
        # Hermite grade m-1
        if i == 1:
            return np.ones(x.shape)
        elif i == 2:
            return 2*x
        elif i == 3:
            return 2*y

    def lap_q(self, x, i):
        #Hermite and lagrange
        if i == 1:
            return 0
        elif i == 2:
            return 0
        elif i == 3:
            return 0

    def grad_q(self, x, y, i):
        # #Hermite m-1
        if i == 1:
            return np.hstack((0, 0)).reshape(1, -1)
        elif i == 2:
            return np.hstack((2, 0)).reshape(1, -1)
        elif i == 3:
            return np.hstack((0, 2)).reshape(1, -1)

    def poly_basis(self, M, i):
        return self.q(M[:, 0].reshape(-1, 1), M[:, 1].reshape(-1, 1), i)

    def Q1(self):
        col1 = self.poly_basis(self.Mi, 1)
        col2 = self.poly_basis(self.Mi, 2)
        col3 = self.poly_basis(self.Mi, 3)
        return np.hstack((col1, col2, col3))

    def Q2(self):
        col1 = self.poly_basis(self.Mb, 1)
        col2 = self.poly_basis(self.Mb, 2)
        col3 = self.poly_basis(self.Mb, 3)
        return np.hstack((col1, col2, col3))

    def G(self, t):
        if self.exact_solution == "1":
            return self.Mb/((t+self.alpha)+(t+self.alpha)**2 * np.exp(self.norm_x(self.Mb).reshape(-1, 1) ** 2/(4*(self.alpha+t))))
        elif self.exact_solution == "2":
            u = 3/4 - 1 / \
                (4*(1+np.exp((4*self.Mb[:, 1] - 4 *
                              self.Mb[:, 0] - t)/(self.nu*32))))
            v = 3/4 + 1 / \
                (4*(1+np.exp((4*self.Mb[:, 1] - 4 *
                              self.Mb[:, 0] - t)/(self.nu*32))))
            return np.hstack((u.reshape(-1, 1), v.reshape(-1, 1)))

    def G_tilde(self, t):
        return np.vstack((self.G(t), self.O1()))

    def A(self):
        A1 = np.hstack((self.K2(), self.Q2()))
        A2 = np.hstack((self.Q2().T, self.O2()))
        return np.vstack((A1, A2))

    def ACaps(self):
        r1 = np.hstack((self.K1(), self.M(), self.Q1()))
        r2 = np.hstack((self.M().T, self.K2(), self.Q2()))
        r3 = np.hstack((self.Q1().T, self.Q2().T, self.O2()))
        return np.vstack((r1, r2, r3))

    def Sinv(self):
        MQ1 = np.hstack((self.M(), self.Q1()))
        MQ1T = np.vstack((self.M().T, self.Q1().T))
        Ainv = np.linalg.inv(self.A())
        S = self.K1() - np.matmul(MQ1, np.matmul(Ainv, MQ1T))
        return np.linalg.inv(S)

    def B(self):
        MQ1 = np.hstack((self.M(), self.Q1()))
        Ainv = np.linalg.inv(self.A())
        return np.matmul(MQ1, Ainv)

    def C(self):
        return np.linalg.inv(self.A()) + np.matmul(self.B().T, np.matmul(self.Sinv(), self.B()))

    def lambda_m(self, phi_m):
        self.op = "lambda"
        return phi_m(self.norm_x(self.matrix_lamb_gamm_thet(self.Mi)).reshape(-1, 1))

    def gamma_m(self, phi_m):
        self.op = "gamma"
        return phi_m(self.norm_x(self.matrix_lamb_gamm_thet(self.Mb)).reshape(-1, 1))

    def theta_m(self, phi_m):
        if phi_m == self.RBF:
            row1 = self.q(self.x[0], self.x[1], 1)
            row2 = self.q(self.x[0], self.x[1], 2)
            row3 = self.q(self.x[0], self.x[1], 3)
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

    def a_m_op(self, lin_op):
        gam_thet = np.vstack((self.gamma_m(lin_op), self.theta_m(lin_op)))
        term_par = np.matmul(self.B(), gam_thet) - self.lambda_m(lin_op)
        return - np.matmul(self.Sinv(), term_par)

    def b_m_op(self, lin_op):
        gam_thet = np.vstack((self.gamma_m(lin_op), self.theta_m(lin_op)))
        fterm = np.matmul(self.C(), gam_thet)
        sterm = np.matmul(self.B().T, np.matmul(
            self.Sinv(), self.lambda_m(lin_op)))
        return fterm - sterm

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
                xy = self.matrix_lamb_gamm_thet(self.Mi)
                comp_x = (-1)**(self.beta + 1) * (M**(2*self.beta-2)) *\
                    xy[:, 0].reshape(-1, 1) *\
                    (2*self.beta*np.log(M+1e-20)+1)
                comp_y = (-1)**(self.beta + 1) * (M**(2*self.beta-2)) *\
                    xy[:, 1].reshape(-1, 1) *\
                    (2*self.beta*np.log(M+1e-20)+1)
            elif op == "gamma":
                xy = self.matrix_lamb_gamm_thet(self.Mb)
                comp_x = (-1)**(self.beta + 1) * (M**(2*self.beta-2)) *\
                    xy[:, 0].reshape(-1, 1) *\
                    (2*self.beta*np.log(M+1e-20)+1)
                comp_y = (-1)**(self.beta + 1) * (M**(2*self.beta-2)) *\
                    xy[:, 1].reshape(-1, 1) *\
                    (2*self.beta*np.log(M+1e-20)+1)
            return self.drop_to_zero(np.hstack((comp_x, comp_y)))

        elif self.rbf == 'MQ':
            if op == 'lambda':
                xy = self.matrix_lamb_gamm_thet(self.Mi)
                comp_x = 3 * self.epsilon**2 *\
                    xy[:, 0].reshape(-1, 1) * np.sqrt(self.epsilon **
                                                      2 * M**2 + 1)  # .reshape(-1, 1)
                comp_y = 3 * self.epsilon**2 *\
                    xy[:, 1].reshape(-1, 1) * np.sqrt(self.epsilon **
                                                      2 * M**2 + 1)  # .reshape(-1, 1)
            elif op == 'gamma':
                xy = self.matrix_lamb_gamm_thet(self.Mb)
                comp_x = 3 * self.epsilon**2 *\
                    xy[:, 0].reshape(-1, 1) * np.sqrt(self.epsilon **
                                                      2 * M**2 + 1)  # .reshape(-1, 1)
                comp_y = 3 * self.epsilon**2 *\
                    xy[:, 1].reshape(-1, 1) * np.sqrt(self.epsilon **
                                                      2 * M**2 + 1)  # .reshape(-1, 1)
            return self.drop_to_zero(np.hstack((comp_x, comp_y)))

    def laplacian_TPS(self, M):
        '''
        RBF laplacian
        '''
        if self.rbf == 'TPS':
            return self.drop_to_zero((-1)**(self.beta + 1) * M**(2*self.beta-2) * (4*self.beta*(self.beta*np.log(M+1e-20)+1)))
        elif self.rbf == 'MQ':
            return self.drop_to_zero(3 * self.epsilon**2 * ((3 * self.epsilon**2 * M**2 + 2)/np.sqrt(self.epsilon**2 * M**2 + 1)))

    def X_0(self):
        if self.exact_solution == "1":
            alpha = self.alpha
            n = self.norm_x(self.Mi)
            c1 = self.Mi[:, 0]/(alpha + (alpha ** 2) *
                                np.exp((n ** 2)/(4*alpha)))
            c2 = self.Mi[:, 1]/(alpha + (alpha ** 2) *
                                np.exp((n ** 2)/(4*alpha)))
            return np.hstack((c1.reshape(-1, 1), c2.reshape(-1, 1)))
        elif self.exact_solution == "2":
            u = 3/4 - 1 / \
                (4 *
                 (1+np.exp((4*self.Mi[:, 1] - 4*self.Mi[:, 0])/(self.nu*32))))
            v = 3/4 + 1 / \
                (4 *
                 (1+np.exp((4*self.Mi[:, 1] - 4*self.Mi[:, 0])/(self.nu*32))))
            return np.hstack((u.reshape(-1, 1), v.reshape(-1, 1)))

    def F_m(self, X0, t, i=0):
        dissipation_term = self.nu * \
            (np.matmul(self.lap_am().T, X0) +
             np.matmul(self.lap_bm().T, self.G_tilde(t)))
        ei = np.zeros((self.ni, 1))
        ei[int(i)] = 1
        advection_term1 = np.matmul(ei.T, X0)
        advection_term2 = np.matmul(
            self.grad_am().T, X0) + np.matmul(self.grad_bm().T, self.G_tilde(t))
        advection_term = self.mu * np.matmul(advection_term1, advection_term2)
        return dissipation_term - advection_term


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


class solve_matrix(assembled_matrix):
    def variables(self, Xk, dt, Y):
        self.Xk = Xk
        self.dt = dt
        self.Y = Y
        self.p = (3 - np.sqrt(3))/6
        self.A_tab = np.array([[self.p, 0], [1 - 2*self.p, self.p]])
        self.b_tab = np.array([1/2, 1/2])
        self.c_tab = np.array([self.p, 1-self.p])
        self.e = np.ones((2, 1))
        self.s = len(self.c_tab)

    def build_matrices(self, tk):
        Y1 = self.Y[:self.ni, :]
        Y2 = self.Y[self.ni:, :]
        for i, x in enumerate(self.Mi):
            self.x = x
            F1 = self.F_m(Y1, tk + self.c_tab[0]*self.dt, i)
            F2 = self.F_m(Y2, tk + self.c_tab[1]*self.dt, i)
            ei = np.zeros((self.ni, 1))
            ei[i] = 1
            wi = self.grad_am().dot(self.Xk.T).dot(ei) - self.nu * self.lap_am()
            Vi_X = self.Xk.T.dot(self.grad_am()) + \
                self.G_tilde(tk).T.dot(self.grad_bm())
            B1k = ei.T.dot(Y1).dot(Vi_X.T)
            B2k = ei.T.dot(Y2).dot(Vi_X.T)
            yield np.vstack((F1, F2)), wi, np.vstack((B1k, B2k))

    def calculate(self, tk):
        objects = np.array(tuple(self.build_matrices(tk)), dtype=object)
        F = F = np.vstack(objects[:, 0])
        W = np.hstack(objects[:, 1])
        Bsk = np.vstack(objects[:, 2])
        Bsk = np.vstack((Bsk[::2, :], Bsk[1::2, :]))
        return F, W, Bsk

    def inexact_Newthon(self, tk):
        F, W, Bsk = self.calculate(tk)
        Rp = np.kron(self.e, self.Xk) + self.dt * \
            np.kron(self.A_tab, np.eye(self.ni)).dot(F) - self.Y
        b = -self.dt*np.kron(self.A_tab, np.eye(self.ni)).dot(Bsk) - Rp
        A = self.dt*np.kron(self.A_tab, W.T) + np.eye(int(self.s*self.ni))
        return A, b

    def Xk1(self, Y, tk):
        Y1 = Y[:self.ni, :]
        Y2 = Y[self.ni:, :]
        for i, x in enumerate(self.Mi):
            self.x = x
            F1 = self.F_m(Y1, tk, i)
            F2 = self.F_m(Y2, tk, i)
            Fk = np.vstack((F1, F2))

        return np.kron(self.e, self.Xk) + self.dt*np.kron(self.A_tab, np.eye(self.ni)).dot(Fk) - self.Y

    # def step(self, Xk):
    #     Xk1 = Xk + self.dt*(np.kron(self.b.T, np.eye(self.ni))).dot(self)


# class inexact_Newthon(assembled_matrix):
#     def vi(self, X, tk):
#         return X.T.dot(self.grad_am()) + self.G_tilde(tk).T.dot(self.grad_am())

#     def wi(self, X, i):
#         ei = np.zeros((self.ni, 1))
#         ei[int(i)] = 1
#         return self.grad_am().dot(X.T).dot(ei) - self.nu*self.lap_am()


class create_domain(object):
    def __init__(self, A=None, t=None, nu=None, radius=0.15, c_x=0.5, c_y=0.5):
        self.A = A
        self.x = A[:, 0]
        self.y = A[:, 1]
        self.t = t
        self.nu = nu
        self.radius = radius
        self.c_x = c_x
        self.c_y = c_y

    def setup(self, domain='unit_square', bound_points=20):
        if domain == 'circle_centre':
            dist_from_center = np.sqrt(
                (self.x - self.c_x)**2 + (self.y - self.c_y)**2)
            #n_round = str(self.radius)
            #n_round = len(n_round.split('.')[-1])
            #maskf = dist_from_center.round(n_round) == self.radius
            x = HaltonPoints(1, bound_points).haltonPoints().reshape(1, -1)[0]
            x = x[(x <= self.c_x + self.radius) &
                  (x >= self.c_x - self.radius)].copy()
            dx = abs(x - self.c_x)
            dy1 = np.sqrt(self.radius**2 - dx**2)
            dy2 = -np.sqrt(self.radius**2 - dx**2)
            y1 = self.c_y + dy1
            y2 = self.c_y + dy2
            f1 = np.hstack((x.reshape(-1, 1), y1.reshape(-1, 1)))
            f2 = np.hstack((x.reshape(-1, 1), y2.reshape(-1, 1)))
            maskd = dist_from_center > self.radius + 1e-2
            # mask_x = np.logical_and(
            #     self.x > self.c_x - self.radius, self.x < self.c_x + self.radius)
            # A = self.A.copy()
            # A = A[mask]
            # mask_y = np.logical_and(
            #     A[:, 1] > self.c_y - self.radius, A[:, 1] < self.c_y + self.radius)
            return self.A[maskd], np.vstack((f1, f2))
        elif domain == 'unit_square':
            return self.A, np.empty((0, 2))

    def hopf_cole_transform(self):
        Omega = self.domain()
        x, y = Omega[:, 0], Omega[:, 1]
        # x, y = np.meshgrid(x, y)
        u = 3/4 - 1/4 * \
            (1 / (1 + np.exp((4*y - 4*x - self.t) / (32*self.nu))))
        v = 3/4 + 1/4 * \
            (1 / (1 + np.exp((4*y - 4*x - self.t) / (32*self.nu))))

        return u, v
