import numpy as np


class terms_uh(object):
    '''
    x:  array
    beta: Natural number
    l: norm order (1,2,np.inf)
    '''

    def __init__(self, Mi, Mb, beta, c, x, poly_b=np.array([1]),  rbf='TPS', l=2):
        self.Mi = Mi
        self.ni = Mi.shape[0]
        self.Mb = Mb
        self.nb = Mb.shape[0]
        self.d = Mi.shape[1]
        self.dm = poly_b.shape[0]
        self.beta = beta
        self.c = c
        self.x = x
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

    def lambda_m(self):
        return self.RBF(self.norm_x(self.matrix_lamb_gamm_thet(self.Mi))).reshape(-1, 1)

    def gamma_m(self):
        return self.RBF(self.norm_x(self.matrix_lamb_gamm_thet(self.Mb))).reshape(-1, 1)

    def theta_m(self):
        return np.ones((self.dm, 1))

    def a_m(self):
        return np.matmul(self.Sinv(), np.matmul(self.B(), np.vstack((self.gamma_m(), self.theta_m()))) - self.lambda_m())

    def b_m(self):
        return np.matmul(self.C(), np.vstack((self.gamma_m(), self.theta_m()))) - np.matmul(self.B().transpose(), np.matmul(self.Sinv(), self.lambda_m()))


class implementation(terms_uh):

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


x_i = np.array([2, 3, 2]).reshape(-1, 1)
y_i = np.array([1, 0, 3]).reshape(-1, 1)
Mi = np.hstack((x_i, y_i))
Mb = np.array([[0., 0.5], [1., 0.5]])
x = np.array([1.1, 2.3])

M1 = np.array([
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
print(implementation(Mi, Mb, 2, 0.1, x).K1())
# MQ1 = implementation(Mi, Mb, 2, 0.1).B()
# A = implementation(Mi, Mb, 2, 0.1).A()
# Ainv = np.linalg.inv(A)

# print(Mb)
# print(x)

# print(x-Mb)
# f =implementation(Mi, Mb, 2, 0.1, x)
# print(Mb)
# M = f.RBF(Mb)
#print(np.linalg.norm(M, 2, axis=-1))

# M2 = np.linalg.norm(Mb, 2, axis = -1)
# print(M2)
# M2 = f.RBF(M2)


#print(np.linalg.norm(x-Mb, axis=-1).reshape(-1, 1))
