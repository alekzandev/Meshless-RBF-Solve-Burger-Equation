import numpy as np


class inexact_Newthon(object):
    def __init__(self, Y, uh):
        self.Y = Y
        self.uh = uh


class terms_matrix(inexact_Newthon):
    def vi_wi(self, X0, tk):
        vi = []
        wi = []
        for i,xi in enumerate(self.uh.Mi):
            self.uh.x = xi
            ei=np.zeros((self.uh.ni, 1))
            ei[int(i)] = 1
            vi.append(X0.T.dot(self.uh.grad_am()) +
                      self.uh.G_tilde(tk).T.dot(self.uh.grad_bm()))
            wi.append(self.uh.grad_am().dot(X0.T).dot(ei) -
                      self.uh.nu * self.uh.lap_am())
        return vi, wi

    def Bk(self, H, X0, tk):
        b = []
        for i in range(self.uh.ni):
            ei = np.zeros((self.uh.ni, 1))
            ei[int(i)] = 1
            b.append(ei.T.dot(H).dot(self.vi_wi(X0, tk)))
        return b

    
    


