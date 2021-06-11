import numpy as np

class mahalanobis_distance(object):
    def __init__(self, X, q=10):
        self.X = X
        self.means_vector = np.mean(X, axis=0)
        self.S = np.cov(X, rowvar=False)
        self.q = q

    def mahalanobis(self):
        S_1 = np.linalg.inv(self.S)
        data_centered = self.X - self.menas_vector.reshape(1, -1)
        for obs in data_centered:
            d = np.sqrt(obs.reshape(1, -1).dot(S_1).dot(obs.reshape(-1, 1)))
            yield d[0, 0]

    def distance(self):
        return np.array(list(self.mahalanobis()))

    def cut_value(self):
        return np.quantile(self.distance(), 1-self.q/100)

    def dataset_cutted(self):
        return self.X[self.distance() < self.cut_value(), :]