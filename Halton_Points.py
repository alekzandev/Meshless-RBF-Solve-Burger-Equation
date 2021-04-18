import numpy as np


class HaltonPoints(object):
    def __init__(self, dim, n_sample, big_number=10, base=2):
        self.dim = dim
        self.n_sample = n_sample
        self.big_number = big_number
        self.base = base

    def primes_from_2_to(self, n):
        """Prime number from 2 to n.
        From `StackOverflow <https://stackoverflow.com/questions/2068372>`_.
        :param int n: sup bound with ``n >= 6``.
        :return: primes in 2 <= p < n.
        :rtype: list
        """
        sieve = np.ones(n // 3 + (n % 6 == 2), dtype=np.bool)
        for i in range(1, int(n ** 0.5) // 3 + 1):
            if sieve[i]:
                k = 3 * i + 1 | 1
                sieve[k * k // 3::2 * k] = False
                sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False
        return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]

    def van_der_corput(self, n_sample, base):
        """Van der Corput sequence.
        :param int n_sample: number of element of the sequence.
        :param int base: base of the sequence.
        :return: sequence of Van der Corput.
        :rtype: list (n_samples,)
        """
        sequence = []
        for i in range(n_sample):
            n_th_number, denom = 0., 1.
            while i > 0:
                i, remainder = divmod(i, base)
                denom *= base
                n_th_number += remainder / denom
            sequence.append(n_th_number)

        return sequence

    def haltonPoints(self):
        """Halton sequence.
        :param int dim: dimension
        :param int n_sample: number of samples.
        :return: sequence of Halton.
        :rtype: array_like (n_samples, n_features)
        """

        while 'Not enought primes':
            base = self.primes_from_2_to(self.big_number)[:self.dim]
            if len(base) == self.dim:
                break
            self.big_number += 1000
            print(self.big_number)

        # Generate a sample using a Van der Corput sequence per dimension.
        sample = [self.van_der_corput(self.n_sample + 1, dim) for dim in base]
        sample = np.stack(sample, axis=-1)[1:]

        return sample
