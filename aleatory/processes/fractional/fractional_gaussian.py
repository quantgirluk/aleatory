import numpy as np

from aleatory.processes.base import StochasticProcess
from aleatory.utils.utils import check_positive_integer, times_to_increments, get_times


class FractionalGaussianIncrements(StochasticProcess):
    """
    Fractional Gaussian increments
    """
    def __init__(self, hurst=0.5, T=1.0, rng=None):
        super().__init__(T=T, rng=rng)
        self.hurst = hurst
        self.name = "Gaussian Noise"
        self.times = None
        self.n = None
        self.paths = None
        self.expectation = None
        self.covariance_matrix = None

    def _cov(self, t,s,):
        twice_hurst = 2*self.hurst
        cov = 0.5*(s**twice_hurst + t**twice_hurst - abs(t-s)**twice_hurst)
        return cov

    def _cov_for_times(self):

        times = self.times
        n = len(times)
        cov_matrix = np.zeros((n-1, n-1))

        for i in range(n):
            for j in range(1,i+1):
                cov = self._cov(times[i], times[j])
                cov_matrix[i-1, j-1] = cov
                cov_matrix[j-1, i-1] = cov
        return cov_matrix

    def _sample_fractional_gaussian_increments(self, n, paths):
        """
        Generates a random sample of size n from N(0, T/n)
        :param n: number of increments
        :return:
        """
        check_positive_integer(n)
        self.n = n
        self.paths = paths
        delta_t = 1.0 * self.T / self.n
        self.times = get_times(self.T, self.n)
        self.expectation = np.zeros(n-1)
        self.covariance_matrix = self._cov_for_times()

        noise = self.rng.multivariate_normal(self.expectation, self.covariance_matrix, size=self.paths)
        # noise = self.rng.normal(scale=np.sqrt(delta_t), size=self.n, )

        return noise


fgi = FractionalGaussianIncrements(hurst=0.5,T=1.0)
sample = fgi._sample_fractional_gaussian_increments(n=6, paths=1)
print(fgi.covariance_matrix)
print(fgi.times)
print(sample)
