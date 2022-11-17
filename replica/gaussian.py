import numpy as np

from base import StochasticProcess
from utils import check_positive_integer, times_to_increments


class Gaussian(StochasticProcess):
    def __int__(self, T=1.0, rng=None):
        super().__init__(T=T, rng=rng)

    def __str__(self):
        return "Gaussian noise generator on interval [0, {T}]".format(T=str(self.T))

    def __repr__(self):
        return "GaussianNoise(T={T})".format(T=str(self.T))

    def _sample_gaussian_noise(self, n):
        """
        Generates a random sample of size n from N(0, T/n)
        :param n:
        :return:
        """
        check_positive_integer(n)
        delta_t = 1.0 * self.T / n
        noise = self.rng.normal(scale=np.sqrt(delta_t), size=n)

        return noise

    def _sample_gaussian_noise_at(self, times):
        """
        Generate Gaussian increments at specified times starting  from zero
        :param times:
        :return:
        """
        if times[0] != 0:
            times = np.concatenate(([0], times))
        increments = times_to_increments(times)
        noise = np.array([self.rng.normal(scale=np.sqrt(inc)) for inc in increments])

        return noise

    def sample(self, n):
        """
        Generate a Gaussian noise realization with n increments.
        :param n:
        :return:
        """
        return self._sample_gaussian_noise(n)

    def sample_at(self, times):
        """
        GGenerate Gaussian increments at specified times starting  from zero
        :param times:
        :return:
        """
        return self._sample_gaussian_noise_at(times)
