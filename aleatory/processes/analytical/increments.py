import numpy as np

from aleatory.processes.base import StochasticProcess
from aleatory.utils.utils import check_positive_integer, times_to_increments, get_times


class IndependentIncrements(StochasticProcess):
    """
    Independent increments
    """

    def __int__(self, T=1.0, rng=None):
        super().__init__(T=T, rng=rng)
        self.name = "Random Increments"
        self.times = None
        self.n = None
        self.paths = None

    def __str__(self):
        return "Random Independent Increments on [0, {T}]".format(T=str(self.T))

    def __repr__(self):
        return "IndependentIncrements(T={T})".format(T=str(self.T))

    def _sample(self, n):
        """
        Generates a random sample of size n from N(0, T/n)
        :param n: number of increments
        :return:
        """
        check_positive_integer(n)
        self.n = n
        delta_t = 1.0 * self.T / self.n
        self.times = get_times(self.T, self.n)
        noise = self.rng.normal(scale=np.sqrt(delta_t), size=self.n)

        return noise

    def _sample_at(self, times):
        """
        Generates random Gaussian increments corresponding to the specified times.
        :param times: given times
        :return:
        """
        if times[0] != 0:
            times = np.concatenate(([0], times))
        increments = times_to_increments(times)
        self.times = times
        noise = np.array([self.rng.normal(scale=np.sqrt(inc)) for inc in increments])
        noise = np.concatenate(([0], noise))

        return noise

    def sample(self, n):
        """
        Generates a random sample of size n from N(0, T/n)
        :param n:
        :return:
        """
        return self._sample(n)

    def sample_at(self, times):
        """
        Generates random Gaussian increments corresponding to the specified times
        Note that the first increment always starts with zero.
        :param times:
        :return:
        """
        return self._sample_at(times)


class GammaIncrements(IndependentIncrements):

    def __init__(self, k, theta, T=1.0, rng=None):
        super().__init__(T=T, rng=rng)
        self.name = "Gamma Independent Increments"
        self.k = k  # shape
        self.theta = theta  # scale

    def _sample(self, n):
        """
        Generates a random sample of size n from N(0, T/n)
        :param n: number of increments
        :return:
        """
        check_positive_integer(n)
        self.n = n
        delta_t = 1.0 * self.T / self.n
        self.times = get_times(self.T, self.n)
        noise = self.rng.gamma(self.k * delta_t, self.theta, size=self.n)
        return noise

    def _sample_at(self, times):
        """
        Generates random Gaussian increments corresponding to the specified times.
        :param times: given times
        :return:
        """
        if times[0] != 0:
            times = np.concatenate(([0], times))
        increments = times_to_increments(times)
        self.times = times

        noise = np.array(
            [self.rng.gamma(shape=self.k * inc, scale=self.theta) for inc in increments]
        )
        noise = np.concatenate(([0], noise))
        return noise
