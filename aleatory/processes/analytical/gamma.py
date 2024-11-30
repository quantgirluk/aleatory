"""
Gamma Process
"""
import numpy as np
from scipy.stats import gamma
from aleatory.utils.utils import check_positive_number, get_times

from aleatory.processes.base_analytical import SPAnalytical
from aleatory.processes.analytical.increments import GammaIncrements


class GammaProcess(SPAnalytical):

    def __init__(self, mu=1.0, nu=1.0, T=1.0, rng=None):
        super().__init__(T=T, rng=rng, initial=0.0)
        self.mu = mu
        self.nu = nu
        self.name = "Gamma Process"
        self.n = None
        self.times = None
        self.shape = self.mu ** 2 / self.nu
        self.scale = self.nu / self.mu
        self.rate = 1.0 / self.scale
        self.gamma_increments = GammaIncrements(k=self.shape, theta=self.scale, T=self.T, rng=self.rng)

    def __str__(self):
        return f'Gamma Process with parameters mu = {self.mu} and nu = {self.nu}'

    def __repr__(self):
        return f'GammaProcess(mu={self.mu}, nu={self.nu})'

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        check_positive_number(value, "mu")
        self._mu = value

    @property
    def nu(self):
        return self._nu

    @nu.setter
    def nu(self, value):
        check_positive_number(value, "nu")
        self._nu = value

    def _sample_gamma_process(self, n):
        self.n = n
        self.times = get_times(self.T, self.n)
        increments = np.cumsum(self.gamma_increments.sample(n - 1))
        increments = np.insert(increments, 0, [0])

        path = np.full(n, self.initial) + increments
        return path

    def _sample_gamma_process_at(self, times):
        if times[0] != 0:
            times = np.insert(times, 0, [0])
        self.times = times

        increments = np.cumsum(self.gamma_increments.sample_at(times))
        path = np.full(len(self.times), self.initial) + increments

        return path

    def sample(self, n):
        """
        Generates a discrete time sample from a Gamma process instance.

        :param int n: the number of steps
        :return: numpy array
        """
        return self._sample_gamma_process(n)

    def sample_at(self, times):
        """
        Generates a sample from a Gamma process at the specified times.

        :param times: the times which define the sample
        :return: numpy array
        """
        return self._sample_gamma_process_at(times)

    def get_marginal(self, t):
        marginal = gamma(a=self.shape * t, scale=self.scale)
        return marginal

    def _process_expectation(self, times=None):
        if times is None:
            times = self.times
        return self.mu * times

    def _process_variance(self, times=None):
        if times is None:
            times = self.times
        return np.full(len(times), self.nu)

    def marginal_expectation(self, times=None):
        expectations = self._process_expectation(times=times)
        return expectations

    def marginal_variance(self, times):
        variances = self._process_variance(times=times)
        return variances


# p = GammaProcess(mu=1.0, nu=2.0, T=10)
# # p.sample(n=10)
# p.plot(n=100, N=200)
# p.draw(n=10, N=200)
# p.draw(n=100, N=200)
# p.draw(n=100, N=200, envelope=True)
