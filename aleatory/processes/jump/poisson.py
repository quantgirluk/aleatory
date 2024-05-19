"""Poisson Process"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

from aleatory.processes.base import BaseProcess
from aleatory.utils.utils import check_positive_number, check_positive_integer


class PoissonProcess(BaseProcess):

    def __init__(self, rate=1.0, rng=None):
        super().__init__(rng=rng)
        self.name = 'Poisson Process'
        self.rate = rate
        self.T = None
        self.N = None
        self.paths = None

    def __str__(self):
        return "Poisson Process with intensity rate r={rate}.".format(rate=str(self.rate))

    def __repr__(self):
        return "PoissonProcess(rate={r})".format(r=str(self.rate))

    @property
    def rate(self):
        return self._rate

    @rate.setter
    def rate(self, value):
        if value < 0:
            raise ValueError("rate must be positive")
        self._rate = value

    def _sample_poisson_process(self, jumps=None, T=None):
        if jumps is not None and T is not None:
            raise ValueError("Only one must be provided either jumps or T")
        elif jumps:
            check_positive_integer(jumps)
            exp_mean = 1.0 / self.rate
            exponential_times = self.rng.exponential(exp_mean, size=jumps)
            arrival_times = np.cumsum(exponential_times)
            arrival_times = np.insert(arrival_times, 0, [0])
            return arrival_times
        elif T:
            check_positive_number(T, "Time")
            num_jumps = np.random.poisson(self.rate * T)
            return self._sample_poisson_process(jumps=num_jumps)

    def sample(self, jumps=None, T=None):
        return self._sample_poisson_process(jumps=jumps, T=T)

    def get_marginal(self, t):
        # return expon(scale = self.rate*t)
        return poisson(self.rate * t)

    def marginal_expectation(self, times):
        return self.rate * times

    def simulate(self, N, jumps=None, T=None):
        """
        Simulate paths/trajectories from the instanced stochastic process.

        :param N: number of paths to simulate
        :param jumps: number of jumps
        :param T: time T
        :return: list with N paths (each one is a numpy array of size n)
        """
        self.N = N
        self.paths = [self.sample(jumps=jumps, T=T) for _ in range(N)]
        return self.paths

    def plot(self, N, jumps=None, T=None):
        self.simulate(N, jumps=jumps, T=T)
        paths = self.paths
        for p in paths:
            counts = np.arange(0, len(p))
            plt.step(p, counts)
        plt.show()

    def draw(self, N, jumps=None, T=None):
        self.simulate(N, jumps=jumps, T=T)
        paths = self.paths
        for p in paths:
            counts = np.arange(0, len(p))
            plt.step(p, counts)

        max_time = np.max(np.hstack(paths))
        times = np.linspace(0.1, max_time, 100)
        marginals = [self.get_marginal(t) for t in times]
        expectations = self.marginal_expectation(times)
        qq05 = [m.ppf(0.005) for m in marginals]
        qq95 = [m.ppf(0.995) for m in marginals]
        plt.plot(times, expectations)
        plt.fill_between(times, qq05, qq95, alpha=0.25, color='grey')
        plt.title(self.name)
        plt.show()

# p = PoissonProcess()
# # p.plot(N=10, jumps=10)
# p.draw(N=10, jumps=50)
#
# p = PoissonProcess(rate=0.5)
# # p.plot(N=10, T=20)
# p.draw(N=10, T=20)
