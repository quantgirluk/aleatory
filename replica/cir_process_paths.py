import numpy as np
from scipy.stats import ncx2

from base_paths import StochasticProcessPaths
from cir_process import CIRProcess


class CIRProcessPaths(StochasticProcessPaths):

    def __init__(self, N, theta=1.0, mu=1.0, sigma=1.0, initial=0.0, n=10, T=1.0, rng=None):
        super().__init__(N=N, T=T, rng=rng)
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.initial = initial
        self.n = n
        self.process = CIRProcess(theta=self.theta, mu=self.mu, sigma=self.sigma, initial=self.initial, n=self.n,
                                  T=self.T)
        self.paths = [self.process.sample(n) for _ in range(int(N))]
        self.times = self.process.times
        self.name = "CIR Process"

    def _process_expectation(self):
        expectations = self.initial * np.exp((-1.0) * self.theta * self.times) + self.mu * (
                np.ones(len(self.times)) - np.exp((-1.0) * self.theta * self.times))
        return expectations

    def process_expectation(self):
        expectations = self._process_expectation()
        return expectations

    def _process_variance(self):
        variances = (self.initial * self.sigma ** 2 / self.theta) * (
                np.exp(-1.0 * self.theta * self.times) - np.exp(-2.0 * self.theta * self.times)) + (
                            self.mu * self.sigma ** 2 / 2 * self.theta) * (
                            (np.ones(len(self.times)) - np.exp(-1.0 * self.theta * self.times)) ** 2)
        return variances

    def process_variance(self):
        variances = self._process_variance()
        return variances

    def get_marginal(self, t):
        nu = 4.0 * self.theta * self.mu / self.sigma ** 2
        ct = 4.0 * self.theta / ((self.sigma ** 2) * (1.0 - np.exp(-1.0 * self.theta * t)))
        lambda_t = ct * self.initial * np.exp(-1.0 * self.theta * t)
        scale = 1.0 / (4.0 * self.theta / ((self.sigma ** 2) * (1.0 - np.exp(-1.0 * self.theta * t))))
        marginal = ncx2(nu, lambda_t, scale=scale)
        return marginal

    def plot(self):
        self._plot_paths()
        return 1

    def draw(self):
        self._draw_paths()
        return 1
