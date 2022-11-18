from base_paths import StochasticProcessPaths
from cir_process import CIRProcess
import numpy as np
from scipy.stats import norm, ncx2


class CIRProcessPaths(StochasticProcessPaths):

    def __init__(self, N, theta=1.0, mu=1.0, sigma=1.0, initial=0.0, n=10, T=1.0, rng=None):
        super().__init__(rng=rng)
        self.N = N
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.initial = initial
        self.n = n
        self.T = T
        self._dt = 1.0 * self.T / self.n
        self.times = np.arange(0.0, self.T + self._dt, self._dt)
        process = CIRProcess(theta=self.theta, mu=self.mu, sigma=self.sigma, initial=self.initial, n=self.n, T=self.T)
        self.paths = [process.sample(n) for k in range(int(N))]
        self.name = "CIR Process"

    def _process_expectation(self):
        expectation = self.initial * np.exp((-1.0) * self.theta * self.times) + self.mu * (
                np.ones(len(self.times)) - np.exp((-1.0) * self.theta * self.times))
        return expectation

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
        # mu_x = self.initial * np.exp((-1.0) * self.theta * t) + self.mu * (1.0 - np.exp(-1.0 * self.theta * t))
        # variance_x = (self.initial * self.sigma ** 2 / self.theta) * (
        #             np.exp(-1.0 * self.theta * t) - np.exp(-2.0 * self.theta * t)) + (
        #                         self.mu * self.sigma ** 2 / 2 * self.theta) * (
        #                         (1.0 - np.exp(-1.0 * self.theta * t)) ** 2)
        # sigma_x = np.sqrt(variance_x)
        nu = 4.0 * self.theta * self.mu / self.sigma**2
        ct = 4.0 * self.theta/((self.sigma**2)*(1.0 - np.exp(-1.0*self.theta*t)))
        lambdat = ct * self.initial * np.exp(-1.0*self.theta*t)
        scale = 1.0/(4.0 * self.theta/((self.sigma**2)*(1.0 - np.exp(-1.0*self.theta*t))))
        marginal = ncx2(nu, lambdat, scale=scale)
        return marginal

    def plot(self):
        self._plot_paths()
        return 1

    def draw(self):
        self._draw_paths()
        return 1
