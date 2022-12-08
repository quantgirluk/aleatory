from replica.sp_paths.base_paths import ExactStochasticProcessPaths
from replica.processes.euler_maruyama.ou_process import OUProcess
import numpy as np
from scipy.stats import norm


class OUProcessPaths(ExactStochasticProcessPaths):

    def __init__(self, N, theta=1.0, mu=1.0, sigma=1.0, initial=0.0, n=10, T=1.0, rng=None):
        super().__init__(T=T, N=N, rng=rng)
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.initial = initial
        self.n = n
        self._dt = 1.0 * self.T / self.n
        self.process = OUProcess(theta=self.theta, mu=self.mu, sigma=self.sigma, initial=self.initial, n=self.n,
                                 T=self.T)
        self.paths = [self.process.sample(n) for _ in range(int(N))]
        self.times = self.process.times
        self.name = "Ornstein–Uhlenbeck Process"

    def _process_expectation(self):
        return self.initial * np.exp((-1.0) * self.theta * self.times) + self.mu * (
                np.ones(len(self.times)) - np.exp((-1.0) * self.theta * self.times))

    def process_expectation(self):
        expectations = self._process_expectation()
        return expectations

    def _process_variance(self):
        variances = (self.sigma ** 2) * (1.0 / (2.0 * self.theta)) * (
                np.ones(len(self.times)) - np.exp(-2.0 * self.theta * self.times))
        return variances

    def process_variance(self):
        variances = self._process_variance()
        return variances

    def _process_stds(self):
        stds = np.sqrt(self.process_variance())
        return stds

    def process_stds(self):
        stds = self._process_stds()
        return stds

    def get_marginal(self, t):
        mu_x = self.initial * np.exp(-1.0 * self.theta * t) + self.mu * (1.0 - np.exp(-1.0 * self.theta * t))
        variance_x = (self.sigma ** 2) * (1.0 / (2.0 * self.theta)) * (1.0 - np.exp(-2.0 * self.theta * t))
        sigma_x = np.sqrt(variance_x)
        marginal = norm(loc=mu_x, scale=sigma_x)

        return marginal

    def plot(self):
        self._plot_paths()
        return 1

    def draw(self):
        self._draw_paths(style='3sigma')
        return 1

    def draw_envelope(self):
        self._draw_envelope_paths(style='qq')
        return 1
