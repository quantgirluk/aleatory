from replica.processes.base import SPEulerMaruyama
import numpy as np
from scipy.stats import norm


class OUProcess(SPEulerMaruyama):

    def __init__(self, theta=1.0, mu=1.0, sigma=1.0, initial=0.0, T=1.0, rng=None):
        super().__init__(T=T, rng=rng)
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.initial = initial
        self.n = 1.0
        self.dt = 1.0 * self.T / self.n
        self.times = None
        self.name = "Ornstein–Uhlenbeck process"

        def f(x, _):
            return self.theta * (self.mu - x)

        def g(x, _):
            return self.sigma

        self.f = f
        self.g = g

    def __str__(self):
        return "Orstein-Uhlenbeck process with parameters {speed}, {mean}, and {volatility} on [0, {T}].".format(
            T=str(self.T), speed=str(self.theta), mean=str(self.mu), volatility=str(self.sigma))

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
