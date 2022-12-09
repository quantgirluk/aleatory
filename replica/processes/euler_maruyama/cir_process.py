from replica.processes.base import SPEulerMaruyama
import numpy as np
from scipy.stats import ncx2


class CIRProcess(SPEulerMaruyama):
    def __init__(self, theta=1.0, mu=1.0, sigma=1.0, initial=0.0, T=1.0, rng=None):
        super().__init__(T=T, rng=rng)
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.initial = initial
        self.n = 1.0
        self.dt = 1.0 * self.T / self.n
        self.times = np.arange(0.0, self.T + self.dt, self.dt)
        self.name = "CIR Process"

        def f(x, _):
            return self.theta * (self.mu - x)

        def g(x, _):
            return self.sigma * np.sqrt(x)

        self.f = f
        self.g = g

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        if value < 0:
            raise ValueError("theta must be positive")
        self._theta = value

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        if value <= 0:
            raise ValueError("sigma has to be positive")
        if 2 * self.theta * self.mu <= value ** 2:
            raise ValueError("Condition 2*theta*mu >= sigma**2 must be satisfied")
        self._sigma = value

    def __str__(self):
        return "Cox–Ingersoll–Ross process with parameters {speed}, {mean}, and {volatility} on [0, {T}].".format(
            T=str(self.T), speed=str(self.theta), mean=str(self.mu), volatility=str(self.sigma))

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

    def _process_stds(self):
        stds = np.sqrt(self.process_variance())
        return stds

    def process_stds(self):
        stds = self._process_stds()
        return stds

    def get_marginal(self, t):
        nu = 4.0 * self.theta * self.mu / self.sigma ** 2
        ct = 4.0 * self.theta / ((self.sigma ** 2) * (1.0 - np.exp(-4.0 * self.theta)))
        lambda_t = ct * self.initial * np.exp(-1.0 * self.theta * t)
        scale = 1.0 / (4.0 * self.theta / ((self.sigma ** 2) * (1.0 - np.exp(-1.0 * self.theta * t))))
        marginal = ncx2(nu, lambda_t, scale=scale)
        return marginal

    def draw(self, n, N, marginal=False, envelope=False, style=None):
        self._draw_qqstyle(n=n, N=N, marginal=marginal, envelope=envelope)
        return 1
