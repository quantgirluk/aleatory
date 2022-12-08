from replica.processes.euler_maruyama.base_EMProcess import BaseEulerMaruyamaProcess
import numpy as np


class CIRProcess(BaseEulerMaruyamaProcess):
    def __init__(self, theta=1.0, mu=1.0, sigma=1.0, initial=0.0, T=1.0, n=1, rng=None):
        super().__init__(T=T, rng=rng)
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.initial = initial
        self.n = n
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
