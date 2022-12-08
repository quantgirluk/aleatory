from replica.processes.euler_maruyama.base_EMProcess import BaseEulerMaruyamaProcess
import numpy as np


class CEV_process(BaseEulerMaruyamaProcess):

    def __init__(self, gamma=1.0, mu=1.0, sigma=1.0, initial=1.0, T=1.0, n=1, rng=None):
        super().__init__(T=T, rng=rng)
        self.gamma = gamma
        self.mu = mu
        self.sigma = sigma
        self.initial = initial
        self.n = n
        self.dt = 1.0 * self.T / self.n
        self.times = np.arange(0.0, self.T + self.dt, self.dt)
        self.name = "CEV Process"

        def f(x, _):
            return self.mu * x

        def g(x, _):
            return self.sigma * (x ** self.gamma)

        self.f = f
        self.g = g

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        if value < 0:
            raise ValueError("sigma cannot be negative")
        self._sigma = value

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        if value < 0:
            raise ValueError("gamma cannot be negative")
        self._gamma = value

    def __str__(self):
        return "CEV process with parameters {gamma}, {drift}, and {volatility} on [0, {T}].".format(
            T=str(self.T), gamma=str(self.gamma), drift=str(self.mu), volatility=str(self.sigma))
