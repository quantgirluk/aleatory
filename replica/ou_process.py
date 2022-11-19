from base_EMProcess import BaseEulerMaruyamaProcess
import numpy as np


class OUProcess(BaseEulerMaruyamaProcess):

    def __init__(self, theta=1.0, mu=1.0, sigma=1.0, initial=0.0, T=1.0, n=1, rng=None):
        super().__init__(T=T, rng=rng)
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.initial = initial
        self.n = n
        self.dt = 1.0 * self.T / self.n
        self.times = None

        def f(x, _):
            return self.theta * (self.mu - x)

        def g(x, _):
            return self.sigma

        self.f = f
        self.g = g
