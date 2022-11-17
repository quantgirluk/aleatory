from gaussian import Gaussian
from base import StochasticProcess
from utils import check_positive_number, check_positive_integer
import numpy as np


class BaseEulerMaruyamaProcess(StochasticProcess):

    def __int__(self, T=1.0, f=None, g=None, initial=0.0, rng=None):
        super().__init__(T=T, rng=rng)
        self.f = f
        self.g = g
        self.initial = initial
        self._times = None

    def _sample_em_process(self, n):
        check_positive_integer(n)
        dt = 1.0 * self.T / n
        times = np.arange(0.0, self.T + dt, dt)
        self._times = times
        dWs = self.rng.normal(scale=np.sqrt(dt), size=n)

        simulation = [self.initial]
        previous = self.initial

        for (t, dw) in zip(times, dWs):
            previous += self.f(previous, t) * dt + self.g(previous, t) * dw
            simulation.append(previous)

        return simulation

    def sample(self, n):
        return self._sample_em_process(n)
