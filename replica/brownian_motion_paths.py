from base_paths import StochasticProcessPaths
from brownian_motion import BrownianMotion
from scipy.stats import norm
import numpy as np


class BrownianPaths(StochasticProcessPaths):
    def __init__(self, N, times, drift=0.0, scale=1.0, T=1.0, rng=None):
        super().__init__(T=T, N=N, rng=rng)
        self.times = times
        self.drift = drift
        self.scale = scale
        self.name = "Brownian Motion" if drift == 0.0 else "Brownian Motion with Drift"
        self.process = BrownianMotion(T=self.T, drift=self.drift, scale=self.scale)
        self.paths = [self.process.sample_at(self.times) for _ in range(int(self.N))]

    def _process_expectation(self):
        return self.drift * self.times

    def process_expectation(self):
        expectations = self._process_expectation()
        return expectations

    def _process_variance(self):
        return self.scale * np.sqrt(self.times)

    def process_variance(self):
        variances = self._process_variance()
        return variances

    def get_marginal(self, t):
        marginal = norm(loc=self.drift * t, scale=self.scale * np.sqrt(t))
        return marginal

    def plot(self):
        self._plot_paths()
        return 1

    def draw(self):
        self._draw_paths()
        return 1
