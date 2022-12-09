from replica.processes.exact_solution.geometric_brownian_motion_backup import GeometricBrownianMotion
from replica.sp_paths.base_paths import ExactStochasticProcessPaths
from scipy.stats import lognorm
import numpy as np


class GBMPaths(ExactStochasticProcessPaths):
    def __init__(self, N, times, drift=0.0, volatility=1.0, initial=1.0, T=1.0, rng=None):
        super().__init__(T=T, N=N, rng=rng)
        self.times = times
        self.drift = drift
        self.volatility = volatility
        self.initial = initial
        self.process = GeometricBrownianMotion(T=self.T, drift=drift, volatility=volatility)
        self.paths = [self.process.sample_at(times, initial=initial) for _ in range(int(N))]
        self.name = "Geometric Brownian Motion"

    def _process_expectation(self):
        return self.initial * np.exp(self.drift * self.times)

    def process_expectation(self):
        expectations = self._process_expectation()
        return expectations

    def _process_variance(self):
        variances = (self.initial ** 2) * np.exp(2 * self.drift * self.times) * (
                np.exp(self.times * self.volatility ** 2) - 1)
        return variances

    def process_variance(self):
        variances = self._process_variance()
        return variances

    def get_marginal(self, t):
        mu_x = np.log(self.initial) + (self.drift - 0.5 * self.volatility ** 2) * t
        sigma_x = self.volatility * np.sqrt(t)
        marginal = lognorm(s=sigma_x, scale=np.exp(mu_x))

        return marginal

    def plot(self):
        self._plot_paths()
        return 1

    def draw(self):
        self._draw_paths(style='qq')
        return 1

    def draw_envelope(self):
        self._draw_envelope_paths(style='qq')
        return 1
