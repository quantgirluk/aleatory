from base_paths import KDEStochasticProcessPaths
from cev_process import CEV_process
import numpy as np
from scipy import stats

class CEVProcessPaths(KDEStochasticProcessPaths):

    def __init__(self, N, gamma=1.0, mu=1.0, sigma=1.0, initial=1.0, n=10, T=1.0, rng=None):
        super().__init__(N=N, T=T, rng=rng)
        self.gamma = gamma
        self.mu = mu
        self.sigma = sigma
        self.initial = initial
        self.n = n
        self.process = CEV_process(gamma=self.gamma, mu=self.mu, sigma=self.sigma, initial=self.initial, n=self.n,
                                  T=self.T)
        self.paths = [self.process.sample(n) for _ in range(int(N))]
        self.times = self.process.times
        self.name = "CEV Process"

    def plot(self):
        self._plot_paths()
        return 1

    def draw(self):
        self._draw_paths(style='qq')
        return 1