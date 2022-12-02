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

    def _get_empirical_marginal_samples(self):
        empirical_marginal_samples = np.array(self.paths).transpose()
        return empirical_marginal_samples

    def estimate_expectations(self):

        marginal_samples= self._get_empirical_marginal_samples()
        empirical_means = [np.mean(m) for m in marginal_samples]
        return empirical_means

    def estimate_variances(self):

        marginal_samples = self._get_empirical_marginal_samples()
        empirical_vars = [np.var(m) for m in marginal_samples]
        return empirical_vars

    def estimate_quantiles(self, q):

        marginal_samples = self._get_empirical_marginal_samples()
        empirical_vars = [np.quantile(m, q) for m in marginal_samples]
        return empirical_vars

    def estimate_stds(self):
        variances = self.estimate_variances()
        stds = [np.sqrt(var) for var in variances]
        return stds

    def plot(self):
        self._plot_paths()
        return 1

    def draw(self):
        self._draw_paths(expectations=self.estimate_expectations())
        return 1
