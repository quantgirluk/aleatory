from replica.sp_paths.base_paths import KDEStochasticProcessPaths
from replica.processes.euler_maruyama.cev_process import CEV_process
import numpy as np


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
        self._marginals = None
        self.times = self.process.times
        self.name = "CEV Process"

    def _get_empirical_marginal_samples(self):
        empirical_marginal_samples = np.array(self.paths).transpose()
        return empirical_marginal_samples

    def estimate_expectations(self):

        if self._marginals is None:
            self._marginals = self._get_empirical_marginal_samples()

        empirical_means = [np.mean(m) for m in self._marginals]
        return empirical_means

    def estimate_variances(self):
        if self._marginals is None:
            self._marginals = self._get_empirical_marginal_samples()
        empirical_vars = [np.var(m) for m in self._marginals]
        return empirical_vars

    def estimate_stds(self):
        variances = self.estimate_variances()
        stds = [np.sqrt(var) for var in variances]
        return stds

    def estimate_quantiles(self, q):
        if self._marginals is None:
            self._marginals = self._get_empirical_marginal_samples()
        empirical_quantiles = [np.quantile(m, q) for m in self._marginals]
        return empirical_quantiles

    def plot(self):
        self._plot_paths()
        return 1

    def draw(self):
        expectations = self.estimate_expectations()
        lower = self.estimate_quantiles(0.005)
        upper = self.estimate_quantiles(0.995)
        self._draw_paths(expectations, lower, upper)
        return 1
