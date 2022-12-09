from replica.processes.base import SPEulerMaruyama
import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SAVE = False


class CEVProcess(SPEulerMaruyama):

    def __init__(self, gamma=1.0, mu=1.0, sigma=1.0, initial=1.0, T=1.0, rng=None):
        super().__init__(T=T, rng=rng)
        self.gamma = gamma
        self.mu = mu
        self.sigma = sigma
        self.initial = initial
        self.n = 1.0
        self.dt = 1.0 * self.T / self.n
        self.times = np.arange(0.0, self.T + self.dt, self.dt)
        self.name = "CEV Process"
        self.paths=None
        self._marginals = None


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

    def _get_empirical_marginal_samples(self):
        if self.paths is None:
            self.simulate(self.n, self.N)

        empirical_marginal_samples = np.array(self.paths).transpose()
        return empirical_marginal_samples

    def get_marginal(self, t):
        pass

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

    def _process_expectation(self):
        return self.estimate_expectations()

    def process_expectation(self):
        return self._process_expectation()

    def _process_variance(self):
        return self.estimate_variances()

    def process_variance(self):
        return self._process_variance()

    def _process_stds(self):
        stds = np.sqrt(self.process_variance())
        return stds

    def process_stds(self):
        stds = self._process_stds()
        return stds

    def _draw_paths_kde(self, expectations, lower, upper):

        with plt.style.context('seaborn-whitegrid'):
            plt.rcParams['figure.dpi'] = 300

            fig = plt.figure(figsize=(10, 5))
            gs = GridSpec(1, 5, figure=fig)

            ax1 = fig.add_subplot(gs[:4])
            ax2 = fig.add_subplot(gs[4:], sharey=ax1)

            paths = self.paths
            last_points = [path[-1] for path in paths]

            cm = plt.cm.get_cmap('RdYlBu_r')
            n_bins = int(np.sqrt(self.N))
            n, bins, patches = ax2.hist(last_points, n_bins, color='green', orientation='horizontal', density=True)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            col = bin_centers - min(bin_centers)  # scale values to interval [0,1]
            col /= max(col)
            for c, p in zip(col, patches):
                plt.setp(p, 'facecolor', cm(c))
            my_bins = pd.cut(last_points, bins=bins, labels=range(len(bins) - 1), include_lowest=True)
            colors = [col[b] for b in my_bins]

            T = self.T
            kde = sm.nonparametric.KDEUnivariate(last_points)
            kde.fit()  # Estimate the densities
            ax2.plot(kde.density, kde.support, '--', lw=1.75, alpha=0.6, color='blue', label='$X_T$  KDE', zorder=10)
            ax2.axhline(y=np.mean(last_points), color='black', lw=1.2, label=r'$\overline{X_T}$')
            plt.setp(ax2.get_yticklabels(), visible=False)

            for i in range(self.N):
                ax1.plot(self.times, paths[i], '-', lw=1.5, color=cm(colors[i]))

            ax1.plot(self.times, expectations, '-', lw=1.5, color='black', label=r'$\overline{X_t}$  (Empirical Means)')

            ax1.fill_between(self.times, upper, lower, alpha=0.25, color='grey')

            fig.suptitle(self.name, size=14)
            ax1.set_title('Simulated Paths $X_t, t \in [t_0, T]$', size=12)  # Title
            ax2.set_title('$X_T$', size=12)  # Title
            ax1.set_xlabel('t')
            ax1.set_ylabel('X(t)')
            plt.subplots_adjust(wspace=0.025, hspace=0.025)
            ax1.legend()
            ax2.legend()
            plt.show()

        return 1

    def draw(self, n, N, marginal=False, envelope=False, style=None):
        self.simulate(n, N)
        expectations = self.estimate_expectations()
        lower = self.estimate_quantiles(0.005)
        upper = self.estimate_quantiles(0.995)
        self._draw_paths_kde(expectations, lower, upper)
        return 1
