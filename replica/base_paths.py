from abc import ABC
from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.gridspec import GridSpec

SAVE = False


class StochasticProcess(ABC):
    def __init__(self, rng=None):
        self.rng = rng

    @property
    def rng(self):
        if self._rng is None:
            return np.random.default_rng()
        return self._rng

    @rng.setter
    def rng(self, value):
        if value is None:
            self._rng = None
        elif isinstance(value, (np.random.RandomState, np.random.Generator)):
            self._rng = value
        else:
            raise TypeError("rng must be of type `numpy.random.Generator`")


class KDEStochasticProcessPaths(StochasticProcess):
    def __init__(self, T=1.0, N=1, initial=0.0, rng=None):
        super().__init__(rng=rng)
        self.initial = initial
        self.T = T
        self.N = N
        self.times = None
        self.paths = None
        self.name = None

    def _plot_paths(self):
        for p in self.paths:
            plt.plot(self.times, p)
            plt.title(self.name)
        plt.show()
        return 1

    def _draw_paths(self, style=None):
        with plt.style.context('seaborn-whitegrid'):
            # with plt.style.context(
            #         'https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle'):

            fig = plt.figure(figsize=(12, 6))
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
            ax2.plot(kde.density, kde.support, '--',  lw=1.75, alpha=0.6, color='blue', label='$X_T$ KDE', zorder=10)
            # marginal = self.get_marginal(T)
            # x = np.linspace(marginal.ppf(0.005), marginal.ppf(0.995), 100)
            # ax2.plot(marginal.pdf(x), x, '--', lw=1.75, alpha=0.6, color='blue', label='$X_T$ pdf')
            # ax2.axhline(y=marginal.mean(), color='black', lw=1.2, label='$E[X_T]$')
            plt.setp(ax2.get_yticklabels(), visible=False)

            for i in range(self.N):
                ax1.plot(self.times, paths[i], '-', lw=1.5, color=cm(colors[i]))

            # expectations = self._process_expectation()
            # ax1.plot(self.times, expectations, '-', lw=1.5, color='black', label='$E[X_t]$')

            # if style == '3sigma':
            #     stds = self._process_stds()
            #     upper = expectations + 3.0 * stds
            #     lower = expectations - 3.0 * stds
            #
            # if style == 'qq':
            #     marginals = [self.get_marginal(t) for t in self.times[1:]]
            #     upper = [self.initial] + [m.ppf(0.005) for m in marginals]
            #     lower = [self.initial] + [m.ppf(0.995) for m in marginals]

            # ax1.fill_between(self.times, upper, lower, alpha=0.25, color='grey')

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


class ExactStochasticProcessPaths(StochasticProcess):
    def __init__(self, T=1.0, N=1, initial=0.0, rng=None):
        super().__init__(rng=rng)
        self.initial = initial
        self.T = T
        self.N = N
        self.times = None
        self.paths = None
        self.name = None

    def _plot_paths(self):
        for p in self.paths:
            plt.plot(self.times, p)
            plt.title(self.name)
        plt.show()
        return 1

    @abstractmethod
    def plot(self):  # pragma: no cover
        pass

    @abstractmethod
    def get_marginal(self, t):
        pass

    def _process_expectation(self):
        pass

    def _process_variance(self):
        pass

    def _process_stds(self):
        pass

    def _draw_paths(self, style=None):
        with plt.style.context('seaborn-whitegrid'):
            # with plt.style.context(
            #         'https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle'):

            fig = plt.figure(figsize=(12, 6))
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
            marginal = self.get_marginal(T)
            x = np.linspace(marginal.ppf(0.005), marginal.ppf(0.995), 100)
            ax2.plot(marginal.pdf(x), x, '--', lw=1.75, alpha=0.6, color='blue', label='$X_T$ pdf')
            ax2.axhline(y=marginal.mean(), color='black', lw=1.2, label='$E[X_T]$')
            plt.setp(ax2.get_yticklabels(), visible=False)

            for i in range(self.N):
                ax1.plot(self.times, paths[i], '-', lw=1.5, color=cm(colors[i]))

            expectations = self._process_expectation()
            ax1.plot(self.times, expectations, '-', lw=1.5, color='black', label='$E[X_t]$')

            if style == '3sigma':
                stds = self._process_stds()
                upper = expectations + 3.0 * stds
                lower = expectations - 3.0 * stds

            if style == 'qq':
                marginals = [self.get_marginal(t) for t in self.times[1:]]
                upper = [self.initial] + [m.ppf(0.005) for m in marginals]
                lower = [self.initial] + [m.ppf(0.995) for m in marginals]

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

    def _draw_envelope_paths(self, style=None):
        with plt.style.context('seaborn-whitegrid'):
            paths = self.paths
            fig = plt.figure(figsize=(48 / 5, 6))
            for i in range(self.N):
                plt.plot(self.times, paths[i], '-', lw=1.5)
            expectations = self._process_expectation()
            plt.plot(self.times, expectations, '-', lw=1.5, color='black', label='$E[X_t]$')

            if style == '3sigma':
                stds = self._process_stds()
                upper = expectations + 3.0 * stds
                lower = expectations - 3.0 * stds

            if style == 'qq':
                marginals = [self.get_marginal(t) for t in self.times[1:]]
                upper = [self.initial] + [m.ppf(0.005) for m in marginals]
                lower = [self.initial] + [m.ppf(0.995) for m in marginals]

            plt.fill_between(self.times, upper, lower, color='grey', alpha=0.25)

            plt.suptitle(self.name)
            plt.title('Simulated Paths $X_t, t \in [t_0, T]$')  # Title
            plt.xlabel('t')
            plt.ylabel('X(t)')
            plt.legend()
            plt.show()

    @abstractmethod
    def draw(self):
        pass

    @abstractmethod
    def draw_envelope(self):
        pass
