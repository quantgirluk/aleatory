from abc import ABC
from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

SAVE = False


class StochasticProcessPaths(ABC):
    def __init__(self, T=1.0, N=1, rng=None):
        self.rng = rng
        self.T = T
        self.N = N
        self.times = None
        self.paths = None
        self.name = None

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

    def _draw_paths(self):

        with plt.style.context(
                'https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle'):

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
            x = np.linspace(marginal.ppf(0.001), marginal.ppf(0.999), 100)
            ax2.plot(marginal.pdf(x), x, '--', lw=1.75, alpha=0.6, color='blue', label='$X_T$ pdf')
            ax2.axhline(y=marginal.mean(), color='black', lw=1.2, label='$E[X_T]$')
            plt.setp(ax2.get_yticklabels(), visible=False)

            for i in range(self.N):
                ax1.plot(self.times, paths[i], '-', lw=1.5, color=cm(colors[i]))
            ax1.plot(self.times, self._process_expectation(), '-', lw=1.5, color='black', label='$E[X_t]$')

            fig.suptitle(self.name, size=14)
            ax1.set_title('Simulated Paths $X_t, t \in [t_0, T]$', size=12)  # Title
            ax2.set_title('$X_T$', size=12)  # Title
            ax1.set_xlabel('time')
            ax1.set_ylabel('value')
            plt.subplots_adjust(wspace=0.025, hspace=0.025)
            ax1.legend()
            ax2.legend()
            plt.show()

        return 1

    @abstractmethod
    def draw(self):
        pass
