from brownian_motion import BrownianMotion
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
import numpy as np
import pandas as pd
from abc import ABC
from abc import abstractmethod

SAVE = False
plt.style.use(
    'https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')


class StochasticProcessPaths(ABC):
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

    @abstractmethod
    def sample(self, n):  # pragma: no cover
        pass


class BrownianPaths(StochasticProcessPaths):

    def __init__(self, N, times, drift=0.0, scale=1.0):
        self.N = N
        self.times = times
        self.drift = drift
        self.scale = scale
        brownian = BrownianMotion(drift=self.drift, scale=self.scale)
        self.paths = [brownian.sample_at(times) for k in range(int(N))]

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

    def _draw_paths(self):
        for p in self.paths:
            plt.plot(self.times, p)
            plt.title('Brownian Motion Paths')
        plt.show()
        return 1

    def plot(self):
        self._draw_paths()
        return 1

    def draw(self):

        fig = plt.figure(figsize=(12, 6))
        gs = GridSpec(1, 5, figure=fig)
        ax1 = fig.add_subplot(gs[:4])
        ax2 = fig.add_subplot(gs[4:], sharey=ax1)

        brownian_paths = self.paths
        last_points = [path[-1] for path in brownian_paths]
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

        T = self.times[-1]
        WT_variable = norm(loc=self.drift * T, scale=self.scale * np.sqrt(T))
        x = np.linspace(WT_variable.ppf(0.001), WT_variable.ppf(0.999), 100)
        ax2.plot(WT_variable.pdf(x), x, '--', lw=1.75, alpha=0.6, color='blue', label='$X_T$ pdf')
        ax2.axhline(y=self.drift * T, color='black', lw=1.2, label='$E[X_T]$')
        plt.setp(ax2.get_yticklabels(), visible=False)

        for i in range(self.N):
            ax1.plot(self.times, brownian_paths[i], '-', lw=1.5, color=cm(colors[i]))
        ax1.plot(self.times, self._process_expectation(), '-', lw=1.5, color='black', label='$E[X_t]$')

        fig.suptitle('Brownian Motion with Drift', size=14)
        ax1.set_title("Simulated Paths $X_t, t \in [t_0, T]$", size=12)  # Title
        ax2.set_title('$X_T$', size=12)  # Title
        plt.subplots_adjust(wspace=0.025, hspace=0.025)
        ax1.legend()
        ax2.legend()

        if SAVE:
            plt.savefig(r"Figures/BMDrift.png",
                        dpi=200
                        # bbox_inches='tight',
                        # pad_inches=1
                        )
        plt.show()

        return 1
