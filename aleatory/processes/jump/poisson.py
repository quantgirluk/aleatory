"""Poisson Process"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

from aleatory.processes.base import BaseProcess
from aleatory.utils.utils import check_positive_number, check_positive_integer
from matplotlib.gridspec import GridSpec
import pandas as pd


class PoissonProcess(BaseProcess):

    def __init__(self, rate=1.0, rng=None):
        super().__init__(rng=rng)
        self.name = 'Poisson Process'
        self.rate = rate
        self.T = None
        self.N = None
        self.paths = None

    def __str__(self):
        return "Poisson Process with intensity rate r={rate}.".format(rate=str(self.rate))

    def __repr__(self):
        return "PoissonProcess(rate={r})".format(r=str(self.rate))

    @property
    def rate(self):
        return self._rate

    @rate.setter
    def rate(self, value):
        if value < 0:
            raise ValueError("rate must be positive")
        self._rate = value

    def _sample_poisson_process(self, jumps=None, T=None):
        exp_mean = 1.0 / self.rate
        if jumps is not None and T is not None:
            raise ValueError("Only one must be provided either jumps or T")
        elif jumps:
            check_positive_integer(jumps)

            exponential_times = self.rng.exponential(exp_mean, size=jumps)
            arrival_times = np.cumsum(exponential_times)
            arrival_times = np.insert(arrival_times, 0, [0])
            return arrival_times
        elif T:
            check_positive_number(T, "Time")
            t = 0.
            arrival_times = [0.]
            while t < T:
                t += self.rng.exponential(scale=exp_mean)
                arrival_times.append(t)
            return np.array(arrival_times)

    def sample(self, jumps=None, T=None):
        return self._sample_poisson_process(jumps=jumps, T=T)

    def get_marginal(self, t):
        return poisson(self.rate * t)

    def marginal_expectation(self, times):
        return self.rate * times

    def simulate(self, N, jumps=None, T=None):
        """
        Simulate paths/trajectories from the instanced stochastic process.

        :param N: number of paths to simulate
        :param jumps: number of jumps
        :param T: time T
        :return: list with N paths (each one is a numpy array of size n)
        """
        self.N = N
        self.paths = [self.sample(jumps=jumps, T=T) for _ in range(N)]
        return self.paths

    def plot(self, N, jumps=None, T=None, style="seaborn-v0_8-whitegrid", title=None, **fig_kw):

        plot_title = title if title else self.name
        self.simulate(N, jumps=jumps, T=T)
        paths = self.paths
        # max_time = np.max(np.hstack(paths))

        with plt.style.context(style):
            fig, ax = plt.subplots(**fig_kw)
            for p in paths:
                counts = np.arange(0, len(p))
                ax.step(p, counts)
            ax.set_title(plot_title)
            ax.set_xlabel('$t$')
            ax.set_ylabel('$N(t)$')
            if T is not None:
                ax.set_xlim(right=T)
            if jumps is not None:
                ax.set_ylim(top=jumps+2)
            plt.show()

        return fig


    def draw(self, N, T=None, style="seaborn-v0_8-whitegrid", colormap="RdYlBu_r", envelope=True,
                  marginal=True, colorspos=None, **fig_kw):

        title = self.name
        self.simulate(N, T=T)
        paths = self.paths

        times = np.linspace(0.0, T, 200)
        marginals = [self.get_marginal(t) for t in times]
        expectations = self.marginal_expectation(times)
        lower = [m.ppf(0.005) for m in marginals]
        upper = [m.ppf(0.9995) for m in marginals]

        marginalT = self.get_marginal(T)

        cm = plt.colormaps[colormap]
        last_points = [len(path) - 1 for path in paths]
        n_bins = int(np.sqrt(N))
        col = np.linspace(0, 1, n_bins, endpoint=True)

        with plt.style.context(style):
            if marginal:
                fig = plt.figure(**fig_kw)
                gs = GridSpec(1, 5)
                ax1 = fig.add_subplot(gs[:4])
                ax2 = fig.add_subplot(gs[4:], sharey=ax1)

                n, bins, patches = ax2.hist(last_points, n_bins, orientation='horizontal', density=True)
                for c, p in zip(col, patches):
                    plt.setp(p, 'facecolor', cm(c))
                my_bins = pd.cut(last_points, bins=bins, labels=range(len(bins) - 1), include_lowest=True)
                colors = [col[b] for b in my_bins]

                if marginal and marginalT:
                    marginaldist = marginalT
                    x = np.arange(marginaldist.ppf(0.001), marginaldist.ppf(0.999) + 1)
                    ax2.plot(marginaldist.pmf(x), x, 'o', linestyle='', color="maroon", markersize=2, label='$N_T$ pmf')
                    ax2.axhline(y=marginaldist.mean(), linestyle='--', lw=1.75, label='$E[N_T]$')
                    ax2.legend()

                plt.setp(ax2.get_yticklabels(), visible=False)
                ax2.set_title('$N_T$')

                for i in range(N):
                    counts = np.arange(0, len(paths[i]))
                    ax1.step(paths[i], counts, color=cm(colors[i]), lw=0.75)

                if expectations is not None:
                    ax1.plot(times, expectations, '--', lw=1.75, label='$E[N_t]$')
                    ax1.legend()
                if envelope:
                    ax1.fill_between(times, upper, lower, alpha=0.25, color='grey')
                plt.subplots_adjust(wspace=0.2, hspace=0.5)

            else:
                fig, ax1 = plt.subplots(**fig_kw)
                if colorspos:
                    colors = [path[colorspos] / np.max(np.abs(path)) for path in paths]
                else:
                    _, bins = np.histogram(last_points, n_bins)
                    my_bins = pd.cut(last_points, bins=bins, labels=range(len(bins) - 1), include_lowest=True)
                    colors = [col[b] for b in my_bins]

                for i in range(N):
                    counts = np.arange(0, len(paths[i]))
                    ax1.step(paths[i], counts, color=cm(colors[i]), lw=0.75)

                if expectations is not None:
                    ax1.plot(times, expectations, '--', lw=1.75, label='$E[N_t]$')
                    ax1.legend()
                if envelope:
                    ax1.fill_between(times, upper, lower, color='grey', alpha=0.25)

            fig.suptitle(title)
            ax1.set_xlim(right=T)
            ax1.set_title(r'Simulated Paths $N_t, t \leq T$')  # Title
            ax1.set_xlabel('$t$')
            ax1.set_ylabel('$N(t)$')
            plt.show()

        return fig


# p = PoissonProcess(rate=0.5)
# plot1 = p.plot(N=20, T=20, dpi=200, figsize=(9.5, 6),)
# plot2 = p.plot(N=100, jumps=20, dpi=200,figsize=(9.5, 6)),
# fig1 = p.draw(N=300, T=20, figsize=(12, 6), dpi=200, colormap="cool", )
# fig2 = p.draw(N=300, T=40, figsize=(12, 6), dpi=200, colormap="PiYG", )
# fig3 = p.draw(N=300, T=50, figsize=(9, 6), dpi=200, colormap="BrBG", marginal=False)
# fig4 = p.draw(N=300, T=50, figsize=(12, 6), dpi=200, colormap="PuOr", envelope=False)
#
