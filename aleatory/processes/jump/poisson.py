"""Poisson Process"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

from aleatory.processes.base import BaseProcess
from aleatory.utils.utils import check_positive_number, check_positive_integer
from matplotlib.gridspec import GridSpec
import pandas as pd


class PoissonProcess(BaseProcess):
    r"""
    Poisson Process
    ===============

    .. image:: ../_static/poisson_process_draw.png

    Notes
    -----

    A Poisson point process is a type of random mathematical object that consists of points randomly located on
    a mathematical space with the essential feature that the points occur independently of one another.

    A Poisson process :math:`\{N(t) : t\geq 0\}` with intensity rate :math:`\lambda>0`, is defined by the following properties:

    - :math:`N(0)=0`,
    - :math:`N(t)` has a Poisson distribution with parameter :math:`\lambda t`, for each :math:`t> 0`,
    -  It has independent increments.

    Constructor, Methods, and Attributes
    ------------------------------------

    """

    def __init__(self, rate=1.0, rng=None):
        """
        :parameter float rate: the intensity rate :math:`\lambda>0`,
        :parameter numpy.random.Generator rng: a custom random number generator
        """
        super().__init__(rng=rng)
        self.rate = rate
        self.name = f"Poisson Process $N(\\lambda={self.rate})$"
        self.T = None
        self.N = None
        self.paths = None

    def __str__(self):
        return f"Poisson Process with intensity rate {self.rate}"

    def __repr__(self):
        return f"PoissonProcess(rate={self.rate})"

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
            t = 0.0
            arrival_times = [0.0]
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
        It requires either the number of jumps (`jumps`) or the  time (`T`)
        for the simulation to end.

        - If `jumps` is provided, the function returns :math:`N` paths each one with exactly
            that number of jumps.

        - If `T` is provided, the function returns :math:`N` paths over the time :math:`[0,T]`. Note
            that in this case each path can have a different number of jumps.

        :param int N: number of paths to simulate
        :param int jumps: number of jumps
        :param float T: time T
        :return: list with N paths (each one is a numpy array of size n)

        """
        self.N = N
        self.paths = [self.sample(jumps=jumps, T=T) for _ in range(N)]
        return self.paths

    def plot(
        self,
        N,
        jumps=None,
        T=None,
        style="seaborn-v0_8-whitegrid",
        mode="steps",
        title=None,
        **fig_kw,
    ):
        """
        Simulates and plots paths/trajectories from the instanced stochastic process. Simple plot of times
        versus process values as lines and/or markers
        :param int N: number of paths to simulate
        :param int jumps: number of jumps
        :param float T: time T
        :param str style: style of plot
        :param str mode: type of plot
        :param str title: title of plot
        """

        if jumps and T:
            raise ValueError("Only one must be provided either jumps or T")

        plot_title = title if title else self.name
        self.simulate(N, jumps=jumps, T=T)
        paths = self.paths

        with plt.style.context(style):
            fig, ax = plt.subplots(**fig_kw)
            for p in paths:
                counts = np.arange(0, len(p))
                if mode == "points":
                    ax.scatter(p, counts, s=10)
                elif mode == "steps":
                    ax.step(p, counts, where="post", linewidth=1.25)
                elif mode == "linear":
                    ax.plot(p, counts)
                elif mode == "points+steps":
                    ax.step(p, counts, where="post", alpha=0.5)
                    color = plt.gca().lines[-1].get_color()
                    ax.plot(p, counts, "o", color=color, markersize=6)

            ax.set_title(plot_title)
            ax.set_xlabel("$t$")
            ax.set_ylabel("$N(t)$")
            if T is not None:
                ax.set_xlim(right=T)
            if jumps is not None:
                ax.set_ylim(top=jumps + 2)
            plt.show()

        return fig

    def draw(
        self,
        N,
        T=10.0,
        style="seaborn-v0_8-whitegrid",
        colormap="RdYlBu_r",
        envelope=False,
        marginal=True,
        mode="steps",
        colorspos=None,
        title=None,
        **fig_kw,
    ):

        title = title if title else self.name
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

                n, bins, patches = ax2.hist(
                    last_points, n_bins, orientation="horizontal", density=True
                )
                for c, p in zip(col, patches):
                    plt.setp(p, "facecolor", cm(c))
                my_bins = pd.cut(
                    last_points,
                    bins=bins,
                    labels=range(len(bins) - 1),
                    include_lowest=True,
                )
                colors = [col[b] for b in my_bins]

                if marginal and marginalT:
                    marginaldist = marginalT
                    x = np.arange(marginaldist.ppf(0.001), marginaldist.ppf(0.999) + 1)
                    ax2.plot(
                        marginaldist.pmf(x),
                        x,
                        "o",
                        linestyle="",
                        color="maroon",
                        markersize=2,
                        label="$N_T$ pmf",
                    )
                    ax2.axhline(
                        y=marginaldist.mean(), linestyle="--", lw=1.75, label="$E[N_T]$"
                    )
                    ax2.legend()

                plt.setp(ax2.get_yticklabels(), visible=False)
                ax2.set_title("$N_T$")

                for i, p in enumerate(paths):
                    counts = np.arange(0, len(p))
                    if mode == "points":
                        ax1.scatter(p, counts, color=cm(colors[i]), s=10)
                    elif mode == "steps":
                        ax1.step(
                            p, counts, color=cm(colors[i]), where="post", linewidth=1.25
                        )
                    elif mode == "points+steps":
                        ax1.step(
                            p, counts, color=cm(colors[i]), where="post", linewidth=1.25
                        )
                        ax1.plot(p, counts, "o", color=cm(colors[i]), markersize=6)
                    else:
                        raise ValueError(
                            "mode can only take values 'points', 'steps' or 'points+steps'"
                        )
                if expectations is not None:
                    ax1.plot(times, expectations, "--", lw=1.75, label="$E[N_t]$")
                    ax1.legend()
                if envelope:
                    ax1.fill_between(times, upper, lower, alpha=0.25, color="silver")
                plt.subplots_adjust(wspace=0.2, hspace=0.5)

            else:
                fig, ax1 = plt.subplots(**fig_kw)
                if colorspos:
                    colors = [path[colorspos] / np.max(np.abs(path)) for path in paths]
                else:
                    _, bins = np.histogram(last_points, n_bins)
                    my_bins = pd.cut(
                        last_points,
                        bins=bins,
                        labels=range(len(bins) - 1),
                        include_lowest=True,
                    )
                    colors = [col[b] for b in my_bins]

                for i in range(N):
                    counts = np.arange(0, len(paths[i]))
                    ax1.step(
                        paths[i], counts, color=cm(colors[i]), lw=0.75, where="post"
                    )

                if expectations is not None:
                    ax1.plot(times, expectations, "--", lw=1.75, label="$E[N_t]$")
                    ax1.legend()
                if envelope:
                    ax1.fill_between(times, upper, lower, color="lightgray", alpha=0.25)

            fig.suptitle(title)
            ax1.set_xlim(right=T)
            ax1.set_title(r"Simulated Paths $N_t, t \leq T$")  # Title
            ax1.set_xlabel("$t$")
            ax1.set_ylabel("$N(t)$")
            plt.show()

        return fig


if __name__ == "__main__":

    p = PoissonProcess()
    p.plot(N=5, T=10)
