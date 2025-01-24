"""Galton-Watson process"""

from aleatory.processes.base_analytical import SPAnalytical
import numpy as np
import pandas as pd
from scipy.stats import poisson
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class GaltonWatson(SPAnalytical):
    r"""
    Galton-Watson process
    =====================

    .. image:: ../_static/galton_watson_draw.png


    Notes
    -----

    A Galton–Watson process is a stochastic process :math:`\{X_n : n \in \mathbb{N}\}` which evolves
    according to the following recurrence formula:

    .. math::

        X_{0} = 1,

    .. math::

        X_{n+1} = \sum_{j=1}^{X_n} Z_j^{(n)},

    where :math:`\{Z_j^{(n)}. : n, j \in \mathbb{N}\}` is  a set of independent and identically
    distributed (i.i.d) natural number-valued random variables. In this case, we assume tha they
    follow a Poisson distribution with parameter :math:`\mu>0`, i.e.:

    .. math::
        Z_j^{(n)} \sim Poi(\mu).

    Constructor, Methods, and Attributes
    ------------------------------------
    """

    def __init__(self, mu=1.0, rng=None):
        """
        :parameter float mu: the parameter :math:`\mu>0` in the above definition
        :parameter numpy.random.Generator rng: a custom random number generator
        """
        super().__init__(rng=rng, initial=1)
        self.mu = mu
        self.offspring_dist = poisson(mu=self.mu)
        self.offspring_mean = self.offspring_dist.mean()
        self.offspring_variance = self.offspring_dist.var()

        self.generations = 0
        self.times = None
        self.name = (
            f"Galton-Watson Process with Poisson Branching\n $Z_i\sim Poi(${self.mu}$)$"
        )

    def _sample_galton_watson(self, generations=None):
        self.T = generations
        self.generations = generations
        populations = [self.initial]
        current_population = self.initial
        for i in range(generations):
            if current_population == 0:  # If the population is extinct, stop
                break
            next_population = sum(
                self.offspring_dist.rvs(1)[0] for _ in range(current_population)
            )
            populations.append(next_population)
            current_population = next_population
        return np.array(populations)

    def sample(self, n=None):
        sample = self._sample_galton_watson(generations=n)
        self.times = np.arange(n + 1)
        return sample

    def sample_upto(self, n=None):
        sample = self._sample_galton_watson(generations=n)
        size = len(sample)
        while size < n:
            sample = self._sample_galton_watson(generations=n)
            size = len(sample)
        else:
            return sample

    def simulate(self, N, n):
        """
        Simulate paths/trajectories from the instanced stochastic process.

        :param int N: number of paths to simulate
        :param int n: number of generations to simulate
        :return: list with N paths (each one is a numpy array of size up to n)

        """
        self.N = N
        self.T = n
        self.paths = [self._sample_galton_watson(generations=n) for _ in range(N)]
        self.times = np.arange(n + 1)
        return self.paths

    def simulate_upto(self, N, generations=None):
        """
        Simulate paths/trajectories from the instanced stochastic process.
        It requires either the number of jumps (`jumps`) or the  time (`T`)
        for the simulation to end.

        - If `generations` is provided, the function returns :math:`N` paths each one with exactly
            that number of generations.

        :param int N: number of paths to simulate
        :param int generations: number of generations
        :return: list with N paths (each one is a numpy array of size n)

        """
        self.N = N
        self.T = generations
        self.paths = [self.sample_upto(n=generations) for _ in range(N)]
        self.times = np.arange(generations + 1)
        return self.paths

    def marginal_expectation(self, generations):
        return (self.offspring_mean**generations) * self.initial

    def marginal_variance(self, generations):

        sigma_squared = self.offspring_variance
        mu = self.offspring_mean
        if self.offspring_mean == 1:
            variances = sigma_squared * generations
        else:
            variances = (
                sigma_squared
                * mu ** (generations - 1.0)
                * (1.0 - mu**generations)
                / (1.0 - mu)
            )
        return variances

    def plot(
        self,
        N,
        n,
        mode="steps",
        title=None,
        style="seaborn-v0_8-whitegrid",
        color_survival=False,
        **fig_kw,
    ):
        """
        Simulates and plots paths/trajectories from the instanced stochastic process. Simple plot of times
        versus process values as lines and/or markers.

        :param int N: number of paths to simulate
        :param int n: number of generations to simulate
        :param str mode: type of plot to plot, defaults to "steps"
        :param str title: title of the plot, if None then it defaults to the name of the process
        :param str style: style to be used, it defaults to "seaborn-v0_8-whitegrid"
        :param bool color_survival: if True, then the plot highlights the paths surviving
        """

        plot_title = title if title else self.name
        self.simulate(N, n=n)
        paths = self.paths

        if color_survival:
            last_points = [path[-1] for path in paths]
            colors = ["grey" if x < 1 else "red" for x in last_points]

            with plt.style.context(style):
                fig, ax = plt.subplots(**fig_kw)
                for path, color in zip(paths, colors):
                    times = np.arange(0, len(path))
                    if mode == "points":
                        ax.scatter(times, path, s=7, color=color)
                    elif mode == "steps":
                        ax.step(times, path, where="post", color=color)
                    elif mode == "linear":
                        ax.plot(times, path)
                    elif mode in ["points+steps", "steps+points"]:
                        ax.step(times, path, where="post", alpha=0.5, color=color)
                        color = plt.gca().lines[-1].get_color()
                        ax.plot(times, path, "o", color=color)
                ax.set_title(plot_title)
                ax.set_xlabel("$n$")
                ax.set_ylabel("$X(n)$")
                plt.show()

        else:
            with plt.style.context(style):
                fig, ax = plt.subplots(**fig_kw)
                for path in paths:
                    times = np.arange(0, len(path))
                    if mode == "points":
                        ax.scatter(times, path, s=10)
                    elif mode == "steps":
                        ax.step(times, path, where="post", linewidth=1.75)
                    elif mode == "linear":
                        ax.plot(times, path)
                    elif mode in ["points+steps", "steps+points"]:
                        ax.step(times, path, where="post", alpha=0.75)
                        color = plt.gca().lines[-1].get_color()
                        ax.plot(times, path, "o", color=color)
                    else:
                        raise ValueError(
                            "mode can only take values 'points', 'steps', 'points+steps', or 'linear'."
                        )

                ax.set_xlim(right=n + 1)
                ax.set_title(plot_title)
                ax.set_xlabel("$n$")
                ax.set_ylabel("$X(n)$")
                plt.show()

        return fig

    def draw(
        self,
        N,
        n,
        mode="steps",
        title=None,
        style="seaborn-v0_8-whitegrid",
        colormap="RdYlBu_r",
        envelope=False,
        marginal=True,
        colorspos=None,
        **fig_kw,
    ):
        """
        Simulates and plots paths/trajectories from the instanced stochastic process.
         Visualisation shows:

         - generations versus process values
         - the expectation of the process across time
         - histogram showing the empirical marginal distribution for the last generation
         - envelope of confidence intervals

        :param int N: number of paths to simulate
        :param int n: number of generations to simulate
        :param str [optional, default="steps"] mode: type of plot to plot
        :param str title: title of the plot, if None then it defaults to the name of the process
        :param str [optional, default="seaborn-v0_8-whitegrid"] style:  style to be used
        :param str [optional, default="RdYlBu_r"] colormap: colormap to be used
        :param bool [optional, default="False] envelope: if True, then the plot highlights the paths envelope
        :param bool [optional, default=True] marginal:  if True, then the plot highlights the paths marginal
        :param double [optional, default=None] colorspos: if provided, it affects the color of the paths
        """

        title = title if title else self.name
        self.simulate(N, n=n)
        paths = self.paths
        times = self.times
        expectations = self.marginal_expectation(times)

        cm = plt.colormaps[colormap]
        last_points = [path[-1] for path in paths]
        n_bins = int(np.sqrt(N))
        col = np.linspace(0, 1, n_bins, endpoint=True)

        with plt.style.context(style):
            if marginal:
                fig = plt.figure(**fig_kw)
                gs = GridSpec(1, 5)
                ax1 = fig.add_subplot(gs[:4])
                ax2 = fig.add_subplot(gs[4:], sharey=ax1)

                num, bins, patches = ax2.hist(
                    last_points, n_bins, orientation="horizontal", density=False
                )
                for c, pat in zip(col, patches):
                    plt.setp(pat, "facecolor", cm(c))
                my_bins = pd.cut(
                    last_points,
                    bins=bins,
                    labels=range(len(bins) - 1),
                    include_lowest=True,
                )
                colors = [col[b] for b in my_bins]

                plt.setp(ax2.get_yticklabels(), visible=False)
                ax2.set_title(f"$X_{{{n}}}$")

                for i in range(N):
                    times_i = times[: len(paths[i])]
                    if mode == "points":
                        ax1.scatter(times_i, paths[i], color=cm(colors[i]), s=10)
                    elif mode == "steps":
                        ax1.step(times_i, paths[i], color=cm(colors[i]), where="post")
                    elif mode == "linear":
                        ax1.plot(times_i, paths[i], color=cm(colors[i]))
                    elif mode in ["points+steps", "steps+points"]:
                        ax1.step(
                            times_i,
                            paths[i],
                            color=cm(colors[i]),
                            where="post",
                            alpha=0.75,
                        )
                        ax1.scatter(times_i, paths[i], color=cm(colors[i]), s=10)

                if expectations is not None:
                    ax1.plot(times, expectations, "--", lw=1.75, label=f"$E[X_n]$")
                    ax1.legend()
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
                    times_i = times[: len(paths[i])]
                    if mode == "points":
                        ax1.scatter(times_i, paths[i], color=cm(colors[i]), s=7)
                    elif mode == "steps":
                        ax1.step(times_i, paths[i], color=cm(colors[i]), where="post")
                    elif mode == "linear":
                        ax1.plot(
                            times_i,
                            paths[i],
                            color=cm(colors[i]),
                        )
                    elif mode in ["points+steps", "steps+points"]:
                        ax1.step(
                            times_i,
                            paths[i],
                            color=cm(colors[i]),
                            where="post",
                            alpha=0.75,
                        )
                        ax1.scatter(times_i, paths[i], color=cm(colors[i]))

                if expectations is not None:
                    ax1.plot(times, expectations, "--", lw=1.75, label="$E[X_n]$")
                    ax1.legend()

            fig.suptitle(title)
            ax1.set_xlim(right=n + 1)
            ax1.set_title(r"Simulated Paths")  # Title
            ax1.set_xlabel("$n$")
            ax1.set_ylabel("$X(n)$")
            plt.show()

        return fig


if __name__ == "__main__":
    p = GaltonWatson(mu=1.5)
    p.plot(N=10, n=10, figsize=(12, 7))
    p.plot(N=10, n=10, figsize=(12, 7), color_survival=True)
    p.draw(N=100, n=10, figsize=(12, 7), colormap="summer")

#     for m in ["steps"]:
#         # for par, gen in zip ([0.8, 1.0, 2.0], [25, 100, 10]):
#         par = 1.0
#         gen = 100
#         p = GaltonWatson(mu=par)
#         # p.plot(N=100, generations=gen, color_survival=False, style=qs, mode=m, figsize=(12, 7), dpi=150)
#         p.plot(N=100, generations=gen, color_survival=False, style=qs, mode=m, figsize=(12, 7),
#                title="What proportion of the surnames will have become extinct after $r$ generations?\n Francis Galton (1873)")
# p.draw(N=10, generations=gen, style=qs, figsize=(12, 7), colormap="cool", marginal=False, mode=m)
# p.draw(N=100, generations=gen, style=qs, figsize=(14, 7), colormap="winter", marginal=True, mode=m)
