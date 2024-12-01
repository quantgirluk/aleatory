from aleatory.processes.base_analytical import SPAnalytical
import numpy as np
import pandas as pd
from scipy.stats import poisson
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class GaltonWatson(SPAnalytical):

    def __init__(self, mu=1.0, rng=None):
        super().__init__(rng=rng, initial=0.0)
        self.offspring_dist = poisson(mu=mu)
        self.generations = 0
        self.initial = 1
        self.times = None
        self.name = "Galton Watson Process with Poisson branching"

    def _sample_galton_watson(self, generations=None):
        self.T = generations
        self.generations = generations

        populations = np.full(generations + 1, 0)
        populations[0] = self.initial

        current_population = self.initial
        for i in range(1, len(populations)):
            if current_population == 0:  # If the population is extinct, stop early
                break
            next_population = sum(self.offspring_dist.rvs(1)[0] for _ in range(current_population))
            populations[i] = next_population
            current_population = next_population

        return populations

    def sample(self, generations=None):
        sample = self._sample_galton_watson(generations=generations)
        self.times = np.arange(0, len(sample))
        return

    def simulate(self, N, generations=None):
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
        self.paths = [self._sample_galton_watson(generations=generations) for _ in range(N)]
        self.times = np.arange(0, len(self.paths[0]))
        return self.paths

    def plot(self, N, generations=None, plot_style="steps",
             style="seaborn-v0_8-whitegrid", title=None, **fig_kw):
        """
        Simulates and plots paths/trajectories from the instanced stochastic process. Simple plot of times
        versus process values as lines and/or markers.
        """

        plot_title = title if title else self.name
        self.simulate(N, generations=generations)
        paths = self.paths
        times = self.times

        with plt.style.context(style):
            fig, ax = plt.subplots(**fig_kw)
            for p in paths:
                if plot_style == 'points':
                    ax.scatter(times, p, s=7)
                elif plot_style == 'steps':
                    ax.step(times, p, where='post')
                elif plot_style == 'linear':
                    ax.plot(times, p)
                elif plot_style == 'points+steps':
                    ax.step(times, p, where='post', alpha=0.5)
                    color = plt.gca().lines[-1].get_color()
                    ax.plot(times, p, 'o', color=color)
                else:
                    raise ValueError("plot_style must be 'points', 'steps', 'points+steps' or 'linear'.")

            ax.set_title(plot_title)
            ax.set_xlabel('$t$')
            ax.set_ylabel('$X(t)$')
            plt.show()

        return fig

    def draw(self, N, generations=None, style="seaborn-v0_8-whitegrid", colormap="RdYlBu_r", envelope=False,
             marginal=True, colorspos=None, **fig_kw):

        title = self.name
        self.simulate(N, generations=generations)
        paths = self.paths
        times = self.times

        # marginals = [self.get_marginal(t) for t in times]
        # expectations = self.marginal_expectation(times)
        # lower = [m.ppf(0.005) for m in marginals]
        # upper = [m.ppf(0.9995) for m in marginals]
        #
        # marginalT = self.get_marginal(T)

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

                n, bins, patches = ax2.hist(last_points, n_bins, orientation='horizontal', density=True)
                for c, p in zip(col, patches):
                    plt.setp(p, 'facecolor', cm(c))
                my_bins = pd.cut(last_points, bins=bins, labels=range(len(bins) - 1), include_lowest=True)
                colors = [col[b] for b in my_bins]

                # if marginal and marginalT:
                #     marginaldist = marginalT
                #     x = np.arange(marginaldist.ppf(0.001), marginaldist.ppf(0.999) + 1)
                #     ax2.plot(marginaldist.pmf(x), x, 'o', linestyle='', color="maroon", markersize=2, label='$N_T$ pmf')
                #     ax2.axhline(y=marginaldist.mean(), linestyle='--', lw=1.75, label='$E[N_T]$')
                #     ax2.legend()

                plt.setp(ax2.get_yticklabels(), visible=False)
                ax2.set_title('$N_T$')

                for i in range(N):
                    # counts = np.arange(0, len(paths[i]))
                    ax1.step(times, paths[i], color=cm(colors[i]), lw=0.75, where='post')

                # if expectations is not None:
                #     ax1.plot(times, expectations, '--', lw=1.75, label='$E[N_t]$')
                #     ax1.legend()
                # if envelope:
                #     ax1.fill_between(times, upper, lower, alpha=0.25, color='grey')
                # plt.subplots_adjust(wspace=0.2, hspace=0.5)

            else:
                fig, ax1 = plt.subplots(**fig_kw)
                if colorspos:
                    colors = [path[colorspos] / np.max(np.abs(path)) for path in paths]
                else:
                    _, bins = np.histogram(last_points, n_bins)
                    my_bins = pd.cut(last_points, bins=bins, labels=range(len(bins) - 1), include_lowest=True)
                    colors = [col[b] for b in my_bins]

                for i in range(N):
                    # counts = np.arange(0, len(paths[i]))
                    ax1.step(times, paths[i], color=cm(colors[i]), lw=0.75, where='post')

                # if expectations is not None:
                #     ax1.plot(times, expectations, '--', lw=1.75, label='$E[N_t]$')
                #     ax1.legend()
                # if envelope:
                #     ax1.fill_between(times, upper, lower, color='grey', alpha=0.25)

            fig.suptitle(title)
            # ax1.set_xlim(right=T)
            ax1.set_title(r'Simulated Paths $X_t, t \leq T$')  # Title
            ax1.set_xlabel('$t$')
            ax1.set_ylabel('$X(t)$')
            plt.show()

        return fig

# mu = 1.5
# p = Galton_Watson(mu=mu)
#
# # sim = p.simulate(N=5, generations=10)
# p.plot(N=5, generations=10)
# n = 10
# sample = p.sample(generations=n)
# t = np.linspace(0.0, n, n + 1)
# plt.step(t, sample, where='post')
# plt.show()
