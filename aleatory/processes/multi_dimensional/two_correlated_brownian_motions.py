"""
Correlated Brownian Motions
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from aleatory.processes.base import StochasticProcess
from aleatory.utils.utils import get_times


class CorrelatedBMs(StochasticProcess):

    def __init__(self, rho=0.0, T=1.0):
        super().__init__()
        self.rho = rho
        self.T = T
        self.name = f"Correlated Brownian Motions $\\rho={self.rho}$"
        self.n = None
        self.times = None

    def _sample(self, n):

        Z1 = np.random.randn(n)
        Z2 = np.random.randn(n)
        dt = self.T / n
        # Apply Cholesky decomposition to get correlated Brownian increments
        dW1 = Z1 * np.sqrt(dt)
        dW2 = (self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2) * np.sqrt(dt)

        # Compute the Brownian motion paths
        W1 = np.cumsum(dW1)  # Cumulative sum to get W1(t)
        W2 = np.cumsum(dW2)  # Cumulative sum to get W2(t)
        W = (W1, W2)

        return W

    def sample(self, n):

        self.n = n
        self.times = get_times(self.T, self.n)
        W = self._sample(n)

        return W

    def simulate(self, n, N):
        sim = [self._sample(n) for _ in range(N)]
        self.times = get_times(self.T, self.n)

        return sim

    def plot(
        self, n, N, title=None, style="seaborn-v0_8-whitegrid", colors=None, **fig_kw
    ):

        sim = self.simulate(n, N)
        chart_title = title if title is not None else self.name
        if colors:
            col1, col2 = colors[0], colors[1]
        else:
            col1, col2 = "#0079ff", "#ffb84c"
        with plt.style.context(style):
            fig, ax = plt.subplots(**fig_kw)

            W1, W2 = sim[0]
            ax.plot(self.times, W1, label="W1", color=col1)
            ax.plot(self.times, W2, label="W2", color=col2)
            for sample in sim[1:]:
                W1, W2 = sample
                ax.plot(self.times, W1, color=col1)
                ax.plot(self.times, W2, color=col2)
            ax.set_title(chart_title)
            ax.set_xlabel("t")
            ax.legend(loc="best")
            plt.show()

        return fig

    def plot_sample(self, n, title=None, style="seaborn-v0_8-whitegrid", **fig_kw):

        chart_title = title if title is not None else self.name
        self.n = n
        W1, W2 = self.sample(n)
        times = self.times

        with plt.style.context(style):
            fig, ax = plt.subplots(**fig_kw)
            ax.plot(times, W1, label="W1")
            ax.plot(times, W2, label="W2")
            ax.set_title(chart_title)
            ax.set_xlabel("t")
            ax.legend(loc="best")
            plt.show()
        return fig

    def plot_sample_2d(
        self,
        n,
        title=None,
        color_by="time",
        style="seaborn-v0_8-whitegrid",
        color_map="cool",
        **fig_kw,
    ):

        cmap = plt.get_cmap(color_map)
        chart_title = title if title else self.name
        self.n = n
        x, y = self.sample(n)

        if color_by == "time":
            # Normalize time indices for coloring
            time_indices = p.times
            norm = Normalize(vmin=time_indices.min(), vmax=time_indices.max())
            colors_indices = cmap(norm(time_indices))
            label_title = "Time"
        elif color_by == "distance":
            # Compute distances from the origin and create color mapping based on distance
            distances = np.sqrt(x**2 + y**2)
            norm = Normalize(vmin=distances.min(), vmax=distances.max())
            colors_indices = cmap(norm(distances))
            label_title = "Distance to origin"
        else:
            raise ValueError("color_by must be either 'time' or 'distance'")

        with plt.style.context(style):
            fig, ax = plt.subplots(**fig_kw)
            ax.scatter(x[0], y[0], color="green", label="Start", zorder=5)
            ax.scatter(x[-1], y[-1], color="maroon", label="End", zorder=5)

            # Plotting the 2D Brownian motion
            # Add the colored line segments between consecutive points
            for i in range(len(x) - 1):
                ax.plot(x[i : i + 2], y[i : i + 2], color=colors_indices[i], lw=1.5)

            fig.colorbar(
                ScalarMappable(norm=norm, cmap=cmap),
                label=label_title,
                ax=plt.gca(),
            )
            ax.set_title(chart_title)
            ax.set_ylabel("$W_2$(t)")
            ax.set_xlabel("$W_1(t)$")
            ax.legend()
            ax.grid(True)
            ax.axis("equal")
            plt.show()
        return fig


# p = CorrelatedBMs(rho=-0.5)
# p.plot_sample(n=500, figsize=(12, 8))
# p.plot(n=500, N=5, figsize=(12, 8))
# p.plot_sample_2d(n=2000, figsize=(12, 10))
# p.plot_sample_2d(n=2000, color_by="distance", figsize=(12, 10))
