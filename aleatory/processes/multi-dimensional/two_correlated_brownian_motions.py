"""
Brownian Motion 2-dimensional
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from aleatory.processes.base import StochasticProcess
from aleatory.utils.utils import get_times


class BM2D(StochasticProcess):

    def __init__(self, rho=0.0, T=1.0):
        super().__init__()
        self.rho = rho
        self.T = T
        self.name = "Correlated Brownian Motions"
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

        self.times = get_times(self.T, self.n)
        W = self._sample(n)

        return W

    def simulate(self, n, N):
        sim = [self._sample(n) for _ in range(N)]
        self.times = get_times(self.T, self.n)

        return sim

    def plot(self, n, N, title=None, style="seaborn-v0_8-whitegrid", **fig_kw):

        sim = self.simulate(n, N)
        chart_title = title if title is not None else self.name

        with plt.style.context(style):
            col1 = "cyan"
            col2 = "darkgrey"
            fig, ax = plt.subplots(**fig_kw)
            for sample in sim:
                W1, W2 = sample
                ax.plot(self.times, W1, label="W1", color=col1)
                ax.plot(self.times, W2, label="W2", color=col2)
            ax.set_title(chart_title)
            ax.set_ylabel("W1(t), W2(t)")
            ax.set_xlabel("t")
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
            ax.set_ylabel("W1(t), W2(t)")
            ax.set_xlabel("t")
            plt.show()
        return fig

    def plot_sample_2d(self, n, title=None, color_by="time"):

        cmap = plt.get_cmap("cool")
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

        # for colors, label_title in zip([colors_indices], [label_title]):
        # Plotting the 2D Brownian motion
        plt.figure(figsize=(10, 8))
        # plt.plot(x, y, lw=1.5, color='blue')
        # Add the colored line segments between consecutive points
        for i in range(len(x) - 1):
            plt.plot(x[i : i + 2], y[i : i + 2], color=colors_indices[i], lw=1.5)

        plt.scatter(x[0], y[0], color="green", label="Start", zorder=5)
        plt.scatter(x[-1], y[-1], color="maroon", label="End", zorder=5)

        # plt.colorbar(label=label_title)
        plt.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            label=label_title,
            ax=plt.gca(),
        )

        plt.title(chart_title)
        plt.xlabel("X position")
        plt.ylabel("Y position")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        plt.show()


# p = BM2D(rho=-0.8)
# p.plot_sample(n=200, figsize=(12, 8))
# p.plot(n=200, N=5, figsize=(12, 8))
# # p.plot_sample_2d(n=500)
# p.plot_sample_2d(n=500, color_by="distance")
