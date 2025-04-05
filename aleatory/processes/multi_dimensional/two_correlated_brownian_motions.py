"""
Correlated Brownian Motions
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from aleatory.processes.base import StochasticProcess
from aleatory.utils.utils import get_times
from aleatory.utils.plotters_2d import plot_paths_coordinates


class CorrelatedBMs(StochasticProcess):

    def __init__(self, rho=0.0, T=1.0):
        super().__init__()
        self.rho = rho
        self.T = T
        self.name = f"Correlated Brownian Motions $\\rho={self.rho}$"
        self.n = None
        self.times = None

    def _sample(self, n):

        Z1 = np.random.randn(n - 1)
        Z2 = np.random.randn(n - 1)
        dt = self.T / n
        # Apply Cholesky decomposition to get correlated Brownian increments
        dW1 = Z1 * np.sqrt(dt)
        dW2 = (self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2) * np.sqrt(dt)

        # Compute the Brownian motion paths
        W1 = np.insert(np.cumsum(dW1), 0, 0)  # Cumulative sum to get W1(t)
        W2 = np.insert(np.cumsum(dW2), 0, 0)  # Cumulative sum to get W2(t)
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

    def plot_sample(
        self,
        n,
        coordinates=False,
        title=None,
        style="seaborn-v0_8-whitegrid",
        mode="linear",
        **fig_kw,
    ):
        if coordinates:
            fig = self.plot_sample_coordinates(
                n=n, title=title, style=style, mode=mode, **fig_kw
            )
        else:
            fig = self.plot_sample_2d(n=n, title=title, style=style, **fig_kw)

        return fig

    def plot_sample_coordinates(
        self, n, title=None, style="seaborn-v0_8-whitegrid", mode="linear", **fig_kw
    ):
        chart_title = title if title is not None else self.name
        X, Y = self.sample(n)
        times = self.times
        fig = plot_paths_coordinates(
            times=times,
            paths1=[X],
            paths2=[Y],
            style=style,
            title=chart_title,
            mode=mode,
            **fig_kw,
        )
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
        chart_title = title if title else f"{self.name} Sample Path"
        self.n = n
        x, y = self.sample(n)

        if color_by == "time":
            # Normalize time indices for coloring
            time_indices = self.times
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
            ax.set_ylabel("$X_2$(t)")
            ax.set_xlabel("$X_1(t)$")
            ax.legend()
            ax.grid(True)
            ax.axis("equal")
            plt.show()
        return fig
