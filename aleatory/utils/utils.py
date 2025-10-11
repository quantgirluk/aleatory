from numbers import Number

import numpy as np
from scipy.stats import ncx2
from aleatory.stats import ncx
import math

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import to_rgba


def get_times(end, n):
    """Generate a linspace from 0 to end for n increments."""
    return np.linspace(0, end, n)


def check_positive_integer(n, name=""):
    """Ensure that the number is a positive integer."""
    if not isinstance(n, int):
        raise TypeError(f"{name} must be an integer.")
    if n <= 0:
        raise ValueError(f"{name} must be positive.")


def check_numeric(value, name=""):
    """Ensure that the value is numeric."""
    if not isinstance(value, Number):
        raise TypeError(f"{name} value must be a number.")


def check_positive_number(value, name=""):
    """Ensure that the value is a positive number."""
    check_numeric(value, name)
    if value <= 0:
        raise ValueError(f"{name} value must be positive.")


def check_increments(times):
    increments = np.diff(times)
    if np.any([t < 0 for t in times]):
        raise ValueError("Times must be non-negative.")
    if np.any([t <= 0 for t in increments]):
        raise ValueError("Times must be strictly increasing.")
    return increments


def times_to_increments(times):
    """Ensure a positive, monotonically increasing sequence."""
    return check_increments(times)


def sample_besselq_global(T, initial, dim, n):
    t_size = T / n
    path = np.zeros(n)
    path[0] = initial
    x = initial
    for t in range(n - 1):
        sample = ncx2(df=dim, nc=x / t_size, scale=t_size).rvs(1)[0]
        path[t + 1] = sample
        x = sample

    return path


def sample_bessel_global(T, initial, dim, n):
    t_size = T / n
    tss = math.sqrt(t_size)
    path = np.zeros(n)
    path[0] = initial
    x = initial
    for t in range(n - 1):
        sample = ncx(df=dim, nc=x / tss, scale=tss).rvs(1)[0]
        path[t + 1] = sample
        x = sample

    return path


class ProcessDistributionPlotter:
    def __init__(self, process, n_steps=200, n_paths=1000):
        self.process = process
        self.t0, self.t1 = 0.0, float(process.T)
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.times = np.linspace(self.t0, self.t1, self.n_steps)
        self.paths = None

    def draw_dist(
        self,
        cmap=plt.cm.Blues_r,
        quantiles=(
            0.01,
            0.05,
            0.15,
            0.25,
            0.4,
            0.45,
            0.55,
            0.6,
            0.75,
            0.85,
            0.95,
            0.99,
        ),
    ):
        self.paths = self.process.simulate(self.n_steps, self.n_paths)
        all_vals = np.vstack(self.paths)

        x_grid = np.linspace(np.min(all_vals), np.max(all_vals), 200)
        dens_mat = np.zeros((self.n_steps, 200))
        for i in range(self.n_steps):
            dens_i = self.__estimate_density(all_vals[:, i], x_grid)
            dens_mat[i, :] = dens_i
        T, X = np.meshgrid(self.times, x_grid)
        dens_plot = dens_mat.T
        quantile_curves = {
            q: [np.quantile(all_vals[:, i], q) for i in range(self.n_steps)]
            for q in quantiles
        }
        plt.figure(figsize=(6, 5))
        plt.pcolormesh(T, X, dens_plot, shading="auto", cmap=cmap, alpha=1)
        colors = cmap(np.linspace(0.3, 0.8, len(quantiles) // 2))

        for i in range(len(quantiles) // 2):
            q_low = quantiles[i]
            q_high = quantiles[-(i + 1)]
            y_low = quantile_curves[q_low]
            y_high = quantile_curves[q_high]
            plt.fill_between(
                self.times, y_low, y_high, color=to_rgba(colors[i], alpha=1), zorder=3
            )
            plt.plot(self.times, y_low, color="black", lw=2)
            plt.plot(self.times, y_high, color="black", lw=2)

        plt.plot(
            self.times,
            np.mean(all_vals, axis=0),
            color="red",
            linestyle="--",
            lw=2,
            label=r"$\mathbb{E}[X_t]$",
            zorder=6,
        )

        plt.xlabel(r"$t$", fontsize=13)
        plt.ylabel(r"$f_{X_t}(y)$", fontsize=13)
        plt.grid(True, color="gray", linestyle=":")
        plt.legend(loc="upper left", frameon=True, fontsize=13)
        plt.tight_layout()
        plt.show()

    def __estimate_density(self, data_at_t, x_grid):
        data = np.array(data_at_t, copy=True)
        if np.std(data) < 1e-10:
            mu = np.mean(data)
            return np.exp(-0.5 * ((x_grid - mu) / 1e-6) ** 2)
        else:
            kde = gaussian_kde(data)
            return kde(x_grid)
