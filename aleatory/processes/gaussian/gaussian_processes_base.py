"""Gaussian processes"""

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interactive


from scipy.stats import norm

from aleatory.processes.base_analytical import SPAnalyticalMarginals
from aleatory.utils.plotters_covariances import (
    plot_covariance_matrix,
    plot_paths_and_kernel,
    plot_kernel3d,
)


def check_positive_definite(matrix):
    """Check if a matrix is positive definite"""
    eigenvalues = np.linalg.eigvals(matrix)
    return np.all(eigenvalues > 0)


class GaussianProcess(SPAnalyticalMarginals):
    """
    A Gaussian Process is a collection of random variables, for which any finite number of which have a joint Gaussian distribution.
    It is fully specified by its mean function and covariance function (kernel).
    """

    def __init__(
        self,
        mean_function,
        covariance_function,
        variance_function=None,
        T=1.0,
        rng=None,
    ):
        super().__init__(T=T, rng=rng)
        self.mean = mean_function
        self.covariance = covariance_function
        self.variance = variance_function
        self.kernel = covariance_function
        self.name = "GaussianProcess"
        self.short_name = "GP"
        self.paths = None
        self.times = None

    def sample_at(self, times):
        return self._sample_at(times)

    def _sample_at(self, times):
        mean_evaluated = self.mean(times)
        covariance_evaluated = self.covariance(times)
        return np.random.default_rng(self.rng).multivariate_normal(
            mean_evaluated, covariance_evaluated
        )

    def sample(self, n, T=None):
        if T is None:
            T = self.T
        times = np.linspace(0, T, n)
        return self.sample_at(times)

    def simulate(self, n, N, T=None):
        if T is None:
            T = self.T
        times = np.linspace(0, T, n)
        paths = np.random.default_rng(self.rng).multivariate_normal(
            self.mean(times), self.covariance(times), size=N
        )
        self.times = times
        self.paths = paths
        return paths

    def _process_expectation(self, times=None):
        if times is None:
            times = self.times
        return self.mean(times)

    def _process_variance(self, times=None):
        if times is None:
            times = self.times
        if self.variance is not None:
            return np.asarray(self.variance(times))
        return np.diag(self.covariance(times))

    def _process_stds(self, times=None):
        if times is None:
            times = self.times
        return np.sqrt(self._process_variance(times))

    def get_marginal(self, time):
        expectation = self.mean(time)
        if self.variance is not None:
            variance = np.asarray(self.variance(np.array([time]))).reshape(-1)[0]
        else:
            variance = self.covariance(np.array([time]))[0, 0]
        return norm(loc=expectation, scale=np.sqrt(variance))

    def draw(
        self,
        n,
        N,
        T=None,
        marginal=True,
        envelope=False,
        type="3sigma",
        title=None,
        **fig_kw,
    ):
        """
        Simulates and plots paths/trajectories from the instanced stochastic process.

        Produces different kind of visualisation illustrating the following elements:

        - times versus process values as lines
        - the expectation of the process across time
        - histogram showing the empirical marginal distribution :math:`X_T` (optional when ``marginal = True``)
        - probability density function of the marginal distribution :math:`X_T` (optional when ``marginal = True``)
        - envelope of confidence intervals across time (optional when ``envelope = True``)

        :param int n: number of steps in each path
        :param int N: number of paths to simulate
        :param float T: the endpoint of the time interval [0,T] over which the process is defined. If not passed, it defaults to the value of T passed in the constructor.
        :param bool marginal:  defaults to True
        :param bool envelope:   defaults to False
        :param str type:   defaults to  '3sigma'
        :param str title:  to be used to customise plot title. If not passed, the title defaults to the name of the process.
        :return:
        """

        if type == "3sigma":
            return self._draw_3sigmastyle(
                n=n,
                N=N,
                T=T,
                marginal=marginal,
                envelope=envelope,
                title=title,
                **fig_kw,
            )
        elif type == "qq":
            return self._draw_qqstyle(
                n, N, T=T, marginal=marginal, envelope=envelope, title=title, **fig_kw
            )
        else:
            raise ValueError

    def plot_mean_variance(self, times=None, **fig_kw):
        return super()._plot_mean_variance(times, process_label="B", **fig_kw)

    def plot_covariance(
        self,
        times=None,
        colormap="coolwarm",
        matrix_shape=True,
        title=None,
        **fig_kw,
    ):
        if times is None:
            times = np.linspace(0, self.T, 100)
        covariance_matrix = self.covariance(times)
        title = title if title else f"{self.name} \nCovariance Matrix"

        fig = plot_covariance_matrix(
            times,
            covariance_matrix,
            colormap=colormap,
            matrix_shape=matrix_shape,
            title=title,
            **fig_kw,
        )

        return fig

    def plot_kernel(
        self,
        times=None,
        colormap="coolwarm",
        matrix_shape=False,
        title=None,
        cbar_labels={"cbar": "Kernel K(t, s)"},
    ):
        if title is None:
            title = f"{self.name} \nKernel Function"
        return self.plot_covariance(
            times,
            colormap=colormap,
            matrix_shape=matrix_shape,
            title=title,
            cbar_labels=cbar_labels,
        )

    def plot_kernel3d(self, times=None, title=None, **fig_kw):
        if times is None:
            npoints = int(100 * self.T)
            times = np.linspace(0, self.T, npoints)
        K = self.covariance(times)

        style = fig_kw.pop("style", "seaborn-v0_8-whitegrid")
        fig = plot_kernel3d(times, K, title=title, style=style, **fig_kw)
        return fig

    def plot_mean_function(self, T=None, n=None):
        if T is None:
            T = self.T
        if n is None:
            n = int(100 * T)
        times = np.linspace(0, T, n)
        mean_values = self.mean(times)
        plt.plot(times, mean_values)
        plt.title("Mean Function")
        plt.xlabel("Time")
        plt.ylabel("Mean")
        plt.show()

    def plot_paths_and_kernel(
        self, n, N, T=None, cmap="coolwarm", matrix_shape=False, title=None
    ):
        if T is None:
            T = self.T
        paths = self.simulate(n, N, T)
        times = np.linspace(0, T, n)
        K = self.covariance(times)

        title = title if title else f"{self.name}"
        return plot_paths_and_kernel(
            paths,
            times,
            K,
            title=title,
            cmap=cmap,
            matrix_shape=matrix_shape,
        )


class GaussianSigma(GaussianProcess):

    def __init__(self, sigma=1.0, T=1.0, rng=None):
        super().__init__(
            mean_function=lambda t: np.zeros_like(t),
            covariance_function=self.covariance_function,
            variance_function=self.variance_function,
            T=T,
            rng=rng,
        )

        self.sigma = sigma
        self.name = f"Gaussian Process ($\\sigma$={sigma:.2f})"
        self.short_name = f"GP"

    def variance_function(self, times):
        covariance_matrix = self.covariance_function(times)
        return np.diag(covariance_matrix)

    def make_widget(self, matrix_shape=False, cmap="coolwarm"):

        sigma_slider = widgets.FloatSlider(
            value=self.sigma, min=0.25, max=3.0, step=0.25, description="Sigma"
        )
        n_samples_slider = widgets.IntSlider(
            value=5, min=2, max=10, step=1, description="Samples"
        )

        def update(sigma=self.sigma, n_samples=5):
            self.sigma = sigma
            n_steps = int(100 * self.T)
            self.plot_paths_and_kernel(
                n=n_steps,
                N=n_samples,
                T=self.T,
                title=f"GP ($\\sigma$={sigma:.2f})",
                cmap=cmap,
                matrix_shape=matrix_shape,
            )

        widget = interact(update, sigma=sigma_slider, n_samples=n_samples_slider)
        return widget


class GaussianLengthScaleSigma(GaussianProcess):

    def __init__(self, length_scale=1.0, sigma=1.0, T=1.0, rng=None):
        super().__init__(
            mean_function=lambda t: np.zeros_like(t),
            covariance_function=self.covariance_function,
            variance_function=self.variance_function,
            T=T,
            rng=rng,
        )
        self.length_scale = length_scale
        self.sigma = sigma
        self.name = f"Gaussian Process (l={length_scale:.2f}, $\\sigma$={sigma:.2f})"
        self.short_name = f"GP"

    def variance_function(self, times):
        covariance_matrix = self.covariance_function(times)
        return np.diag(covariance_matrix)

    def make_widget(self, matrix_shape=False, cmap="coolwarm"):
        length_slider = widgets.FloatSlider(
            value=self.length_scale,
            min=0.1,
            max=1.0,
            step=0.1,
            description="Length Scale",
        )
        sigma_slider = widgets.FloatSlider(
            value=self.sigma, min=0.25, max=3.0, step=0.25, description="Sigma"
        )
        n_samples_slider = widgets.IntSlider(
            value=5, min=2, max=10, step=1, description="Samples"
        )

        def update(length_scale, sigma, n_samples=5):
            self.length_scale = length_scale
            self.sigma = sigma
            nsteps = int(100 * self.T)
            self.plot_paths_and_kernel(
                n=nsteps,
                N=n_samples,
                T=self.T,
                title=f"{self.short_name} (l={length_scale:.2f}, $\\sigma$={sigma:.2f})",
                cmap=cmap,
                matrix_shape=matrix_shape,
            )

        widget = interact(
            update,
            length_scale=length_slider,
            sigma=sigma_slider,
            n_samples=n_samples_slider,
        )
        return widget


class GaussianThreeParameter(GaussianProcess):

    def __init__(self, length_scale=1.0, sigma=1.0, nu=1.5, T=1.0, rng=None):
        super().__init__(
            mean_function=lambda t: np.zeros_like(t),
            covariance_function=self.covariance_function,
            variance_function=self.variance_function,
            T=T,
            rng=rng,
        )
        self.length_scale = length_scale
        self.sigma = sigma
        self.nu = nu
        self.name = f"Gaussian Process (l={length_scale:.2f}, $\\sigma$={sigma:.2f}, $\\nu$={nu:.2f})"
        self.short_name = f"GP"

    def variance_function(self, times):
        covariance_matrix = self.covariance_function(times)
        return np.diag(covariance_matrix)

    def make_widget(self, matrix_shape=False, cmap="coolwarm"):
        length_slider = widgets.FloatSlider(
            value=self.length_scale,
            min=0.1,
            max=1.0,
            step=0.1,
            description="Length Scale",
        )
        sigma_slider = widgets.FloatSlider(
            value=self.sigma, min=0.25, max=3.0, step=0.25, description="Sigma"
        )
        nu_slider = widgets.FloatSlider(
            value=self.nu, min=0.5, max=2.5, step=0.5, description="Nu"
        )
        n_samples_slider = widgets.IntSlider(
            value=5, min=2, max=10, step=1, description="Samples"
        )

        def update(length_scale, sigma, nu, n_samples):
            self.length_scale = length_scale
            self.sigma = sigma
            self.nu = nu
            nsteps = int(100 * self.T)
            self.plot_paths_and_kernel(
                n=nsteps,
                N=n_samples,
                T=self.T,
                title=f"{self.short_name} (l={length_scale:.2f}, $\\sigma$={sigma:.2f}, $\\nu$={nu:.2f})",
                cmap=cmap,
                matrix_shape=matrix_shape,
            )

        widget = interact(
            update,
            length_scale=length_slider,
            sigma=sigma_slider,
            nu=nu_slider,
            n_samples=n_samples_slider,
        )
        return widget
