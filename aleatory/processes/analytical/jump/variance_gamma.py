import numpy as np

from aleatory.processes.base_analytical import SPAnalytical
from aleatory.utils.utils import (
    check_positive_integer,
    get_times,
)


class VarianceGammaProcess(SPAnalytical):

    def __init__(self, theta=0.0, nu=1.0, sigma=1.0, T=1.0, rng=None):
        super().__init__(T=T, rng=rng, initial=0.0)
        self.theta = theta
        self.nu = nu
        self.sigma = sigma
        self.name = f"Variance Gamma Process X($\\theta$={self.theta}, $\\nu$={self.nu}, $\\sigma$={self.sigma})"
        self.n = None
        self.times = None

    def __str__(self):
        return f"Variance Gamma Process with parameters  theta = {self.theta}, nu = {self.nu}, sigma = {self.sigma}"

    def __repr__(self):
        return f"VarianceGammaProcess(theta = {self.theta}, nu={self.nu}, sigma={self.sigma})"

    def _sample_variance_gamma_process(self, n):
        """Generate a realization of a Variance gamma process."""
        check_positive_integer(n)

        dt = self.T / n
        shape = dt / self.nu
        scale = self.nu

        gammas = self.rng.gamma(shape=shape, scale=scale, size=n - 1)
        gaussian_increments = self.rng.normal(loc=0, scale=1, size=n - 1)

        increments = (
            self.theta * gammas + self.sigma * np.sqrt(gammas) * gaussian_increments
        )
        sample = np.cumsum(increments)
        sample = np.concatenate(([0], sample))

        return sample

    def _sample_gamma_bridge_sampling(self, n):
        """
        Gamma Bridge Sampling for the Variance Gamma process.
        """
        pass
        # dt = self.T / n
        # times = self.times
        #
        # # Simulate terminal value of the process
        # G_T = np.random.gamma(shape=self.T / self.nu, scale=self.nu)
        # X_T = self.rng.normal(loc=self.theta * G_T, scale=(self.sigma**2) * G_T)
        #
        # # Initialize the Gamma process
        # gamma_process = np.zeros(n)
        # gamma_process[-1] = G_T
        #
        # # Bridge sampling
        # for i in range(1, n - 1):
        #     t_i = times[i]
        #     mean = (t_i / self.T) * G_T
        #     variance = (t_i * (self.T - t_i)) / self.T**2 * G_T / self.nu
        #     shape = mean**2 / variance
        #     scale = variance / mean
        #     gamma_process[i] = np.random.gamma(shape=shape, scale=scale)

        # return gamma_process

    def sample(self, n):
        """Generate a realization.

        :param int n: the number of increments to generate
        """

        self.n = n
        self.times = get_times(self.T, self.n)
        return self._sample_variance_gamma_process(n)

    def _process_expectation(self, times=None):
        if times is None:
            times = self.times
        expectations = self.theta * times
        return expectations

    def _process_variance(self, times=None):
        if times is None:
            times = self.times
        variances = ((self.theta**2) * self.nu + self.sigma**2) * times
        return variances

    def marginal_expectation(self, times=None):
        expectations = self._process_expectation(times=times)
        return expectations

    def marginal_variance(self, times=None):
        variances = self._process_variance(times=times)
        return variances

    def plot(self, n, N, mode="steps", title=None, **fig_kw):
        """
        Simulates and plots paths/trajectories from the instanced stochastic process.
        Simple plot of times versus process values as lines and/or markers.

        :parameter int n: number of steps in each path
        :parameter int N: number of paths to simulate
        :parameter str mode: defines the type of plot to produce
        :parameter str title: string to customise plot title
        :return:

        """
        self._plot_process(n=n, N=N, mode=mode, title=title, **fig_kw)

    def draw(
        self, n, N, marginal=True, envelope=False, mode="steps", title=None, **fig_kw
    ):
        """
        Simulates and plots paths/trajectories from the instanced stochastic process.
        Visualisation shows
        - times versus process values as lines
        - the expectation of the process across time
        - histogram showing the empirical marginal distribution :math:`X_T`
        - probability density function of the marginal distribution :math:`X_T`
        - envelope of confidence intervals

        :param int n: number of steps in each path
        :param int N: number of paths to simulate
        :param bool marginal: defaults to True
        :param bool envelope: defaults to False
        :param mode: defines the type of plot to produce (e.g. "steps", "points" or "steps+points")
        :param str title: string optional default to the name of the process
        :return:
        """
        return self._draw_qqstyle(
            n, N, marginal=marginal, mode=mode, envelope=envelope, title=title, **fig_kw
        )


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    qs = "https://raw.githubusercontent.com/quantgirluk/matplotlib-stylesheets/main/quant-pastel-light.mplstyle"
    plt.style.use(qs)

    p = VarianceGammaProcess(T=10.0, theta=0.0, sigma=1.0)

    for n_grid in [20, 200]:
        p.draw(n=n_grid, N=100, figsize=(12, 7), style=qs, colormap="cividis")

    #
    # p.plot(n=10, N=5, figsize=(10, 6), style=qs)
    # p.plot(
    #     n=20,
    #     N=5,
    #     figsize=(10, 6),
    #     style=qs,
    #     mode="steps+points",
    #     title="Variance Gamma Process Paths",
    # )
    # p.draw(
    #     n=100,
    #     N=100,
    #     figsize=(12, 7),
    #     style=qs,
    #     colormap="viridis",
    #     marginal=False,
    #     mode="steps+points",
    # )
    # p.draw(n=100, N=100, figsize=(12, 7), style=qs, colormap="cool")
    # p.plot(n=20, N=5, figsize=(10, 6), style=qs)
    # p = VarianceGammaProcess(T=100)
    # p.draw(n=100, N=100, figsize=(12, 7), style=qs, colormap="viridis")
    # p = VarianceGammaProcess(theta=1.5, nu=0.5, sigma=2.0, T=20)
    # p.draw(n=100, N=200, figsize=(12, 7), style=qs, colormap="twilight")
    # p = VarianceGammaProcess(theta=0.0, nu=0.5, sigma=1.0, T=1.0)
    # p.draw(n=100, N=200, figsize=(12, 7), style=qs, colormap="winter", envelope=True)
    # p = VarianceGammaProcess(theta=-1.0, nu=4.0, sigma=2.0, T=100.0)
    # p.draw(n=100, N=200, figsize=(12, 7), style=qs, colormap="summer", envelope=True)
