"""
Gamma Process
"""

import numpy as np
from scipy.stats import gamma
from aleatory.utils.utils import check_positive_number, get_times

from aleatory.processes.base_analytical import SPAnalytical
from aleatory.processes.analytical.increments import GammaIncrements


class GammaProcess(SPAnalytical):
    r"""
    Gamma process
    =============

    .. image:: ../_static/gamma_process_draw.png

    Notes
    -----

    The gamma process :math:`X = \{ X(t; \mu,\nu) : t \geq 0\}` with mean parameter
    :math:`\mu` and variance parameter :math:`\nu` is a continuous-time process with stationary,
    independent increments such that

    .. math::
        X(t + h; \mu, \nu)− X(t; \mu, \nu) \sim Gamma\left( \frac{\mu^2 h}{\nu}, \frac{\nu}{\mu} \right),

    for any :math:`h > 0`.

    Constructor, Methods, and Attributes
    ------------------------------------
    """

    def __init__(self, mu=1.0, nu=1.0, T=10.0, rng=None):
        """

        :parameter float mu: the parameter :math:`\mu` in the above definition
        :parameter float nu: the parameter :math:`\nu` in the above definition

        """
        super().__init__(T=T, rng=rng, initial=0.0)
        self.mu = mu
        self.nu = nu
        self.name = f"Gamma Process X($\\mu$={self.mu}, $\\nu$={self.nu})"
        self.n = None
        self.times = None
        self.shape = self.mu**2 / self.nu
        self.scale = self.nu / self.mu
        self.rate = 1.0 / self.scale
        self.gamma_increments = GammaIncrements(
            k=self.shape, theta=self.scale, T=self.T, rng=self.rng
        )

    def __str__(self):
        return f"Gamma Process with parameters mu = {self.mu} and nu = {self.nu}"

    def __repr__(self):
        return f"GammaProcess(mu={self.mu}, nu={self.nu})"

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        check_positive_number(value, "mu")
        self._mu = value

    @property
    def nu(self):
        return self._nu

    @nu.setter
    def nu(self, value):
        check_positive_number(value, "nu")
        self._nu = value

    def _sample_gamma_process(self, n):
        self.n = n
        self.times = get_times(self.T, self.n)
        increments = np.cumsum(self.gamma_increments.sample(n - 1))
        increments = np.insert(increments, 0, [0])

        path = np.full(n, self.initial) + increments
        return path

    def _sample_gamma_process_at(self, times):
        if times[0] != 0:
            times = np.insert(times, 0, [0])
        self.times = times

        increments = np.cumsum(self.gamma_increments.sample_at(times))
        path = np.full(len(self.times), self.initial) + increments

        return path

    def sample(self, n):
        """
        Generates a discrete time sample from a Gamma process instance.

        :param int n: the number of steps
        :return: numpy array
        """
        return self._sample_gamma_process(n)

    def sample_at(self, times):
        """
        Generates a sample from a Gamma process at the specified times.

        :param times: the times which define the sample
        :return: numpy array
        """
        return self._sample_gamma_process_at(times)

    def get_marginal(self, t):
        marginal = gamma(a=self.shape * t, scale=self.scale)
        return marginal

    def _process_expectation(self, times=None):
        if times is None:
            times = self.times
        return self.mu * times

    def _process_variance(self, times=None):
        if times is None:
            times = self.times
        return self.nu * times

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
        return self._plot_process(n=n, N=N, mode=mode, title=title, **fig_kw)

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
    p = GammaProcess()
    p.plot(n=100, N=5, figsize=(12, 8), style=qs)
    p.draw(n=100, N=200, figsize=(12, 8), style=qs)

    # p.draw(n=100, N=200, figsize=(12, 8), style=qs)

#     exps = p.process_expectation()
#     vars = p.process_variance()
#     p.plot(
#         n=100,
#         N=5,
#         figsize=(12, 8),
#         style=qs,
#         title=f"5 Paths from a Gamma Process on $[0,10]$",
#     )
#     p.draw(n=100, N=200, figsize=(12, 8), style=qs)
#     p = GammaProcess(mu=1.5, nu=0.5, T=10)
#     p.draw(n=100, N=200, figsize=(12, 8), style=qs, colormap="twilight")
#     p = GammaProcess(mu=2.0, nu=4.0, T=10)
#     p.draw(n=100, N=200, envelope=True, figsize=(12, 8), style=qs, colormap="winter")
