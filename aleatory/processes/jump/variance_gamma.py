import numpy as np

from aleatory.processes.base_analytical import SPAnalytical
from aleatory.utils.utils import (
    check_positive_integer,
    get_times,
)
from aleatory.stats.variance_gamma import vg


class VarianceGammaProcess(SPAnalytical):
    r"""
    Variance Gamma Process
    ======================

    .. image:: ../_static/variance_gamma_draw.png

    Notes
    -----

    In the theory of stochastic processes, a part of the mathematical theory of probability, the variance gamma
    (VG) process, also known as Laplace motion, is a Lévy process determined by a random time change. The process has
    finite moments, distinguishing it from many Lévy processes. There is no diffusion component in the VG process and
    it is thus a pure jump process. The increments are independent and follow a variance-gamma distribution,
    which is a generalization of the Laplace distribution. There are several representations of the Variance-Gamma
    process that relate it to other processes. It can for example be written as a Brownian motion :math:`W` with
    drift  :math:`\theta t` subjected to a random time change which follows Gamma process :math:`\Gamma(t;1 ,\nu)`.
    That is,

    .. math::

        X(t; \sigma, \nu, \theta) = \theta \Gamma(t;1, \nu) + \sigma W(\Gamma(t;1, \nu)); \quad t \in [0,T].



    Constructor, Methods, and Attributes
    ------------------------------------

    """

    def __init__(self, theta=0.0, nu=1.0, sigma=1.0, T=1.0, rng=None):
        """
        :parameter double theta: the :math:`\theta` parameter in the above expression
        :parameter double nu: the :math:`\nu` parameter in the above expression
        :parameter double sigma: the :math:`\sigma` parameter in the above expression
        :parameter float T: the right hand endpoint of the time interval :math:`[0,T]` for the process
        :parameter numpy.random.Generator rng: a custom random number generator
        """
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
        check_positive_integer(n)
        dt = self.T / n
        gammas = self.rng.gamma(shape=dt / self.nu, scale=self.nu, size=n - 1)
        gaussian_increments = self.rng.normal(loc=0.0, scale=1.0, size=n - 1)
        increments = (
            self.theta * gammas + self.sigma * np.sqrt(gammas) * gaussian_increments
        )
        sample = np.cumsum(increments)
        sample = np.concatenate(([0], sample))
        return sample

    def sample(self, n):
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

    def get_marginal(self, t):

        a = 2.0 * t / self.nu
        b = 0.5 * self.theta * self.nu
        c = self.sigma * np.sqrt(0.5 * self.nu)
        marginal = vg(r=a, theta=b, sigma=c)
        return marginal

    def plot(self, n, N, mode="steps", title=None, **fig_kw):
        return self._plot_process(n=n, N=N, mode=mode, title=title, **fig_kw)

    def draw(
        self,
        n,
        N,
        mode="steps",
        marginal=True,
        envelope=False,
        empirical_envelope=True,
        title=None,
        **fig_kw,
    ):
        return self._draw_qqstyle(
            n,
            N,
            marginal=marginal,
            mode=mode,
            envelope=envelope,
            title=title,
            empirical_envelope=empirical_envelope,
            estimate_quantiles=True,
            **fig_kw,
        )


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    qs = "https://raw.githubusercontent.com/quantgirluk/matplotlib-stylesheets/main/quant-pastel-light.mplstyle"
    plt.style.use(qs)

    #     for T in [5.0, 10.0, 20.0]:
    #         p = VarianceGammaProcess(T=T, theta=0.0, sigma=1.0)
    #
    #         for n_grid in [500]:
    #             p.draw(n=n_grid, N=200, figsize=(12, 7), style=qs, envelope=True)
    # p.draw(n=n_grid, N=200, figsize=(12, 7), style=qs, envelope=True)
    #
    # end_points = [path[-1] for path in p.paths]
    # plt.hist(end_points, density=True)
    # plt.show()
    # vals = p.marginal_density()

    p = VarianceGammaProcess()
    # p.plot(n=100, N=5, figsize=(10, 6), style=qs)
    p.plot(
        n=50,
        N=5,
        figsize=(10, 6),
        style=qs,
        mode="steps+points",
        title="Variance Gamma Process Paths",
    )
    p.draw(
        n=50,
        N=100,
        figsize=(12, 7),
        style=qs,
        colormap="viridis",
        marginal=False,
        mode="steps+points",
    )
    p.draw(n=50, N=100, figsize=(12, 7), style=qs, colormap="cool")
    p.plot(n=50, N=5, figsize=(10, 6), style=qs)
    p = VarianceGammaProcess(T=100)
    p.draw(n=100, N=100, figsize=(12, 7), style=qs, colormap="viridis")
    p = VarianceGammaProcess(theta=1.5, nu=0.5, sigma=2.0, T=20)
    p.draw(n=100, N=200, figsize=(12, 7), style=qs, colormap="twilight")
    p = VarianceGammaProcess(theta=0.0, nu=0.5, sigma=1.0, T=1.0)
    p.draw(n=100, N=200, figsize=(12, 7), style=qs, colormap="winter", envelope=False)
    p = VarianceGammaProcess(theta=-1.0, nu=4.0, sigma=2.0, T=100.0)
    p.draw(n=100, N=200, figsize=(12, 7), style=qs, colormap="summer", envelope=False)
