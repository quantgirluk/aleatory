from aleatory.processes.base_analytical import SPAnalytical
from scipy.stats import invgauss
from aleatory.utils.utils import get_times
import numpy as np


class InverseGaussian(SPAnalytical):
    r"""
    Inverse Gaussian process
    ========================

    .. image:: ../_static/inverse_gaussian_process_draw.png

    Notes
    -----

    An Inverse Gaussian process is a stochastic process :math:`\{X(t),t\geq 0\}` where:

    - :math:`X(0)=0`,
    - the increments :math:`X(t)−X(s)` (for all :math:`t>s`) are independent and follow the Inverse Gaussian distribution with mean :math:`\mu(t-s)` and scale parameter :math:`\eta`.

    Constructor, Methods, and Attributes
    ------------------------------------

    """

    def __init__(self, mu=1.0, scale=1.0, T=1.0, rng=None):
        """
        :parameter double mu: the :math:`\mu>0` which defines the mean of the increments of the Inverse Gaussian process.
        :parameter double scale: the :math:`\eta>0` which defines the scale of the increments of the Inverse Gaussian process
        :parameter numpy.random.Generator rng: a custom random number generator
        """
        super().__init__(T=T, rng=rng)
        self.mu = mu
        self.scale = scale
        self.mean = lambda x: self.mu * x
        self.name = f"Inverse Gaussian process $X(\mu={self.mu}, \eta={self.scale})$"
        self.n = None
        self._ms = None

    def _check_mean(self, left, right):
        """Check the validity of the mean function."""
        delta = self.mean(right) - self.mean(left)
        if delta <= 0:
            raise ValueError("Mean function must be monotonically increasing.")
        return delta

    def _sample_inverse_gaussian_process(self, n):
        """
        Generate an inverse Gaussian process realization with n increments.
        """
        self.n = n
        self.times = get_times(self.T, self.n)
        delta_t = 1.0 * self.T / self.n
        lam = self.scale * (self.mean(delta_t) ** 2)
        nu = self.mean(delta_t)
        ign = invgauss.rvs(mu=nu / lam, loc=0.0, scale=lam, size=self.n - 1)
        ig = np.array(ign).cumsum()
        ig = np.insert(ig, [0], 0)
        return ig

    def sample(self, n):
        return self._sample_inverse_gaussian_process(n)

    def get_marginal(self, t):

        lam = self.scale * (self.mean(t) ** 2)
        nu = self.mean(t)

        marginal = invgauss(
            mu=nu / lam,
            loc=0.0,
            scale=lam,
        )
        return marginal

    def _process_expectation(self, times=None):
        if times is None:
            times = self.times
        return [self.mean(t) for t in times]

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

    p1 = InverseGaussian(T=3.0)
    p2 = InverseGaussian(mu=2.0, scale=2.0, T=3.0)
    p3 = InverseGaussian(scale=3.0, T=3.0)
    p4 = InverseGaussian(scale=0.5, T=3.0)

    for p, cm in [
        (p1, "Spectral"),
        (p2, "Purples"),
        (p3, "Oranges"),
        (p4, "Blues"),
    ]:

        p.draw(
            n=300,
            N=500,
            figsize=(12, 7),
            style=qs,
            colormap=cm,
            # orientation="vertical",
            # mode="steps+points",
            envelope=False,
        )
