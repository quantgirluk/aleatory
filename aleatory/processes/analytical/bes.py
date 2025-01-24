"""
Bessel Process BES
"""

import math
from functools import partial
from multiprocessing import Pool

import numpy as np
from scipy.special import eval_genlaguerre

from aleatory.processes.analytical.brownian_motion import BrownianMotion
from aleatory.processes.base_analytical import SPAnalytical
from aleatory.stats import ncx
from aleatory.utils.utils import check_positive_integer, get_times, sample_bessel_global


def _sample_bessel_global(T, initial, dim, n):
    path = sample_bessel_global(T=T, initial=initial, dim=dim, n=n)
    return path


class BESProcess(SPAnalytical):
    r"""
    Bessel process
    ==============

    .. image:: ../_static/bes_process_drawn.png

    Notes
    -----

    A Bessel process :math:`BES^{n}_{0},` for :math:`n\geq 2` integer is a continuous stochastic process
    :math:`\{X(t) : t \geq  0\}` characterised as the Euclidian norm of an :math:`n`-dimensional
    Brownian motion. That is,

    .. math::
        X_t = \sqrt{\sum_{i=1}^n (W^i_t)^2}.


    More generally, for any :math:`\delta >0`, and :math:`x_0 \geq 0`, a Bessel process of dimension :math:`\delta`
    starting at :math:`x_0`, denoted by

    .. math::
        BES_{{x_0}}^{{\delta}}

    can be defined by the following SDE

    .. math::
        dX_t = \frac{(\delta-1)}{2}  \frac{dt}{X_t} + dW_t \ \ \ \ t\in (0,T]


    with initial condition :math:`X_0 = x_0\geq 0.`,  where

    - :math:`\delta` is a positive real
    - :math:`W_t` is a standard one-dimensional Brownian Motion.

    Constructor, Methods, and Attributes
    ------------------------------------

    """

    def __init__(self, dim=1.0, initial=0.0, T=1.0, rng=None):
        """
        :param double dim: the dimension of the process :math:`n`
        :param double initial: the initial point of the process :math:`x_0`
        :param double T: the right hand endpoint of the time interval :math:`[0,T]`
            for the process
        :param numpy.random.Generator rng: a custom random number generator
        """
        super().__init__(T=T, rng=rng, initial=initial)
        self.dim = dim
        self._brownian_motion = BrownianMotion(T=T, rng=rng)
        self.name = f"Bessel process $BES^{{{self.dim}}}_{{{self.initial}}}$"
        self.n = None
        self.times = None

    def __str__(self):
        return "Bessel process with dimension {dim} and starting condition {initial} on [0, {T}].".format(
            T=str(self.T), dim=str(self.dim), initial=str(self.initial)
        )

    def __repr__(self):
        return "BESProcess(dimension={dim}, initial={initial}, T={T})".format(
            T=str(self.T), dim=str(self.dim), initial=str(self.initial)
        )

    @property
    def dim(self):
        """Bessel Process dimension."""
        return self._dim

    @dim.setter
    def dim(self, value):
        if value < 0:
            raise TypeError("Dimension must be positive")
        self._dim = value

    @property
    def initial(self):
        """Bessel Process initial point."""
        return self._initial

    @initial.setter
    def initial(self, value):
        if value < 0:
            raise TypeError("Initial point must be positive")
        self._initial = value

    def _sample_bessel_integer_dim_zero(self, n):
        check_positive_integer(n)

        self.n = n
        self.times = get_times(self.T, n)
        brownian_samples = [self._brownian_motion.sample(n) for _ in range(self.dim)]
        norm = np.array([np.linalg.norm(coord) for coord in zip(*brownian_samples)])
        return norm

    def sample(self, n):

        if isinstance(self.dim, int) and self.initial == 0:
            return self._sample_bessel_integer_dim_zero(n)
        else:
            return _sample_bessel_global(self.T, self.initial, self.dim, n)

    def simulate(self, n, N):
        """
        Simulate paths/trajectories from the instanced stochastic process.

        :param int n: number of steps in each path
        :param int N: number of paths to simulate
        :return: list with N paths (each one is a numpy array of size n)
        """
        self.n = n
        self.N = N
        self.times = get_times(self.T, n)

        if isinstance(self.dim, int) and self.initial == 0:

            self.paths = [self.sample(n) for _ in range(N)]
            return self.paths

        else:
            pool = Pool()
            initial = self.initial
            dim = self.dim
            T = self.T
            func = partial(_sample_bessel_global, T, initial, dim)
            results = pool.map(func, [n] * N)
            pool.close()
            pool.join()

            self.paths = results
            return self.paths

    def _process_expectation(self, times=None):
        # TODO: Add the case when times is zero, at the moment this fails because nc required division by t
        if times is None:
            times = self.times

        alpha = (self.dim / 2.0) - 1.0

        if np.isscalar(times):
            nc = (self.initial**2) / times
            expectations = (
                np.sqrt(times)
                * math.sqrt(math.pi / 2.0)
                * eval_genlaguerre(0.5, alpha, (-1.0 / 2.0) * nc)
            )
        else:
            nc = (self.initial**2) / times[1:]
            expectations = (
                np.sqrt(times[1:])
                * math.sqrt(math.pi / 2.0)
                * eval_genlaguerre(0.5, alpha, (-1.0 / 2.0) * nc)
            )
            expectations = np.insert(expectations, 0, self.initial)
        return expectations

    def _process_variance(self, times=None):
        if times is None:
            times = self.times
        expectations = self._process_expectation(times)
        variances = self.dim * times + self.initial**2 - expectations**2
        return variances

    def _process_stds(self, times=None):
        if times is None:
            times = self.times
        variances = self._process_variance(times=times)
        stds = np.sqrt(variances)
        return stds

    def get_marginal(self, t):
        marginal = ncx(df=self.dim, nc=self.initial / np.sqrt(t), scale=np.sqrt(t))
        return marginal

    def marginal_expectation(self, times=None):
        expectations = self._process_expectation(times=times)
        return expectations

    def marginal_variance(self, times):
        variances = self._process_variance(times=times)
        return variances


if __name__ == "__main__":
    # import matplotlib.pyplot as plt

    qs = "https://raw.githubusercontent.com/quantgirluk/matplotlib-stylesheets/main/quant-pastel-light.mplstyle"
    # plt.style.use(qs)

    p1 = BESProcess()
    p2 = BESProcess(dim=4.0)
    p3 = BESProcess(initial=5.0, dim=2.5)
    p4 = BESProcess(initial=3.0, dim=1.25)
    p5 = BESProcess(initial=1.0, dim=3.0)

    p1.plot(n=200, N=5, figsize=(12, 7), style=qs)
    p1.draw(n=200, N=200, figsize=(12, 7), style=qs, envelope=True)
    for p, cm in [
        (p1, "twilight"),
        (p2, "PuBuGn"),
        (p3, "Oranges"),
        (p4, "RdBu"),
        (p5, "Purples"),
        # (p6, "Oranges"),
    ]:

        p.draw(
            n=200,
            N=200,
            figsize=(12, 7),
            style=qs,
            colormap=cm,
            envelope=False,
        )
