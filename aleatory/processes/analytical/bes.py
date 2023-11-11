"""
Bessel Process BES
"""
import math
from functools import partial
from multiprocessing import Pool

import numpy as np
from scipy.special import gamma, eval_genlaguerre
from scipy.stats import chi, ncx2

from aleatory.processes.analytical.brownian_motion import BrownianMotion
from aleatory.processes.base import SPExplicit

from aleatory.utils.utils import plot_paths, draw_paths, check_positive_integer, get_times, sample_besselq_global, \
    draw_paths_bessel


def _sample_bessel_global(T, initial, dim, n):
    path = np.sqrt(sample_besselq_global(T=T, initial=initial ** 2, dim=dim, n=n))

    return path


class BESProcess(SPExplicit):
    r"""Bessel process

    .. image:: _static/bes_process_drawn.png


    A Bessel process :math:`BES^{n}_x` for :math:`n` integer is a continuous stochastic process
    :math:`\{X(t) : t \geq  0\}` is characterised as the Euclidian norm of an :math:`n`-dimensional
    Brownian motion. That is,

    .. math::
        X_t = \sqrt{\sum_{i=1}^n (W^i_t)^2}.

    It satisfies the following SDE

    .. math::
        dX_t = \frac{(n-1)}{2}  \frac{dt}{X_t} + dW_t \ \ \ \ t\in (0,T]


    with initial condition :math:`X_0 = 0`,  where

    - :math:`n` is an integer
    - :math:`W_t` is a standard one-dimensional Brownian Motion.


    :param float dim: the dimension of the process :math:`n`
    :param float T: the right hand endpoint of the time interval :math:`[0,T]`
        for the process
    :param numpy.random.Generator rng: a custom random number generator

    """

    def __init__(self, dim=1.0, initial=0.0, T=1.0, rng=None):
        super().__init__(T=T, rng=rng, initial=initial)
        self.dim = dim
        self._brownian_motion = BrownianMotion(T=T, rng=rng)
        self.name = f'$BES^{{{self.dim}}}_{{{self.initial}}}$'
        self.n = None
        self.times = None

    def __str__(self):
        return "Bessel process with dimension {dim} and starting condition {initial} on [0, {T}].".format(
            T=str(self.T), dim=str(self.dim), initial=str(self.initial))

    def __repr__(self):
        return "BESProcess(dimension={dim}, initial={initial}, T={T})".format(
            T=str(self.T), dim=str(self.dim), initial=str(self.initial))

    @property
    def dim(self):
        """Bessel Process dimension."""
        return self._dim

    @dim.setter
    def dim(self, value):
        if value < 0:
            raise TypeError("Dimension must be positive")
        # if not isinstance(value, int):
        #     raise TypeError("Current implementation is restricted to integer dimension.")
        self._dim = value

    def _sample_bessel_alpha_integer(self, n):
        check_positive_integer(n)

        self.n = n
        self.times = get_times(self.T, n)
        brownian_samples = [self._brownian_motion.sample(n) for _ in range(self.dim)]
        norm = np.array([np.linalg.norm(coord) for coord in zip(*brownian_samples)])
        return norm

    def sample(self, n):

        if isinstance(self.dim, int) and self.initial == 0:
            return self._sample_bessel_alpha_integer(n)
        else:
            return _sample_bessel_global(self.T, self.initial, self.dim, n)

    def simulate(self, n, N):
        """
        Simulate paths/trajectories from the instanced stochastic process.

        :param n: number of steps in each path
        :param N: number of paths to simulate
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

    def get_marginal(self, t):
        marginal = ncx2(df=self.dim, nc=self.initial ** 2 / t, scale=t)
        return marginal

    def _process_expectation(self, times=None):
        if times is None:
            times = self.times

        alpha = (self.dim / 2.0) - 1.0
        nc = (self.initial ** 2) / times
        expectations = np.sqrt(times) * math.sqrt(math.pi / 2.0) * eval_genlaguerre(0.5, alpha, (-1.0 / 2.0) * nc)
        # expectations = self.initial + np.sqrt(times) * np.sqrt(2) * gamma((self.dim + 1) / 2) / gamma(self.dim / 2)
        return expectations

    def marginal_expectation(self, times=None):
        expectations = self._process_expectation(times=times)
        return expectations

    def _process_variance(self, times=None):
        if times is None:
            times = self.times
        expectations = self._process_expectation(times)
        variances = self.dim * times + self.initial ** 2 - expectations ** 2
        return variances

    def _process_stds(self):
        stds = np.sqrt(self._process_variance())
        return stds

    def process_stds(self):
        stds = self._process_stds()
        return stds

    def _draw_paths(self, n, N, marginal=False, envelope=False, type=None, title=None, **fig_kw):
        self.simulate(n, N)
        expectations = self._process_expectation()

        if envelope:
            marginals = [self.get_marginal(t) for t in self.times[1:]]
            upper = [self.initial] + [np.sqrt(m.ppf(0.005)) for m in marginals]
            lower = [self.initial] + [np.sqrt(m.ppf(0.995)) for m in marginals]
        else:
            upper = None
            lower = None

        if marginal:
            marginalT = self.get_marginal(self.T)
        else:
            marginalT = None

        chart_title = title if title else self.name
        fig = draw_paths_bessel(times=self.times, paths=self.paths, N=N, title=chart_title, expectations=expectations,
                                marginal=marginal, marginalT=marginalT, envelope=envelope, lower=lower, upper=upper,
                                **fig_kw)
        return fig

    def draw(self, n, N, marginal=True, envelope=False, title=None, **fig_kw):
        """
        Simulates and plots paths/trajectories from the instanced stochastic process.
        Visualisation shows
        - times versus process values as lines
        - the expectation of the process across time
        - histogram showing the empirical marginal distribution :math:`X_T`
        - probability density function of the marginal distribution :math:`X_T`
        - envelope of confidence intervals

        :param n: number of steps in each path
        :param N: number of paths to simulate
        :param marginal: bool, default: True
        :param envelope: bool, default: False
        :param title: string optional default to None
        :return:
        """
        return self._draw_paths(n, N, marginal=marginal, envelope=envelope, title=title, **fig_kw)
