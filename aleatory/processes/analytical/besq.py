"""
BESQ Process
"""
from functools import partial
from multiprocessing import Pool

import numpy as np
from scipy.stats import ncx2

from aleatory.processes.analytical.brownian_motion import BrownianMotion
from aleatory.processes.base import SPExplicit
from aleatory.utils.utils import get_times, check_positive_integer, sample_besselq_global


def _sample_besselq_global(T, initial, dim, n):
    path = sample_besselq_global(T=T, initial=initial, dim=dim, n=n)

    return path


class BESQProcess(SPExplicit):
    r"""Squared Bessel process

    .. image:: _static/besq_process_drawn.png


    A squared Bessel process :math:`BESQ^{n}_{0}`, for :math:`n` integer is a continuous stochastic process
    :math:`\{X(t) : t \geq  0\}` which is characterised as the squared Euclidian norm of an :math:`n`-dimensional
    Brownian motion. That is,

    .. math::
        X_t = \sum_{i=1}^n (W^i_t)^2.

    More generally, for any :math:`\delta >0`, and :math:`x_0 \geq 0`, a squared Bessel process of
    dimension :math:`\delta` starting at :math:`x_0`, denoted by

    .. math::
        BESQ_{{x_0}}^{{\delta}}

    can be defined by the following SDE

    .. math::
        dX_t = \delta dt + 2\sqrt{X_t} dW_t \ \ \ \ t\in (0,T]


    with initial condition :math:`X_0 = x_0`,  where

    - :math:`\delta` is a positive real
    - :math:`W_t` is a standard Brownian Motion.


    :param double dim: the dimension of the process :math:`n`
    :param double initial: the initial point of the process :math:`x_0`
    :param double T: the right hand endpoint of the time interval :math:`[0,T]`
        for the process
    :param numpy.random.Generator rng: a custom random number generator

    """

    def __init__(self, dim=1.0, initial=0.0, T=1.0, rng=None):
        super().__init__(T=T, rng=rng, initial=initial)
        self.dim = dim
        self._brownian_motion = BrownianMotion(T=T, rng=rng)
        self.name = f'$BESQ^{{{self.dim}}}_{{{self.initial}}}$'
        self.n = None
        self.times = None

    def __str__(self):
        return "BESQ process with dimension {dim} and starting condition {initial} on [0, {T}].".format(
            T=str(self.T), dim=str(self.dim), initial=str(self.initial))

    def __repr__(self):
        return "Squared Bessel Process(dimension={dim}, initial={initial}, T={T})".format(
            T=str(self.T), dim=str(self.dim), initial=str(self.initial))

    @property
    def dim(self):
        """Bessel Process dimension."""
        return self._dim

    @dim.setter
    def dim(self, value):
        if value < 0:
            raise TypeError("Dimension must be positive.")
        self._dim = value

    def _sample_besselq_alpha_integer(self, n):
        check_positive_integer(n)

        self.n = n
        self.times = get_times(self.T, n)
        brownian_samples = [self._brownian_motion.sample(n) for _ in range(self.dim)]
        norm_squared = np.array([np.linalg.norm(coord) ** 2 for coord in zip(*brownian_samples)])
        return norm_squared

    def sample(self, n):

        if isinstance(self.dim, int) and self.initial == 0:
            return self._sample_besselq_alpha_integer(n)
        else:
            return _sample_besselq_global(self.T, self.initial, self.dim, n)

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
            func = partial(_sample_besselq_global, T, initial, dim)
            results = pool.map(func, [n] * N)
            pool.close()
            pool.join()

            self.paths = results
            return self.paths

    def get_marginal(self, t):

        marginal = ncx2(df=self.dim, nc=self.initial / t, scale=t)
        # ncx2(df=dim, nc=x / t_size, scale=t_size).rvs(1)[0]
        return marginal

    def _process_expectation(self, times=None):
        if times is None:
            times = self.times
        expectations = self.initial + self.dim * np.array(times)
        return expectations

    def marginal_expectation(self, times=None):
        expectations = self._process_expectation(times=times)
        return expectations

    def _process_variance(self, times=None):
        if times is None:
            times = self.times
        variances = 2.0 * (self.dim + 2.0*self.initial/times) * times**2

        return variances

    def marginal_variance(self, times):
        variances = self._process_variance(times=times)
        return variances

    def _process_stds(self):
        stds = np.sqrt(self._process_variance())
        return stds

    def process_stds(self):
        stds = self._process_stds()
        return stds
