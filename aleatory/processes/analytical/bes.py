"""
Bessel Process BES
"""
import math

import numpy as np
from scipy.stats import chi
from scipy.special import gamma

from aleatory.processes.base import SPExplicit
from aleatory.processes.analytical.brownian_motion import BrownianMotion
from aleatory.utils.utils import get_times, check_positive_integer


class BESProcess(SPExplicit):
    r"""Bessel process

    .. image:: _static/geometric_brownian_motion_drawn.png


    A Bessel process :math:`BESProcess^{n}_x` for :math:`n` integer is a continuous stochastic process
    :math:`\{X(t) : t \geq  0\}` is characterised as the Euclidian norm of an :math:`n`-dimensional
    Brownian motion. That is,

    .. math::
        X_t = \sqrt{\sum_{i=1}^n (W^i_t)^2}.

    It satisfies the following SDE

    .. math::
        dX_t = \frac{(n-1)}{2}  \frac{dt}{X_t} + dW_t \ \ \ \ t\in (0,T]


    with initial condition :math:`X_0 = x_0\geq0`,  where

    - :math:`n` is an integer
    - :math:`W_t` is a standard Brownian Motion.


    :param float dim: the dimension of the process :math:`n`
    :param float T: the right hand endpoint of the time interval :math:`[0,T]`
        for the process
    :param numpy.random.Generator rng: a custom random number generator

    """

    def __init__(self, dim=1.0, T=1.0, rng=None):
        super().__init__(T=T, rng=rng, initial=0.0)
        self.dim = dim
        self._brownian_motion = BrownianMotion(T=T, rng=rng)
        self.name = f'$BES^{{{self.dim}}}_0$'
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
        if not isinstance(value, int):
            raise TypeError("Current implementation is restricted to integer dimension.")
        self._dim = value

    def _sample_bessel_alpha_integer(self, n):
        check_positive_integer(n)

        self.n = n
        self.times = get_times(self.T, n)
        brownian_samples = [self._brownian_motion.sample(n) for _ in range(self.dim)]
        norm = np.array([np.linalg.norm(coord) for coord in zip(*brownian_samples)])
        return norm

    def sample(self, n):
        return self._sample_bessel_alpha_integer(n)

    def get_marginal(self, t):
        marginal = chi(df=self.dim, scale=(math.sqrt(t)))
        return marginal

    def _process_expectation(self, times=None):
        if times is None:
            times = self.times
        expectations = np.sqrt(times) * np.sqrt(2) * gamma((self.dim + 1) / 2) / gamma(self.dim / 2)
        return expectations

    def marginal_expectation(self, times=None):
        expectations = self._process_expectation(times=times)
        return expectations

    def _process_variance(self, times=None):
        if times is None:
            times = self.times
        variances = times * (self.dim - 2. * (gamma((self.dim + 1) / 2) / gamma(self.dim / 2)) ** 2)

        return variances

    def _process_stds(self):
        stds = np.sqrt(self._process_variance())
        return stds

    def process_stds(self):
        stds = self._process_stds()
        return stds
