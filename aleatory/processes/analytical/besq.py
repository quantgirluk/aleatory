"""
BESQ Process
"""
import numpy as np
from scipy.stats import chi2

from aleatory.processes.base import SPExplicit
from aleatory.processes.analytical.brownian_motion import BrownianMotion
from aleatory.utils.utils import get_times, check_positive_integer


class BESQProcess(SPExplicit):
    r"""Squared Bessel process

    .. image:: _static/geometric_brownian_motion_drawn.png


    A squared Bessel process :math:`BESQ^{n}_0` for :math:`n` integer is a continuous stochastic process
    :math:`\{X(t) : t \geq  0\}` which is characterised as the squared Euclidian norm of an :math:`n`-dimensional
    Brownian motion. That is,

    .. math::
        X_t = \sum_{i=1}^n (W^i_t)^2.

    It satisfies the following SDE

    .. math::
        dX_t = n dt + 2\sqrt{X_t} dW_t \ \ \ \ t\in (0,T]


    with initial condition :math:`X_0 = 0`,  where

    - :math:`n` is an integer
    - :math:`W_t` is a standard Brownian Motion.


    :param int dim: the dimension of the process :math:`n`
    :param float T: the right hand endpoint of the time interval :math:`[0,T]`
        for the process
    :param numpy.random.Generator rng: a custom random number generator

    """

    def __init__(self, dim=1.0, T=1.0, rng=None):
        super().__init__(T=T, rng=rng, initial=0.0)
        self.dim = dim
        self._brownian_motion = BrownianMotion(T=T, rng=rng)
        self.name = f'$BESQ^{{{self.dim}}}_0$'
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
        if not isinstance(value, int):
            raise TypeError("Current implementation is restricted to integer dimension.")
        self._dim = value

    def _sample_bessel_alpha_integer(self, n):
        check_positive_integer(n)

        self.n = n
        self.times = get_times(self.T, n)
        brownian_samples = [self._brownian_motion.sample(n) for _ in range(self.dim)]
        norm_squared = np.array([np.linalg.norm(coord)**2 for coord in zip(*brownian_samples)])
        return norm_squared

    def sample(self, n):
        return self._sample_bessel_alpha_integer(n)

    def get_marginal(self, t):
        marginal = chi2(df=self.dim, scale=t)
        return marginal

    def _process_expectation(self, times=None):
        if times is None:
            times = self.times
        expectations = self.dim*times
        return expectations

    def marginal_expectation(self, times=None):
        expectations = self._process_expectation(times=times)
        return expectations

    def _process_variance(self, times=None):
        if times is None:
            times = self.times
        variances = 2.0*self.dim*times**2

        return variances

    def _process_stds(self):
        stds = np.sqrt(self._process_variance())
        return stds

    def process_stds(self):
        stds = self._process_stds()
        return stds
