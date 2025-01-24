"""
Geometric Brownian Motion
"""

import numpy as np
from scipy.stats import lognorm

from aleatory.processes.base_analytical import SPAnalytical
from aleatory.processes.analytical.brownian_motion import BrownianMotion
from aleatory.utils.utils import (
    check_positive_number,
    check_numeric,
    get_times,
    check_positive_integer,
)


class GBM(SPAnalytical):
    r"""
    Geometric Brownian Motion
    =========================

    A Geometric Brownian Motion object.

    .. image:: ../_static/geometric_brownian_motion_drawn.png

    Notes
    -----

    A Geometric Brownian Motion :math:`\{X(t) : t \geq  0\}` is characterised by
    the following SDE.

    .. math::
        dX_t = \mu X_t dt + \sigma X_t dW_t \ \ \ \ t\in (0,T]


    with initial condition :math:`X_0 = x_0\geq0`, where

    - :math:`\mu` is the drift
    - :math:`\sigma>0` is the volatility
    - :math:`W_t` is a standard Brownian Motion.


    The solution to this equation can be written as

    .. math::

        X_t = x_0\exp\left((\mu + \frac{\sigma^2}{2} )t +\sigma W_t\right)

    and each :math:`X_t` follows a log-normal distribution.

    Examples
    --------

    .. highlight:: python
    .. code-block:: python

        from aleatory.processes import GBM
        process = GBM()
        fig = process.draw(n=100, N=5, figsize=(12, 7))
        fig.show()

    Constructor, Methods, and Attributes
    ------------------------------------

    """

    def __init__(self, drift=1.0, volatility=0.5, initial=1.0, T=1.0, rng=None):
        """
        :param float drift: the parameter :math:`\mu` in the above SDE
        :param float volatility: the parameter :math:`\sigma>0` in the above SDE
        :param float initial: the initial condition :math:`x_0` in the above SDE
        :param float T: the right hand endpoint of the time interval :math:`[0,T]`
            for the process
        :param numpy.random.Generator rng: a custom random number generator

        """

        super().__init__(T=T, rng=rng, initial=initial)
        self.drift = drift
        self.volatility = volatility
        self._brownian_motion = BrownianMotion(T=T, rng=rng)
        self.name = f"Geometric Brownian Motion $X(\\mu={self.drift}, \\sigma={self.volatility})$"
        self.n = None
        self.times = None

    def __str__(self):
        return "Geometric Brownian motion with drift {d} and volatility {v} on [0, {T}].".format(
            T=str(self.T), d=str(self.drift), v=str(self.volatility)
        )

    def __repr__(self):
        return "GeometricBrownianMotion(drift={d}, volatility={v}, T={T})".format(
            T=str(self.T), d=str(self.drift), v=str(self.volatility)
        )

    @property
    def drift(self):
        """Geometric Brownian motion drift parameter."""
        return self._drift

    @drift.setter
    def drift(self, value):
        check_numeric(value, "Drift")
        self._drift = value

    @property
    def volatility(self):
        """Geometric Brownian motion volatility parameter."""
        return self._volatility

    @volatility.setter
    def volatility(self, value):
        check_positive_number(value, "Volatility")
        self._volatility = value

    @property
    def initial(self):
        """Geometric Brownian motion initial point."""
        return self._initial

    @initial.setter
    def initial(self, value):
        check_positive_number(value, "Initial Point")
        self._initial = value

    def _sample_geometric_brownian_motion(self, n):
        """Generate a realization of a geometric Brownian motion."""
        check_positive_integer(n)
        check_positive_number(self.initial, "Initial")
        self.n = n
        self.times = get_times(self.T, n)
        return self.initial * np.exp(
            (self.drift - 0.5 * self.volatility**2) * self.times
            + self.volatility * self._brownian_motion.sample(n)
        )

    def _sample_geometric_brownian_motion_at(self, times):
        """Generate a realization of a Geometric Brownian motion."""
        self.times = times
        return self.initial * np.exp(
            (self.drift - 0.5 * self.volatility**2) * times
            + self.volatility * self._brownian_motion.sample_at(times)
        )

    def sample(self, n):
        """Generate a realization."""
        return self._sample_geometric_brownian_motion(n)

    def sample_at(self, times):
        """Generate a realization using specified times."""
        return self._sample_geometric_brownian_motion_at(times)

    def _process_expectation(self, times=None):
        if times is None:
            times = self.times
        return self.initial * np.exp(self.drift * times)

    def _process_variance(self, times=None):
        if times is None:
            times = self.times
        variances = (
            (self.initial**2)
            * np.exp(2 * self.drift * times)
            * (np.exp(times * self.volatility**2) - 1)
        )
        return variances

    def _process_stds(self, times=None):
        if times is None:
            times = self.times
        variances = self._process_variance(times=times)
        stds = np.sqrt(variances)
        return stds

    def get_marginal(self, t):
        mu_x = np.log(self.initial) + (self.drift - 0.5 * self.volatility**2) * t
        sigma_x = self.volatility * np.sqrt(t)
        marginal = lognorm(s=sigma_x, scale=np.exp(mu_x))

        return marginal

    def marginal_expectation(self, times=None):
        expectations = self._process_expectation(times=times)
        return expectations

    def marginal_variance(self, times=None):
        variances = self._process_variance(times=times)
        return variances
