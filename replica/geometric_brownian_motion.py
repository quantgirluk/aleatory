import numpy as np
from base import StochasticProcess
from brownian_motion import BrownianMotion
from utils import check_numeric, check_positive_integer, check_positive_number, get_times


class GeometricBrownianMotion(StochasticProcess):
    r"""Geometric Brownian motion process.
    .. image:: _static/geometric_brownian_motion.png
        :scale: 50%
    A geometric Brownian motion :math:`S_t` is the analytic solution to the
    stochastic differential equation with Wiener process :math:`W_t`:
    .. math::
        dS_t = \mu S_t dt + \sigma S_t dW_t
    and can be represented with initial value :math:`S_0` in the form:
    .. math::
        S_t = S_0 \exp \left( \left( \mu - \frac{\sigma^2}{2} \right) t +
        \sigma W_t \right)
    :param float drift: the parameter :math:`\mu`
    :param float volatility: the parameter :math:`\sigma`
    :param float T: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param numpy.random.Generator rng: a custom random number generator
    """

    def __init__(self, drift=0, volatility=1, T=1, rng=None):
        super().__init__(T=T, rng=rng)
        self._brownian_motion = BrownianMotion(T=T, rng=rng)
        self.drift = drift
        self.volatility = volatility
        self._n = None
        self._times = None

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

    def _sample_geometric_brownian_motion(self, n, initial=1.0):
        """Generate a realization of geometric Brownian motion."""
        check_positive_integer(n)
        check_positive_number(initial, "Initial")

        # Opt for repeated use
        if self._n != n:
            self._n = n
            self._times = get_times(self.T, n)
        return initial * np.exp((self.drift - 0.5 * self.volatility ** 2) * self._times
                                + self.volatility * self._brownian_motion.sample(n))

    def _sample_geometric_brownian_motion_at(self, times, initial=1.0):
        """Generate a realization of geometric Brownian motion."""

        return initial * np.exp((self.drift - 0.5 * self.volatility ** 2) * times
                                + self.volatility * self._brownian_motion.sample_at(times))

    def sample(self, n, initial=1):
        """Generate a realization.
        """
        return self._sample_geometric_brownian_motion(n, initial)

    def sample_at(self, times, initial=1):
        """Generate a realization using specified times.
        """
        return self._sample_geometric_brownian_motion_at(times, initial)
