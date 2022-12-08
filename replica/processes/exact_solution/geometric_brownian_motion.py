import numpy as np
from replica.processes.base import StochasticProcess
from replica.processes.exact_solution.brownian_motion import BrownianMotion
from replica.utils.utils import check_numeric, check_positive_integer, check_positive_number, get_times


class BrownianMotion(StochasticProcess):
    """
    Geometric Brownian motion X(t) : t >= 0
    dX(t) = X(t)*drift*dt + X(t)*volatility*dW(t)
    where W(t) is a standard Brownian motion.
    """
    def __init__(self, drift=0, volatility=1, T=1, rng=None):
        super().__init__(T=T, rng=rng)
        self._brownian_motion = BrownianMotion(T=T, rng=rng)
        self.drift = drift
        self.volatility = volatility
        self.n = None
        self.times = None
        self.name = "Geometric Brownian Motion"

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
        """Generate a realization of a geometric Brownian motion."""
        check_positive_integer(n)
        check_positive_number(initial, "Initial")
        self.n = n
        self.times = get_times(self.T, n)
        return initial * np.exp((self.drift - 0.5 * self.volatility ** 2) * self.times
                                + self.volatility * self._brownian_motion.sample(n))

    def _sample_geometric_brownian_motion_at(self, times, initial=1.0):
        """Generate a realization of a Geometric Brownian motion."""
        self.times = times
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
