import numpy as np
from scipy.stats import lognorm

from replica.processes.base import SPExplicit
from replica.processes.exact_simulation.brownian_motion import BrownianMotion
from replica.utils.utils import check_positive_number, check_numeric, get_times, check_positive_integer


class GBM(SPExplicit):

    def __init__(self, drift=1.0, volatility=1.0, initial=1.0, T=1.0, rng=None):
        super().__init__(T=T, rng=rng, initial=initial)
        self.drift = drift
        self.volatility = volatility
        self._brownian_motion = BrownianMotion(T=T, rng=rng)
        self.name = "Geometric Brownian Motion"
        self.n = None
        self.times = None

    def __str__(self):
        return "Geometric Brownian motion with drift {d} and volatility {v} on [0, {T}].".format(
            T=str(self.T), d=str(self.drift), v=str(self.volatility))

    def __repr__(self):
        return "GeometricBrownianMotion(drift={d}, volatility={v}, T={T})".format(
            T=str(self.T), d=str(self.drift), v=str(self.volatility))

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

    def _process_expectation(self):
        return self.initial * np.exp(self.drift * self.times)

    def process_expectation(self):
        expectations = self._process_expectation()
        return expectations

    def _process_variance(self):
        variances = (self.initial ** 2) * np.exp(2 * self.drift * self.times) * (
                np.exp(self.times * self.volatility ** 2) - 1)
        return variances

    def process_variance(self):
        variances = self._process_variance()
        return variances

    def _process_stds(self):
        variances = self.process_variance()
        stds = np.sqrt(variances)
        return stds

    def get_marginal(self, t):
        mu_x = np.log(self.initial) + (self.drift - 0.5 * self.volatility ** 2) * t
        sigma_x = self.volatility * np.sqrt(t)
        marginal = lognorm(s=sigma_x, scale=np.exp(mu_x))

        return marginal

    def draw(self, n, N, marginal=False, envelope=False, style=None):
        self._draw_qqstyle(n=n, N=N, marginal=marginal, envelope=envelope)
        return 1
