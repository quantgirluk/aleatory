"""
Brownian Excursion
"""
import numpy as np

from aleatory.processes import BrownianBridge
from aleatory.utils.utils import check_positive_integer
from scipy.stats import chi


class BrownianExcursion(BrownianBridge):
    r"""
    Brownian Excursion

    .. image:: _static/brownian_excursion_drawn.png

    A Brownian excursion process, is a Wiener process (or Brownian motion) conditioned
    to be positive and to take the value 0 at time 1. Alternatively, it can be defined as a Brownian
    Bridge process conditioned to be positive.

    :param float T: the right hand endpoint of the time interval :math:`[0,T]`
        for the process
    :param numpy.random.Generator rng: a custom random number generator

    """
    def __init__(self, T=1.0, rng=None):
        super().__init__(T=T, rng=rng)
        self.name = "Brownian Excursion"
        self.description = "Brownian Excursion"
        self._brownian_bridge = BrownianBridge(initial=0.0, end=0.0, T=T, rng=rng)
        self.n = None
        self.times = None

    def __str__(self):
        return "Brownian Excursion"

    def __repr__(self):
        return "BrownianExcursion"

    def _sample_brownian_excursion(self, n):
        """Generate a random sample of the Brownian Excursion."""
        check_positive_integer(n)
        self.n = n
        self.times = np.linspace(0, 1, n)
        bridge_path = self._brownian_bridge.sample(n)
        id_bridge_min = np.argmin(bridge_path)
        excursion_path = [bridge_path[(id_bridge_min + idx) % (n - 1)] - bridge_path[id_bridge_min] for idx in range(n)]
        return np.asarray(excursion_path)

    def _sample_brownian_excursion_at(self, times):
        self.times = times
        bridge_path = self._brownian_bridge.sample_at(times)
        id_bridge_min = np.argmin(bridge_path)
        n = len(times)
        excursion_path = [bridge_path[(id_bridge_min + idx) % (n - 1)] - bridge_path[id_bridge_min] for idx in range(n)]
        return np.asarray(excursion_path)

    def sample(self, n):
        return self._sample_brownian_excursion(n)

    def sample_at(self, times):
        return self._sample_brownian_excursion_at(times)

    def _process_expectation(self, times=None):
        if times is None:
            times = self.times

        return np.sqrt(times * (1.0 - times)) * chi.mean(df=3)

    def _process_variance(self, times=None):
        if times is None:
            times = self.times
        return times * (1.0 - times) * chi.var(df=3)

    def get_marginal(self, t):
        scale = np.sqrt(t * (1.0 - t))
        marginal = chi(df=3, scale=scale)
        return marginal
