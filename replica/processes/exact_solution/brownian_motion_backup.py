from replica.processes.exact_simulation.gaussian import GaussianIncrements
from replica.utils.utils import check_positive_number, check_numeric, get_times
import numpy as np


class BrownianMotion(GaussianIncrements):
    """
    Brownian motion :math:`B(t) : t >= 0`

    A standard Brownian motion has the following properties:

    1. Starts at zero, i.e. :math:`B(0) = 0`
    2. Independent increments
    3. :math:`B(t) - B(s)` follows a Gaussian distribution :math:`N(0, t-s)`
    4. Almost surely continuous


    A more general version of a Brownian motion is defined as
    :math:`W(t) = drift*t + scale*B(t)`

    """

    def __init__(self, drift=0.0, scale=1.0, T=1.0, rng=None):
        super().__init__(T=T, rng=rng)
        self.drift = drift
        self.scale = scale
        self.times = None
        self.name = "Brownian Motion"

    @property
    def drift(self):
        return self._drift

    @drift.setter
    def drift(self, value):
        check_numeric(value, "Drift")
        self._drift = value

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        check_positive_number(value, "Scale")
        self._scale = value

    def _sample_brownian_motion(self, n):
        self.n = n
        self.times = get_times(self.T, self.n)
        bm = np.cumsum(self.scale * self._sample_gaussian_noise(n))
        if self.drift == 0:
            return bm
        else:
            return self.times * self.drift + bm

    def sample(self, n):
        """
        Generates a sample from a Brownian motion
        :param n:
        :return:
        """
        return self._sample_brownian_motion(n)

    def _sample_brownian_motion_at(self, times):
        """Generate a sample from Brownian motion at specified times."""
        self.times = times
        bm = np.cumsum(self.scale * self._sample_gaussian_noise_at(times))

        if times[0] != 0:
            bm = np.insert(bm, 0, [0])

        if self.drift != 0:
            bm += [self.drift * t for t in times]

        return bm

    def sample_at(self, times):
        """
        Generates a sample from a Brownian motion at the specified times.
        :param times:
        :return:
        """
        temp = self._sample_brownian_motion_at(times)
        return temp
