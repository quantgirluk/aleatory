from gaussian import Gaussian
from utils import check_positive_number, check_numeric, get_times
import numpy as np

class BrownianMotion(Gaussian):

    def __init__(self, drift=0.0, scale=1.0, T=1.0, rng=None):
        super().__init__(T=T, rng=rng)
        self.drift = drift
        self.scale = scale
        self.times = None

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

        self._n = n
        self._times = get_times(self.T, n)

        bm = np.cumsum(self.scale * self._sample_gaussian_noise(n))
        bm = np.insert(bm, [0], 0)

        if self.drift == 0:
            return bm
        else:
            return self._times*self.drift + bm

    def sample(self, n):

        return self._sample_brownian_motion(n)

    def sample_at(self, times):
        """
        GGenerate Gaussian increments at specified times starting  from zero
        :param times:
        :return:
        """
        return self._sample_gaussian_noise_at(times)
