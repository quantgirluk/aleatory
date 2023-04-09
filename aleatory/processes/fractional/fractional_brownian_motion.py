import numpy as np
from aleatory.processes.base import StochasticProcess
from aleatory.utils.utils import check_positive_integer, times_to_increments, get_times, check_in_zero_one

class fBM(StochasticProcess):
    r"""Fractional Brownian Motion
    """
    def __int__(self, hurst=0.5, T=1.0, rng=None):
        super().__init__(T=T, rng=rng)
        self.hurst = hurst
        self.name = "Fractional Brownian Motion"
        self.times = None
        self.n = None
        self.paths = None

    def __str__(self):
        return "Fractional Brownian Motion [0, {T}]".format(T=str(self.T))

    def __repr__(self):
        return "fBM(T={T})".format(T=str(self.T))

    @property
    def hurst(self):
        return self._hurst

    @hurst.setter
    def hurst(self, value):
        check_in_zero_one(value, "Hurst Exponent")
        self._hurst = value

    

    def _sample_fractional_brownian_motion(self, n):
        self.n = n
        self.times = get_times(self.T, self.n)

        # bm = np.cumsum(self.scale * self.gaussian_increments.sample(n - 1))
        # bm = np.insert(bm, 0, [0])
        # if self.drift == 0:
        #     return bm
        # else:
        #     return self.times * self.drift + bm



