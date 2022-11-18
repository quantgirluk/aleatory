from abc import ABC
from abc import abstractmethod

from utils import check_positive_number, check_positive_integer, get_times

import numpy as np


class BaseProcess(ABC):
    def __init__(self, rng=None):
        self.rng = rng

    @property
    def rng(self):
        if self._rng is None:
            return np.random.default_rng()
        return self._rng

    @rng.setter
    def rng(self, value):
        if value is None:
            self._rng = None
        elif isinstance(value, (np.random.RandomState, np.random.Generator)):
            self._rng = value
        else:
            raise TypeError("rng must be of type `numpy.random.Generator`")

    @abstractmethod
    def sample(self, n):  # pragma: no cover
        pass


class StochasticProcess(BaseProcess, ABC):
    def __init__(self, T=1.0, rng=None):
        super().__init__(rng=rng)
        self.T = T
        self._n = None
        self.times = None

    @property
    def T(self):
        """End time of the process."""
        return self._T

    @T.setter
    def T(self, value):
        check_positive_number(value, "Time end")
        self._T = float(value)

    # def _set_times(self, n):
    #     if self._n != n:
    #         check_positive_integer(n)
    #         self._n = n
    #         self.times = get_times(self.T, n - 1)
    #
    # def times(self, n):
    #     """Generate times associated with n increments on [0, t].
    #     :param int n: the number of increments
    #     """
    #     self._set_times(n)
    #     return self.times
