from abc import ABC

import numpy as np

from aleatory.utils.utils import check_positive_number


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


class StochasticProcess(BaseProcess, ABC):
    """
    Base class for all one-factor stochastic processes classes.
    All processes of this type are defined on a finite interval $[0,T]$.
    """

    def __init__(self, T=1.0, rng=None):
        super().__init__(rng=rng)
        self.T = T

    @property
    def T(self):
        """End time of the process."""
        return self._T

    @T.setter
    def T(self, value):
        check_positive_number(value, "Time end")
        self._T = float(value)
