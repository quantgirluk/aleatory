from abc import ABC
from abc import abstractmethod
import matplotlib.pyplot as plt
from replica.utils.utils import plot_paths, draw_paths, check_positive_integer, get_times
import numpy as np

from replica.utils.utils import check_positive_number


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


class SPExplicit(StochasticProcess):

    def __init__(self, initial=0.0, name=None, T=1.0, rng=None):
        super().__init__(T=T, rng=rng)
        self.initial = initial
        self.name = name
        self.n = None
        self.times = None
        self.N = None
        self.paths = None

    @abstractmethod
    def sample(self, n):  # pragma: no cover
        pass

    def simulate(self, n, N):
        self.n = n
        self.N = N
        self.paths = [self.sample(n) for _ in range(N)]
        return self.paths

    @abstractmethod
    def get_marginal(self, t):
        pass

    @abstractmethod
    def _process_expectation(self):
        pass

    @abstractmethod
    def _process_variance(self):
        pass

    @abstractmethod
    def _process_stds(self):
        pass

    def plot(self, n, N):
        self.simulate(n, N)
        plot_paths(self.times, self.paths, self.name)
        return 1

    def _draw_paths(self, n, N, marginal=False, envelope=False, style=None):

        self.simulate(n, N)
        expectations = self._process_expectation()

        if envelope:
            if style == '3sigma':
                stds = self._process_stds()
                upper = expectations + 3.0 * stds
                lower = expectations - 3.0 * stds
            else:
                marginals = [self.get_marginal(t) for t in self.times[1:]]
                upper = [self.initial] + [m.ppf(0.005) for m in marginals]
                lower = [self.initial] + [m.ppf(0.995) for m in marginals]
        else:
            upper = None
            lower = None

        if marginal:
            marginalT = self.get_marginal(self.T)
        else:
            marginalT = None

        draw_paths(times=self.times, paths=self.paths, N=N, expectations=expectations, name=self.name,
                   marginal=marginal,
                   marginalT=marginalT, envelope=envelope, lower=lower, upper=upper)
        return 1

    def _draw_qqstyle(self, n, N, marginal=False, envelope=False):

        self._draw_paths(n=n, N=N, marginal=marginal, envelope=envelope, style='qq')
        return 1

    def draw(self, n, N, marginal=False, envelope=False, style=None):

        self._draw_paths(n, N, marginal=marginal, envelope=envelope, style=style)


class SPEulerMaruyama(SPExplicit):
    def __int__(self,  f=None, g=None, initial=0.0, T=1.0, rng=None):
        super().__init__(T=T, rng=rng)
        self.f = f
        self.g = g
        self.initial = initial
        self.n = None
        self.dt = None
        self.times = None
        self.name = None

    def _sample_em_process(self, n):

        check_positive_integer(n)
        self.n = n
        self.dt = 1.0 * self.T / self.n
        self.times = get_times(self.T, self.n)
        dws = self.rng.normal(scale=np.sqrt(self.dt), size=self.n)
        path = [self.initial]
        previous = self.initial

        for (t, dw) in zip(self.times[1:], dws[1:]):
            previous += self.f(previous, t) * self.dt + self.g(previous, t) * dw
            path.append(previous)

        return path

    def sample(self, n):
        return self._sample_em_process(n)

