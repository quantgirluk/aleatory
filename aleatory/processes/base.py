from abc import ABC
from abc import abstractmethod

import numpy as np

from aleatory.utils.utils import check_positive_number
from aleatory.utils.utils import plot_paths, draw_paths, check_positive_integer, get_times


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
        """
        Simulate paths/trajectories from the instanced stochastic process.

        :param n: number of steps in each path
        :param N: number of paths to simulate
        :return: list with N paths (each one is a numpy array of size n)
        """
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

    def plot(self, n, N, title=None, **fig_kw):
        """
        Simulates and plots paths/trajectories from the instanced stochastic process.
        Simple plot of times versus process values as lines and/or markers.

        :param n: number of steps in each path
        :param N: number of paths to simulate
        :param title: string to customise plot title
        :return:
        """
        self.simulate(n, N)
        if title:
            figure = plot_paths(self.times, self.paths, title=title, **fig_kw)
        else:
            figure = plot_paths(self.times, self.paths, title=self.name, **fig_kw)
        return figure

    def _draw_paths(self, n, N, marginal=False, envelope=False, type=None, title=None, **fig_kw):
        self.simulate(n, N)
        expectations = self._process_expectation()

        if envelope:
            if type == '3sigma':
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

        chart_title = title if title else self.name
        fig = draw_paths(times=self.times, paths=self.paths, N=N, title=chart_title, expectations=expectations,
                         marginal=marginal, marginalT=marginalT, envelope=envelope, lower=lower, upper=upper,
                         **fig_kw)
        return fig

    def _draw_qqstyle(self, n, N, marginal=False, envelope=False, title=None,
                      **fig_kw):

        fig = self._draw_paths(n=n, N=N, marginal=marginal, envelope=envelope, type='qq', title=title, **fig_kw)
        return fig

    def _draw_3sigmastyle(self, n, N, marginal=False, envelope=False, title=None, **fig_kw):

        fig = self._draw_paths(n=n, N=N, marginal=marginal, envelope=envelope, type='3sigma', title=title, **fig_kw)
        return fig

    def draw(self, n, N, marginal=True, envelope=False, title=None, **fig_kw):
        """
        Simulates and plots paths/trajectories from the instanced stochastic process.
        Visualisation shows
        - times versus process values as lines
        - the expectation of the process across time
        - histogram showing the empirical marginal distribution :math:`X_T`
        - probability density function of the marginal distribution :math:`X_T`
        - envelope of confidence intervals

        :param n: number of steps in each path
        :param N: number of paths to simulate
        :param marginal: bool, default: True
        :param envelope: bool, default: False
        :param title: string optional default to None
        :return:
        """
        return self._draw_qqstyle(n, N, marginal=marginal, envelope=envelope, title=title, **fig_kw)


class SPEulerMaruyama(SPExplicit):
    def __int__(self, f=None, g=None, initial=0.0, T=1.0, rng=None):
        super().__init__(T=T, rng=rng, initial=initial)
        self.f = f
        self.g = g
        self.n = None
        self.dt = None
        self.times = None
        self.name = None

    def _sample_em_process(self, n, log=False):
        check_positive_integer(n)
        self.n = n
        self.dt = 1.0 * self.T / self.n
        self.times = get_times(self.T, self.n)
        dws = self.rng.normal(scale=np.sqrt(self.dt), size=self.n)

        if log:
            origin = np.log(self.initial)
        else:
            origin = self.initial

        path = [origin]
        previous = origin

        for (t, dw) in zip(self.times[1:], dws[1:]):
            previous += self.f(previous, t) * self.dt + self.g(previous, t) * dw
            path.append(previous)

        if log:
            return np.exp(path)

        return path

    def sample(self, n):
        return self._sample_em_process(n)
