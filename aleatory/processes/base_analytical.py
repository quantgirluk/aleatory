from abc import abstractmethod

import numpy as np

from aleatory.processes.base import StochasticProcess
from aleatory.utils.plotters import plot_paths, draw_paths


class SPAnalytical(StochasticProcess):

    def __init__(self, initial=0.0, name=None, T=1.0, rng=None):
        super().__init__(T=T, rng=rng)
        self.initial = initial
        self.name = name
        self.n = None
        self.times = None
        self.N = None
        self.paths = None
        self._empirical_marginals = None

    @abstractmethod
    def sample(self, n):  # pragma: no cover
        pass

    def simulate(self, n, N):
        """
        Simulate paths/trajectories from the instanced stochastic process.

        :param int n: number of steps in each path
        :param int N: number of paths to simulate
        :return: list with N paths (each one is a numpy array of size n)
        """
        self.n = n
        self.N = N
        self._empirical_marginals = (
            None  # Cleaning the empirical marginals from previous simulations
        )
        self.paths = [self.sample(n) for _ in range(N)]
        return self.paths

    def _get_empirical_marginal_samples(self):
        if self.paths is None:
            self.simulate(self.n, self.N)

        empirical_marginal_samples = np.array(self.paths).transpose()
        return empirical_marginal_samples

    def estimate_expectations(self):
        if self._empirical_marginals is None:
            self._empirical_marginals = self._get_empirical_marginal_samples()
        empirical_means = [np.mean(m) for m in self._empirical_marginals]
        return empirical_means

    def estimate_variances(self):
        if self._empirical_marginals is None:
            self._empirical_marginals = self._get_empirical_marginal_samples()
        empirical_vars = [np.var(m) for m in self._empirical_marginals]
        return empirical_vars

    def estimate_stds(self):
        variances = self.estimate_variances()
        stds = [np.sqrt(var) for var in variances]
        return stds

    def estimate_quantiles(self, q):
        if self._empirical_marginals is None:
            self._empirical_marginals = self._get_empirical_marginal_samples()
        empirical_quantiles = [np.quantile(m, q) for m in self._empirical_marginals]
        return empirical_quantiles

    def _process_expectation(self):
        return self.estimate_expectations()

    def process_expectation(self):
        return self._process_expectation()

    def _process_variance(self):
        return self.estimate_variances()

    def process_variance(self):
        return self._process_variance()

    def _process_stds(self):
        stds = np.sqrt(self.process_variance())
        return stds

    def process_stds(self):
        stds = self._process_stds()
        return stds

    # def get_marginal(self, t):
    #     pass

    def _plot_process(self, n, N, title=None, **fig_kw):
        """
        Simulates and plots paths/trajectories from the instanced stochastic process.
        Simple plot of times versus process values as lines and/or markers.

        :parameter int n: number of steps in each path
        :parameter int N: number of paths to simulate
        :parameter str title: string to customise plot title
        :return:

        """
        self.simulate(n, N)
        if title:
            figure = plot_paths(
                times=self.times, paths=self.paths, title=title, **fig_kw
            )
        else:
            figure = plot_paths(
                times=self.times, paths=self.paths, title=self.name, **fig_kw
            )
        return figure

    def plot(self, n, N, title=None, **fig_kw):
        """
        Simulates and plots paths/trajectories from the instanced stochastic process.
        Simple plot of times versus process values as lines and/or markers.

        :parameter int n: number of steps in each path
        :parameter int N: number of paths to simulate
        :parameter str title: string to customise plot title
        :return:

        """
        self.simulate(n, N)
        if title:
            figure = plot_paths(
                times=self.times, paths=self.paths, title=title, **fig_kw
            )
        else:
            figure = plot_paths(
                times=self.times, paths=self.paths, title=self.name, **fig_kw
            )
        return figure

    def _draw_paths(
        self,
        n,
        N,
        marginal=False,
        envelope=False,
        type=None,
        title=None,
        empirical_envelope=False,
        **fig_kw,
    ):
        self.simulate(n, N)
        expectations = self._process_expectation()

        marginal_available = getattr(self, "get_marginal", None)
        marginal_available = callable(marginal_available)

        if envelope:
            if type == "3sigma":
                stds = self._process_stds()
                upper = expectations + 3.0 * stds
                lower = expectations - 3.0 * stds
            else:
                if marginal_available and empirical_envelope == False:
                    marginals = [self.get_marginal(t) for t in self.times[1:]]
                    upper = [self.initial] + [m.ppf(0.005) for m in marginals]
                    lower = [self.initial] + [m.ppf(0.995) for m in marginals]
                else:
                    upper = self.estimate_quantiles(0.005)
                    lower = self.estimate_quantiles(0.995)
        else:
            upper = None
            lower = None

        if marginal and marginal_available:
            marginalT = self.get_marginal(self.T)
        else:
            marginalT = None

        chart_title = title if title else self.name
        fig = draw_paths(
            times=self.times,
            paths=self.paths,
            N=N,
            title=chart_title,
            expectations=expectations,
            marginal=marginal,
            marginalT=marginalT,
            envelope=envelope,
            lower=lower,
            upper=upper,
            **fig_kw,
        )
        return fig

    def _draw_qqstyle(self, n, N, marginal=False, envelope=False, title=None, **fig_kw):

        fig = self._draw_paths(
            n=n,
            N=N,
            marginal=marginal,
            envelope=envelope,
            type="qq",
            title=title,
            **fig_kw,
        )
        return fig

    def _draw_3sigmastyle(
        self, n, N, marginal=False, envelope=False, title=None, **fig_kw
    ):

        fig = self._draw_paths(
            n=n,
            N=N,
            marginal=marginal,
            envelope=envelope,
            type="3sigma",
            title=title,
            **fig_kw,
        )
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

        :param int n: number of steps in each path
        :param int N: number of paths to simulate
        :param bool marginal: defaults to True
        :param bool envelope: defaults to False
        :param str title: string optional default to the name of the process
        :return:
        """
        return self._draw_qqstyle(
            n, N, marginal=marginal, envelope=envelope, title=title, **fig_kw
        )
