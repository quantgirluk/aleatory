from abc import abstractmethod

import numpy as np

from aleatory.processes.base import StochasticProcess
from aleatory.utils.plotters import plot_paths, draw_paths
from aleatory.utils.plotters_marginals import plot_mean_variance

class SPAnalytical(StochasticProcess):

    def __init__(self, initial=0.0, name=None, T=1.0, rng=None):
        super().__init__(T=T, rng=rng)
        self.initial = initial
        self.name = name
        self.name = name
        self.n = None
        self.times = None
        self.N = None
        self.paths = None
        self._empirical_marginals = None

    @abstractmethod
    def sample(self, n):  # pragma: no cover
        pass

    def simulate(self, n, N, T=None):
        """
        Simulate paths/trajectories from the instanced stochastic process.

        :param int n: number of steps in each path
        :param int N: number of paths to simulate
        :return: list with N paths (each one is a numpy array of size n)
        """
        if T:
            self.T = T
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

    def _plot_process(self, n, N, T=None, title=None, suptitle=None, **fig_kw):
        """
        Simulates and plots paths/trajectories from the instanced stochastic process.
        Simple plot of times versus process values as lines and/or markers.

        :parameter int n: number of steps in each path
        :parameter int N: number of paths to simulate
        :parameter float T: the right hand endpoint of the time interval [0,T]
        :parameter str title: string to customise plot title
        :return:

        """
        self.simulate(n, N, T=T)
        chart_suptitle = suptitle if suptitle is not None else self.name
        figure = plot_paths(
            times=self.times,
            paths=self.paths,
            title=title,
            suptitle=chart_suptitle,
            **fig_kw,
        )
        return figure

    def plot(self, n, N, T=None, title=None, suptitle=None, **fig_kw):
        """
        Simulates and plots paths/trajectories from the instanced stochastic process.
        Simple plot of times versus process values as lines and/or markers.

        :parameter int n: number of steps in each path
        :parameter int N: number of paths to simulate
        :parameter float T: the right hand endpoint of the time interval [0,T]
        :parameter str title: string to customise plot title
        :return:

        """
        self.simulate(n, N, T=T)
        chart_suptitle = suptitle if suptitle is not None else self.name
        figure = plot_paths(
            times=self.times,
            paths=self.paths,
            title=title,
            suptitle=chart_suptitle,
            **fig_kw,
        )
        return figure

    def _draw_paths(
        self,
        n,
        N,
        T=None,
        marginal=False,
        envelope=False,
        type=None,
        title=None,
        suptitle=None,
        empirical_envelope=False,
        **fig_kw,
    ):  
        
        self.simulate(n, N, T=T)
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

        chart_suptitle = suptitle if suptitle is not None else self.name
        fig = draw_paths(
            times=self.times,
            paths=self.paths,
            N=N,
            title=title,
            suptitle=chart_suptitle,
            expectations=expectations,
            marginal=marginal,
            marginalT=marginalT,
            envelope=envelope,
            lower=lower,
            upper=upper,
            **fig_kw,
        )
        return fig

    def _draw_qqstyle(
        self,
        n,
        N,
        T=None,
        marginal=False,
        envelope=False,
        title=None,
        suptitle=None,
        **fig_kw,
    ):

        fig = self._draw_paths(
            n=n,
            N=N,
            T=T,
            marginal=marginal,
            envelope=envelope,
            type="qq",
            title=title,
            suptitle=suptitle,
            **fig_kw,
        )
        return fig

    def _draw_3sigmastyle(
        self,
        n,
        N,
        T=None,
        marginal=False,
        envelope=False,
        title=None,
        suptitle=None,
        **fig_kw,
    ):

        fig = self._draw_paths(
            n=n,
            N=N,
            T=T,
            marginal=marginal,
            envelope=envelope,
            type="3sigma",
            title=title,
            suptitle=suptitle,
            **fig_kw,
        )
        return fig

    def draw(
        self,
        n,
        N,
        T=None,
        marginal=True,
        envelope=False,
        title=None,
        suptitle=None,
        **fig_kw,
    ):
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
            n,
            N,
            T=T,
            marginal=marginal,
            envelope=envelope,
            title=title,
            suptitle=suptitle,
            **fig_kw,
        )


class SPAnalyticalMarginals(SPAnalytical):

    def marginal_expectation(self, times=None):
        expectations = self._process_expectation(times=times)
        return expectations

    def marginal_variance(self, times):
        variances = self._process_variance(times=times)
        return variances

    def marginal_stds(self, times=None):
        variances = self._process_variance(times=times)
        stds = np.sqrt(variances)
        return stds
    
    def _plot_mean_variance(self, times, title=None, **fig_kw):
        """
        Plots the expectation and variance of the process as a function of time.

        :param list times: list of times to evaluate the expectation and variance
        :param str title: string optional default to the name of the process
        :param fig_kw: keyword arguments for the figure
        :return:
        """

        plot_title = title if title else self.name

        expectations = self.marginal_expectation(times=times)
        variances = self.marginal_variance(times=times)
        fig = plot_mean_variance(
            times=times,
            means=expectations,
            variances=variances,
            name=plot_title,
            **fig_kw,
        )
        return fig
    
    def plot_mean_variance(self, times, title=None, **fig_kw):

        return self._plot_mean_variance(times=times, title=title, **fig_kw)
