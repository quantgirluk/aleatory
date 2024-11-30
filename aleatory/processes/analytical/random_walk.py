"""
Random Walk
"""
from aleatory.processes.base_analytical import SPAnalytical
from abc import ABC
from aleatory.utils.utils import check_positive_integer, get_times, plot_paths_random_walk
import numpy as np


class GeneralRandomWalk(SPAnalytical, ABC):

    def __init__(self, p, rng=None):
        super().__init__(rng=rng)
        self.step_sizes = (1., -1.)
        self.p = p
        self.q = 1. - p
        self.probs = (self.p, self.q)
        self.paths = None
        self.n = None
        self.N = None
        self.name = "General Random Walk"
        self.times = None

    def __str__(self):
        return f'General Random Walk with step sizes {self.step_sizes} and probabilities {self.probs}'

    def __repr__(self):
        return f'GeneralRandomWalk(step_sizes={self.step_sizes}, probabilities={self.probs})'

    def _sample_random_walk_steps(self, n):
        """Generate a sample of a general random walk increments"""
        check_positive_integer(n)
        steps = self.rng.choice(self.step_sizes, p=self.probs, size=n)
        return steps

    def _sample_random_walk(self, n):
        """Generate a sample from a general random walk"""
        self.T = n
        self.n = n
        self.times = get_times(self.T, self.n + 1)
        sample = np.array([0] + list(np.cumsum(self._sample_random_walk_steps(n))))
        return sample

    def sample(self, n):
        sample = self._sample_random_walk(n)
        return sample

    def _process_expectation(self, times=None):
        if times is None:
            times = self.times
        return times * (self.p - self.q)

    def _process_variance(self, times=None):
        if times is None:
            times = self.times
        return times * 4.0 * self.p * self.q

    def marginal_expectation(self, times=None):
        expectations = self._process_expectation(times=times)
        return expectations

    def marginal_variance(self, times):
        variances = self._process_variance(times=times)
        return variances

    def plot(self, *args, n, N, title=None, **fig_kw):
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
            figure = plot_paths_random_walk(*args, times=self.times, paths=self.paths, title=title, **fig_kw)
        else:
            figure = plot_paths_random_walk(*args, times=self.times, paths=self.paths, title=self.name, **fig_kw)
        return figure

    def draw(self, n, N, marginal=True, envelope=True, title=None, **fig_kw):
        """
        Simulates and plots paths/trajectories from the instanced stochastic process.

        Produces different kind of visualisation illustrating the following elements:

        - times versus process values as lines
        - the expectation of the process across time
        - histogram showing the empirical marginal distribution :math:`X_T` (optional when ``marginal = True``)
        - probability density function of the marginal distribution :math:`X_T` (optional when ``marginal = True``)
        - envelope of confidence intervals across time (optional when ``envelope = True``)

        :param int n: number of steps in each path
        :param int N: number of paths to simulate
        :param bool marginal:  defaults to True
        :param bool envelope:   defaults to False
        :param str type:   defaults to  '3sigma'
        :param str title:  to be used to customise plot title. If not passed, the title defaults to the name of the process.
        :return:
        """

        return self._draw_3sigmastyle(n=n, N=N, marginal=marginal, envelope=envelope, title=title, **fig_kw)


class RandomWalk(GeneralRandomWalk):

    def __init__(self, rng=None):
        super().__init__(p=0.5, rng=rng)
        self.name = 'Random Walk'

    def __str__(self):
        return f'Random Walk with step sizes {self.step_sizes} and probabilities {self.probs}'

    def __repr__(self):
        return f'Random Walk (step_sizes={self.step_sizes}, probabilities={self.probs})'

# p = GeneralRandomWalk(p=0.7)
# p = RandomWalk()
# s = p.sample(n=10)
# sim = p.simulate(n=100, N=10)
# p.plot(n=100, N=10, figsize=(12, 10))
# p.plot(n=100, N=50, figsize=(12, 10), plot_style='steps')
# p.plot(n=100, N=50, figsize=(12, 10), plot_style='linear')
#
# p.draw(n=10, N=50, figsize=(14, 10), envelope=False)
# p.draw(n=10, N=50, figsize=(14, 10), envelope=True)
# p.draw(n=300, N=200, figsize=(14, 10), envelope=True)
