"""
Random Walk
"""

from aleatory.processes.base_analytical import SPAnalytical
from abc import ABC
from aleatory.utils.utils import (
    check_positive_integer,
    get_times,
)
from aleatory.utils.plotters import plot_paths_random_walk
import numpy as np


class SimpleRandomWalk(SPAnalytical, ABC):

    def __init__(self, p=0.5, rng=None):
        super().__init__(rng=rng)
        self.step_sizes = (1.0, -1.0)
        self.p = p
        self.q = 1.0 - p
        self.probs = (self.p, self.q)
        self.paths = None
        self.n = None
        self.N = None
        if p == 0.5:
            self.name = "Simple Random Walk"
        else:
            self.name = f"Simple Random Walk with p={self.p}"
        self.times = None

    def __str__(self):
        return f"General Random Walk with step sizes {self.step_sizes} and probabilities {self.probs}"

    def __repr__(self):
        return f"GeneralRandomWalk(step_sizes={self.step_sizes}, probabilities={self.probs})"

    def _sample_random_walk_steps(self, n):
        """Generate a sample of a general random walk increments"""
        check_positive_integer(n)
        steps = self.rng.choice(self.step_sizes, p=self.probs, size=n)
        return steps

    def _sample_random_walk(self, n):
        """Generate a sample from a general random walk"""
        self.T = n
        self.n = n
        sample = np.array([0] + list(np.cumsum(self._sample_random_walk_steps(n))))
        return sample

    def sample(self, n):
        sample = self._sample_random_walk(n)
        self.times = get_times(self.T, self.n + 1)
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
            figure = plot_paths_random_walk(
                *args, times=self.times, paths=self.paths, title=title, **fig_kw
            )
        else:
            figure = plot_paths_random_walk(
                *args, times=self.times, paths=self.paths, title=self.name, **fig_kw
            )
        return figure

    def draw(
        self, n, N, marginal=True, envelope=False, mode="steps", title=None, **fig_kw
    ):
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
        :param str mode: defaults to 'steps'
        :param str title:  to be used to customise plot title. If not passed, the title defaults to the name of the process.
        :return:
        """

        return self._draw_3sigmastyle(
            n=n,
            N=N,
            marginal=marginal,
            envelope=envelope,
            title=title,
            mode=mode,
            **fig_kw,
        )


class RandomWalk(SimpleRandomWalk):
    r"""
    Simple Random Walk
    ==================

    .. image:: ../_static/simple_random_walk_draw.png

    Notes
    -----

    Let :math:`\{Z_i, i \geq 1\}` be a sequence of real-valued independent an identically
    distributed (i.i.d.) random variables defined on a probability
    space :math:`(\Omega, \mathcal{F}, \mathbb{P})`, such that

    .. math::
        \mathbb{P}(Z_1 = 1) = p,

    and

    .. math::
        \mathbb{P}(Z_1 = -1) = 1-p,


    Then, the stochastic process :math:`\{X_n , n\geq 0\}`,
    defined as :math:`X_0 =0`, and

    .. math::

        X_n = \sum_{i=1}^n Z_i, \qquad \forall n\geq 1,


    is called a Simple Random Walk.

    Constructor, Methods, and Attributes
    ------------------------------------

    """

    def __init__(self, rng=None):
        """
        :parameter numpy.random.Generator rng: a custom random number generator
        """

        super().__init__(p=0.5, rng=rng)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    qs = "https://raw.githubusercontent.com/quantgirluk/matplotlib-stylesheets/main/quant-pastel-light.mplstyle"
    plt.style.use(qs)

    for prob in [0.25, 0.5, 0.75]:
        p = SimpleRandomWalk(p=prob)

        p.plot(
            n=10,
            N=10,
            figsize=(12, 7),
            mode="steps+points",
            style=qs,
        )
        p.draw(
            n=100,
            N=200,
            figsize=(12, 7),
            style=qs,
        )
