"""
General Random Walk
"""

from aleatory.processes.base_analytical import SPAnalytical
from abc import ABC
from aleatory.utils.utils import (
    check_positive_integer,
    get_times,
)
from aleatory.utils.plotters import plot_paths_random_walk
import numpy as np


class GeneralRandomWalk(SPAnalytical, ABC):
    r"""
    General Random Walk
    ===================

    .. image:: ../_static/gen_random_walk_draw.png

    Notes
    -----

    Let :math:`\{Z_i, i \geq 1\}` be a sequence of real-valued independent an identically
    distributed (i.i.d.) random variables defined on a probability
    space :math:`(\Omega, \mathcal{F}, \mathbb{P})`. Then, the stochastic process :math:`\{X_n , n\geq 0\}`,
    defined as :math:`X_0 =0`, and

    .. math::

        X_n = \sum_{i=1}^n Z_i, \qquad \forall n\geq 1,

    is called random walk, or more precisely one-dimensional random walked based on :math:`\{Z_i, i \geq 1\}`.

    Constructor, Methods, and Attributes
    ------------------------------------
    """

    def __init__(
        self,
        step_dist=None,
        step_args=None,
        step_kwargs=None,
        normalised=False,
        rng=None,
    ):
        """
        :parameter step_dist: an object representing the random variable :math:`Z_i` above (e.g.scipy.stats.norm)
        :parameter step_args: arguments (if any) to pass to the chosen step distribution
        :parameter step_kwargs: keyword arguments (if any) to pass to the chosen step distribution
        :parameter bool normalised: normalised or not
        :parameter numpy.random.Generator rng: a custom random number generator
        """
        super().__init__(rng=rng)
        self.step_dist = step_dist
        self.step_args = step_args if step_args is not None else tuple()
        self.step_kwargs = step_kwargs if step_kwargs is not None else dict()
        self.mean = self.step_dist.mean()
        self.std = self.step_dist.std()
        self.normalised = normalised
        self.paths = None
        self.n = None
        self.N = None
        self.name = "General Random Walk"
        self.times = None

    def __str__(self):
        return f"General Random Walk"

    def __repr__(self):
        return f"GeneralRandomWalk"

    def _sample_random_walk_steps(self, n):
        """Generate a sample of a general random walk increments"""
        check_positive_integer(n)
        steps = self.step_dist.rvs(size=n)

        if self.normalised:
            steps = (steps - self.mean) / self.std
        # steps = self.rng.choice(self.step_sizes, p=self.probs, size=n)
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


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from scipy.stats import norm, expon, binom

    qs = "https://raw.githubusercontent.com/quantgirluk/matplotlib-stylesheets/main/quant-pastel-light.mplstyle"
    plt.style.use(qs)

    p = GeneralRandomWalk(step_dist=norm, normalised=True)
    p.plot(
        n=10,
        N=20,
        figsize=(12, 7),
        title="Random Walk with Normalised Gaussian Steps",
        mode="steps+points",
        style=qs,
    )
    p.draw(
        n=100,
        N=200,
        figsize=(12, 7),
        title="Random Walk with Normalised Gaussian Steps",
        style=qs,
    )

    p = GeneralRandomWalk(step_dist=expon, normalised=True)
    p.plot(
        n=50,
        N=200,
        figsize=(12, 7),
        title="Random Walk with Normalised Exponential Steps",
        mode="steps+points",
        style=qs,
    )
    # p.draw(
    #     n=100,
    #     N=200,
    #     figsize=(12, 7),
    #     title="Random Walk with Normalised Exponential Steps",
    #     style=qs,
    #     colormap="viridis",
    # )
    #
    # p = GeneralRandomWalk(step_dist=binom(n=10, p=0.2), normalised=True)
    # p.plot(
    #     n=10,
    #     N=200,
    #     figsize=(12, 7),
    #     title="Random Walk Normalised Binomial Steps",
    #     mode="steps+points",
    #     style=qs,
    # )
    # p.draw(
    #     n=20,
    #     N=200,
    #     figsize=(12, 7),
    #     title="Random Walk Normalised Binomial Steps",
    #     style=qs,
    #     colormap="magma",
    # )
