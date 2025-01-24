"""
Brownian Bridge
"""

import numpy as np
from scipy.stats import norm

from aleatory.processes.analytical.brownian_motion import BrownianMotion
from aleatory.utils.utils import check_numeric
from aleatory.utils.plotters import draw_paths_with_end_point
from aleatory.utils.utils import check_positive_integer, get_times


class BrownianBridge(BrownianMotion):
    r"""
    Brownian Bridge
    ===============

    .. image:: ../_static/brownian_bridge_drawn.png

    Definition
    ----------

    A Brownian bridge is a continuous-time stochastic process  :math:`\{B_t : t \geq 0\}`
    whose probability distribution is the conditional probability distribution of a
    standard Wiener process (Brownian Motion)  :math:`\{W_t : t \geq 0\}`
    subject to the condition that :math:`W(T) = 0`, so that the process is pinned to
    the same value at both :math:`t = 0` and :math:`t = T`. More specifically,

    .. math::

        B_t = (W_t | W_T = 0), \ \ \ \   t\in (0,T].

    More generally, a Brownian Bridge is  subject to the conditions :math:`W(0) = a` and :math:`W(T) = b`.

    Constructor, Methods, and Attributes
    ------------------------------------
    """

    def __init__(self, initial=0.0, end=0.0, T=1.0, rng=None):
        """

        :param float initial: initial condition
        :param float end: end condition
        :param float T: the right hand endpoint of the time interval :math:`[0,T]` for the process
        :param numpy.random.Generator rng: a custom random number generator

        """
        super().__init__(T=T, rng=rng)
        self.initial = initial
        self.end = end
        self._brownian_motion = BrownianMotion(T=T, rng=rng)
        if self.initial == 0.0 and self.end == 0.0:
            self.name = "Brownian Bridge"
        else:
            self.name = f"Brownian Bridge from {self.initial} to {self.end}"
        self.n = None
        self.times = None

    def __str__(self):
        return (
            "Brownian Bridge starting at  {a} and ending at  {b} on [0, {T}].".format(
                T=str(self.T), a=str(self.initial), b=str(self.end)
            )
        )

    def __repr__(self):
        return "BrownianBridge(initial={a}, end={b}, T={T})".format(
            T=str(self.T), a=str(self.initial), b=str(self.end)
        )

    @property
    def initial(self):
        return self._initial

    @property
    def end(self):
        return self._end

    @initial.setter
    def initial(self, value):
        check_numeric(value, "Initial Point")
        self._initial = value

    @end.setter
    def end(self, value):
        check_numeric(value, "End Point")
        self._end = value

    def _sample_brownian_bridge(self, n):
        """Generate a realization of a Brownian Bridge."""
        check_positive_integer(n)
        self.n = n
        self.times = get_times(self.T, n)
        brownian_path = self._brownian_motion.sample(n)
        a = self.initial
        b = self.end
        scaled_times = self.times / self.T
        bridge_path = (
            a * (1.0 - scaled_times)
            + b * scaled_times
            + brownian_path
            - scaled_times * brownian_path[-1]
        )
        return bridge_path

    def _sample_brownian_bridge_at(self, times):
        self.times = times
        brownian_path = self._brownian_motion.sample_at(times)
        a = self.initial
        b = self.end
        scaled_times = self.times / self.T
        bridge_path = (
            a * (1.0 - scaled_times)
            + b * scaled_times
            + (brownian_path - scaled_times * brownian_path[-1])
        )
        return bridge_path

    def sample(self, n):
        return self._sample_brownian_bridge(n)

    def sample_at(self, times):
        return self._sample_brownian_bridge_at(times)

    def _process_expectation(self, times=None):
        if times is None:
            times = self.times
        scaled_times = times / self.T
        return self.initial * (1.0 - scaled_times) + self.end * scaled_times

    def _process_variance(self, times=None):
        if times is None:
            times = self.times
        scaled_times = times / self.T
        return (self.T - times) * scaled_times

    def get_marginal(self, t):
        scaled_time = t / self.T
        mean = self.initial * (1.0 - scaled_time) + self.end * scaled_time
        var = (self.T - t) * scaled_time
        marginal = norm(loc=mean, scale=np.sqrt(var))
        return marginal

    def _draw_paths(self, n, N, envelope=False, type=None, title=None, **fig_kw):
        self.simulate(n, N)
        expectations = self._process_expectation()

        if envelope:
            marginals = [self.get_marginal(t) for t in self.times[1:-1]]
            upper = [self.initial] + [m.ppf(0.005) for m in marginals] + [self.end]
            lower = [self.initial] + [m.ppf(0.995) for m in marginals] + [self.end]
        else:
            upper = None
            lower = None

        chart_title = title if title else self.name
        if "marginal" in fig_kw:
            fig_kw.pop("marginal")
        if "orientation" in fig_kw:
            fig_kw.pop("orientation")
        fig = draw_paths_with_end_point(
            times=self.times,
            paths=self.paths,
            expectations=expectations,
            title=chart_title,
            envelope=envelope,
            lower=lower,
            upper=upper,
            **fig_kw,
        )
        return fig

    def draw(self, n, N, envelope=False, title=None, **fig_kw):
        """
        Simulates and plots paths/trajectories from the instanced stochastic process.

        Produces different kind of visualisation illustrating the following elements:

        - times versus process values as lines
        - the expectation of the process across time
        - envelope of confidence intervals across time (optional when ``envelope = True``)

        :param n: number of steps in each path
        :param N: number of paths to simulate
        :param envelope: bool, default: False
        :param title: string to customise plot title
        :return:
        """

        return self._draw_paths(n, N, envelope=envelope, title=title, **fig_kw)


# if __name__ == "__main__":
#
#     import matplotlib.pyplot as plt
#
#     qs = "https://raw.githubusercontent.com/quantgirluk/matplotlib-stylesheets/main/quant-pastel-light.mplstyle"
#     plt.style.use(qs)
#
#     for p, cm in [
#         (BrownianBridge(), "PRGn"),
#         (BrownianBridge(initial=10.0, end=11.0), "viridis"),
#         (BrownianBridge(initial=1.0, end=-1.0), "PiYG"),
#     ]:
#         # p.plot(n=100, N=5, figsize=(10, 7), style=qs)
#         p.draw(n=200, N=200, figsize=(10, 7), style=qs, colormap=cm)
