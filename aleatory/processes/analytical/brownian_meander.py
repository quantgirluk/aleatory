"""
Brownian Meander
"""
import numpy as np

from aleatory.processes import BrownianBridge, BrownianMotion
from aleatory.utils.utils import check_positive_integer, draw_paths_with_end_point, draw_paths


class BrownianMeander(BrownianMotion):
    r"""
    Brownian Meander

    .. image:: _static/brownian_meander_drawn.png
    .. image:: _static/tied_brownian_meander_drawn.png

    Let :math:`\{W_t : t \geq 0\}` be a standard Brownian motion and

    .. math::
        \tau = \sup\{ t \in [0,1] : W_t =0\},

    i.e. the last time before t = 1 when :math:`W_t` visits zero. Then the
    Brownian Meander is defined as follows

    .. math::
        W_t^{+} = \frac{1}{\sqrt{1-\tau}} |W(\tau  + t (1-\tau))|, \ \ \ \   t\in (0,1].

    :param float T: the right hand endpoint of the time interval :math:`[0,T]`
        for the process
    :param bool fixed_end: flag to indicate if the process has a fixed end point. Defaults to `False`
    :param float end: end point for the Meander, in the case of `fixed_end` equal `True`
    :param numpy.random.Generator rng: a custom random number generator

    """

    def __init__(self, T=1.0, fixed_end=False, end=None, rng=None):
        super().__init__(T=T, rng=rng)
        self.name = "Tied Brownian Meander" if fixed_end else "Brownian Meander"
        self.fixed_end = fixed_end
        self.end = end
        self._BrownianBridge = BrownianBridge(T=self.T)
        self.n = None
        self.times = None

    def __str__(self):
        return f"Brownian Meander"

    def __repr__(self):
        return f"BrownianMeander"

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, value):
        if value is None:
            self.exponential = self.rng.exponential(1)
            self._end = np.sqrt(2.0 * self.T * self.rng.exponential(1))
        elif value < 0:
            raise ValueError("end point cannot be negative")
        else:
            self._end = True
            self._end = value

    def _sample_brownian_meander(self, n):
        """
        Generate a realization from Brownian Meander
        """
        check_positive_integer(n)
        self.n = n
        self.times = np.linspace(0, 1, n)
        bridges = self._BrownianBridge.simulate(n=n, N=3)
        end = self.end if self.fixed_end else np.sqrt(2.0 * self.T * self.rng.exponential(1))
        times_scaled = self.times / self.T
        meander = np.sqrt((end * times_scaled + bridges[0]) ** 2 + bridges[1] ** 2 + bridges[2] ** 2)
        return meander

    def _sample_brownian_meander_at(self, times):
        self.times = times
        bridges = self._BrownianBridge.sample_at(times)
        end = np.sqrt(2.0 * self.T * self.rng.exponential(1))
        times_scaled = self.times / self.T
        meander = np.sqrt((end * times_scaled + bridges[0]) ** 2 + bridges[1] ** 2 + bridges[2] ** 2)
        return meander

    def sample(self, n):
        return self._sample_brownian_meander(n)

    def sample_at(self, times):
        return self._sample_brownian_meander_at(times)

    def _draw_paths(self, n, N, title=None, **fig_kw):
        self.simulate(n, N)
        chart_title = title if title else self.name
        fig_kw['envelope'] = False

        if self.fixed_end:
            if 'marginal' in fig_kw:
                fig_kw.pop('marginal')
            if 'orientation' in fig_kw:
                fig_kw.pop('orientation')
            fig = draw_paths_with_end_point(times=self.times, paths=self.paths,
                                            title=chart_title,
                                            **fig_kw)
        else:
            fig = draw_paths(times=self.times, paths=self.paths, N=N, expectations=None, KDE=False, title=chart_title,
                             **fig_kw)

        return fig

    def draw(self, n, N, title=None, **fig_kw):
        """
        Simulates and plots paths/trajectories from the instanced stochastic process.

        Produces different kind of visualisation illustrating the following elements:

        - times versus process values as lines
        - the expectation of the process across time
        - histogram showing the empirical marginal distribution :math:`X_T` (optional when ``marginal = True``)
        - probability density function of the marginal distribution :math:`X_T` (optional when ``marginal = True``)
        - envelope of confidence intervals across time (optional when ``envelope = True``)


        :param n: number of steps in each path
        :param N: number of paths to simulate
        :param title: string to customise plot title
        :return:
        """
        return self._draw_paths(n, N, title, **fig_kw)
