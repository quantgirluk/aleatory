"""Brownian Motion"""

import numpy as np
from scipy.stats import norm

from aleatory.processes.base_analytical import SPAnalytical
from aleatory.processes.analytical.gaussian import GaussianIncrements
from aleatory.utils.utils import check_positive_number, check_numeric, get_times


class BrownianMotion(SPAnalytical):
    r"""
    Brownian Motion
    ===============

    A one-dimensional standard Brownian motion object.

    .. image:: ../_static/brownian_motion_drawn.png

    Notes
    -----

    A standard Brownian motion :math:`\{W_t : t \geq 0\}` is defined by the following properties:

    1. Starts at zero, i.e. :math:`W(0) = 0`
    2. Independent increments
    3. :math:`W(t) - W(s)` follows a Gaussian distribution :math:`N(0, t-s)`
    4. Almost surely continuous

    A more general version of a Brownian motion, is the Arithmetic Brownian Motion which  is defined
    by the following SDE

    .. math::

        dX_t = \mu dt + \sigma dW_t \ \ \ \   t\in (0,T]


    with initial condition :math:`X_0 = x_0\in\mathbb{R}`, where

    - :math:`\mu` is the drift
    - :math:`\sigma>0` is the volatility
    - :math:`W_t` is a standard Brownian Motion

    Clearly, the solution to this equation can be written as

    .. math::

        X_t = x_0 +  \mu t + \sigma W_t \ \ \ \ t \in [0,T]

    and each :math:`X_t \sim N(\mu t, \sigma^2 t)`.

    Examples
    --------

    .. highlight:: python
    .. code-block:: python

        from aleatory.processes import BrownianMotion
        process = BrownianMotion()
        fig = process.plot(n=100, N=5, figsize=(12, 7))
        fig.show()


    .. code-block:: python

        from aleatory.processes import BrownianMotion
        process = BrownianMotion()
        fig = process.draw(n=100, N=100, figsize=(12, 7))
        fig.show()

    Constructor, Methods, and Attributes
    ------------------------------------

    """

    def __init__(self, drift=0.0, scale=1.0, initial=0.0, T=1.0, rng=None):
        """
        :param double drift: the drift parameter :math:`\mu` in the above SDE
        :param double scale: the scale parameter :math:`\sigma` in the above SDE
        :param double initial: the initial condition :math:`x_0` in the above SDE
        :param double T: the endpoint of the time interval :math:`[0,T]` over which the process is defined
        """
        super().__init__(T=T, rng=rng, initial=0.0)
        self.drift = drift
        self.scale = scale
        self.initial = initial
        standard_condition = drift == 0.0 and scale == 1.0 and initial == 0.0
        self.name = (
            "Brownian Motion" if standard_condition else "Arithmetic Brownian Motion"
        )
        self.n = None
        self.times = None
        self.gaussian_increments = GaussianIncrements(T=self.T, rng=self.rng)

    @property
    def drift(self):
        return self._drift

    @drift.setter
    def drift(self, value):
        check_numeric(value, "Drift")
        self._drift = value

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        check_positive_number(value, "Scale")
        self._scale = value

    @property
    def initial(self):
        return self._initial

    @initial.setter
    def initial(self, value):
        self._initial = value

    def _sample_brownian_motion(self, n):
        self.n = n
        self.times = get_times(self.T, self.n)
        # increments = np.random.normal(scale=np.sqrt(self.T/self.n), size=self.n)
        increments = np.cumsum(self.gaussian_increments.sample(n - 1))
        increments = np.insert(increments, 0, [0])

        path = (
            np.full(n, self.initial) + self.drift * self.times + self.scale * increments
        )
        # bm = np.cumsum(self.scale *self.gaussian_increments.sample(n - 1))
        # bm = np.insert(bm, 0, [0])
        # if self.drift == 0:
        #     return bm
        # else:
        # return self.initial + self.drift+self.times + bm
        return path

    def __str__(self):

        return (
            "Brownian Motion with drift={drift}, and scale={scale} on [0, {T}].".format(
                T=str(self.T),
                drift=str(self.drift),
                scale=str(self.scale),
                initial=str(self.initial),
            )
        )

    def sample(self, n):
        """
        Generates a discrete time sample from a Brownian Motion instance.

        :param int n: the number of steps
        :return: numpy array
        """
        return self._sample_brownian_motion(n)

    def _sample_brownian_motion_at(self, times):
        if times[0] != 0:
            times = np.insert(times, 0, [0])
        self.times = times

        increments = np.cumsum(self.gaussian_increments.sample_at(times))
        # increments = np.insert(increments, 0, [0])

        path = (
            np.full(len(self.times), self.initial)
            + self.drift * self.times
            + self.scale * increments
        )
        # bm = np.cumsum(self.scale * self.gaussian_increments.sample_at(times))

        #
        # if self.drift != 0:
        #     bm += [self.drift * t for t in times]

        return path

    def sample_at(self, times):
        """
        Generates a sample from a Brownian motion at the specified times.

        :param times: the times which define the sample
        :return: numpy array
        """
        return self._sample_brownian_motion_at(times)

    def _process_expectation(self, times=None):
        if times is None:
            times = self.times
        return self.initial + self.drift * times

    def _process_variance(self, times=None):
        if times is None:
            times = self.times
        return (self.scale**2) * times

    def _process_stds(self, times=None):
        if times is None:
            times = self.times
        return self.scale * np.sqrt(times)

    def process_stds(self):
        stds = self._process_stds()
        return stds

    def get_marginal(self, t):
        marginal = norm(
            loc=self.initial + self.drift * t, scale=self.scale * np.sqrt(t)
        )
        return marginal

    def marginal_expectation(self, times=None):
        expectations = self._process_expectation(times=times)
        return expectations

    def marginal_variance(self, times):
        variances = self._process_variance(times=times)
        return variances

    def draw(
        self, n, N, marginal=True, envelope=False, type="3sigma", title=None, **fig_kw
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
        :param str type:   defaults to  '3sigma'
        :param str title:  to be used to customise plot title. If not passed, the title defaults to the name of the process.
        :return:
        """

        if type == "3sigma":
            return self._draw_3sigmastyle(
                n=n, N=N, marginal=marginal, envelope=envelope, title=title, **fig_kw
            )
        elif type == "qq":
            return self._draw_qqstyle(
                n, N, marginal=marginal, envelope=envelope, title=title, **fig_kw
            )
        else:
            raise ValueError


if __name__ == "__main__":

    p = BrownianMotion()
    p.draw(n=20, N=100, figsize=(12, 7))
