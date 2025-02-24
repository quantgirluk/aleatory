from aleatory.processes.base_eu import SPEulerMaruyama
import numpy as np
from scipy.stats import ncx2


class CIRProcess(SPEulerMaruyama):
    r"""
    Cox–Ingersoll–Ross (CIR) Process
    ================================

    .. image:: ../_static/cir_process_drawn.png

    Notes
    -----

    A Cox–Ingersoll–Ross process :math:`X = \{X : t \geq  0\}` is characterised by the following
    Stochastic Differential Equation

    .. math::

      dX_t = \theta(\mu - X_t) dt + \sigma \sqrt{X_t} dW_t, \ \ \ \ \forall t\in (0,T],

    with initial condition :math:`X_0 = x_0`, where

    - :math:`\theta` is the rate of mean reversion
    - :math:`\mu` is the long term mean value.
    - :math:`\sigma>0` is the instantaneous volatility
    - :math:`W_t` is a standard Brownian Motion.


    It can be seen that each :math:`X_t` follows a non-central chi-square distribution.

    Constructor, Methods, and Attributes
    ------------------------------------

    """

    def __init__(self, theta=1.0, mu=2.0, sigma=0.5, initial=5.0, T=1.0, rng=None):
        r"""
        :param float theta: the parameter :math:`\theta` in the above SDE
        :param float mu: the parameter :math:`\mu` in the above SDE
        :param float sigma: the parameter :math:`\sigma>0` in the above SDE
        :param float initial: the initial condition :math:`x_0` in the above SDE
        :param float T: the right hand endpoint of the time interval :math:`[0,T]`
            for the process
        :param numpy.random.Generator rng: a custom random number generator
        """

        super().__init__(T=T, rng=rng, initial=initial)
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.n = 1.0
        self.dt = 1.0 * self.T / self.n
        self.times = np.arange(0.0, self.T + self.dt, self.dt)
        self.name = (
            f"CIR Process $X(\\theta={self.theta}, \\mu={self.mu}, \\sigma={self.sigma})$"
            f"\n starting at $x_0=${self.initial}"
        )

        def f(x, _):
            # return self.theta * (self.mu - x)
            return np.exp(-x) * (
                self.theta * (self.mu - np.exp(x)) - 0.5 * self.sigma**2
            )

        def g(x, _):
            # return self.sigma * np.sqrt(x)
            return self.sigma * np.exp(-0.5 * x)

        self.f = f
        self.g = g

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        if value <= 0:
            raise ValueError("theta must be positive")
        self._theta = value

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        if value <= 0:
            raise ValueError("mu must be positive")
        self._mu = value

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        if value <= 0:
            raise ValueError("sigma has to be positive")
        if 2 * self.theta * self.mu <= value**2:
            raise ValueError("Condition 2*theta*mu >= sigma**2 must be satisfied")
        self._sigma = value

    def __str__(self):
        return "Cox–Ingersoll–Ross process with parameters {speed}, {mean}, and {volatility} on [0, {T}].".format(
            T=str(self.T),
            speed=str(self.theta),
            mean=str(self.mu),
            volatility=str(self.sigma),
        )

    def _process_expectation(self, times=None):
        if times is None:
            times = self.times
        expectations = self.initial * np.exp((-1.0) * self.theta * times) + self.mu * (
            np.ones(len(times)) - np.exp((-1.0) * self.theta * times)
        )
        return expectations

    def marginal_expectation(self, times=None):
        expectations = self._process_expectation(times=times)
        return expectations

    def _process_variance(self, times=None):
        if times is None:
            times = self.times
        variances = (self.sigma**2 / self.theta) * self.initial * (
            np.exp(-1.0 * self.theta * times) - np.exp(-2.0 * self.theta * times)
        ) + (self.mu * self.sigma**2 / (2 * self.theta)) * (
            (np.ones(len(times)) - np.exp(-1.0 * self.theta * times)) ** 2
        )
        return variances

    def _process_degrees_of_freedom(self):
        df = 4.0 * self.theta * self.mu / (self.sigma**2)

        return df

    def marginal_df(self):

        return self._process_degrees_of_freedom()

    def _process_nc_parameter(self, times=None):
        if times is None:
            times = self.times
        c = (
            2.0
            * self.theta
            / ((1.0 - np.exp(-1.0 * self.theta * times)) * self.sigma**2)
        )
        ncs = 2.0 * c * self.initial * np.exp(-1.0 * self.theta * times)

        return ncs

    def marginal_nc_parameter(self, times=None):
        ncs = self._process_nc_parameter(times=times)

        return ncs

    def _process_scales(self, times=None):
        if times is None:
            times = self.times
        c = (
            2.0
            * self.theta
            / ((1.0 - np.exp(-1.0 * self.theta * times)) * self.sigma**2)
        )
        scales = 1.0 / (2.0 * c)

        return scales

    def marginal_scale(self, times=None):
        scales = self._process_scales(times=times)

        return scales

    def marginal_variance(self, times=None):
        variances = self._process_variance(times=times)
        return variances

    def _process_stds(self):
        stds = np.sqrt(self._process_variance())
        return stds

    def process_stds(self):
        stds = self._process_stds()
        return stds

    def get_marginal(self, t):
        a = self.theta
        b = self.mu
        sigma = self.sigma

        c = 2.0 * a / ((1.0 - np.exp(-1.0 * a * t)) * sigma**2)
        df = 4.0 * a * b / sigma**2
        nc = 2.0 * c * self.initial * np.exp(-1.0 * a * t)
        scale = 1.0 / (2 * c)
        marginal = ncx2(df, nc, scale=scale)

        return marginal

    def sample(self, n):
        return self._sample_em_process(n, log=True)


# if __name__ == "__main__":
#
#     import matplotlib.pyplot as plt
#
#     qs = "https://raw.githubusercontent.com/quantgirluk/matplotlib-stylesheets/main/quant-pastel-light.mplstyle"
#     plt.style.use(qs)
#
#     p1 = CIRProcess()
#     p2 = CIRProcess(theta=1.0, mu=2.0, sigma=1.0, initial=7.0, T=5.0)
#     p3 = CIRProcess(theta=1.0, mu=10.0, sigma=2.0, initial=1.0, T=10.0)
#     p4 = CIRProcess(theta=1.0, mu=10.0, sigma=0.2, initial=1.0, T=1.0)
#
#     for p, cm in [
#         (p1, "terrain"),
#         (p2, "RdPu"),
#         (p3, "Oranges"),
#         (p4, "Blues"),
#     ]:
#
#         p.draw(
#             n=500,
#             N=300,
#             figsize=(12, 7),
#             style=qs,
#             colormap=cm,
#             envelope=True,
#         )
#
#     p1.plot(n=500, N=10, figsize=(12, 7), style=qs)
#     p1.draw(n=500, N=300, figsize=(12, 7), style=qs, envelope=True)
