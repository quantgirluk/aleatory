"""
Vasicek Process
"""
from aleatory.processes.base import SPEulerMaruyama
import numpy as np
from scipy.stats import norm


class Vasicek(SPEulerMaruyama):
    r"""
    Vasicek Process

    .. image:: _static/vasicek_process_drawn.png


    A Vasicek process :math:`X = \{X : t \geq  0\}` is characterised by the following
    Stochastic Differential Equation

    .. math::

      dX_t = \theta(\mu - X_t) dt + \sigma X_t dW_t, \ \ \ \ \forall t\in (0,T],

    with initial condition :math:`X_0 = x_0`, where

    - :math:`\theta` is the speed of reversion
    - :math:`\mu` is the long term mean value.
    - :math:`\sigma>0` is the instantaneous volatility
    - :math:`W_t` is a standard Brownian Motion.


    Each :math:`X_t` follows a normal distribution.


    :param float theta: the parameter :math:`\theta` in the above SDE
    :param float mu: the parameter :math:`\mu` in the above SDE
    :param float sigma: the parameter :math:`\sigma>0` in the above SDE
    :param float initial: the initial condition :math:`x_0` in the above SDE
    :param float T: the right hand endpoint of the time interval :math:`[0,T]`
        for the process
    :param numpy.random.Generator rng: a custom random number generator
    """

    def __init__(self, theta=1.0, mu=3.0, sigma=0.5, initial=1.0, T=1.0, rng=None):
        super().__init__(T=T, rng=rng)
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.initial = initial
        self.n = 1.0
        self.dt = 1.0 * self.T / self.n
        self.times = None
        self.name = "Vasicek Process"

        def f(x, _):
            return self.theta * (self.mu - x)

        def g(x, _):
            return self.sigma

        self.f = f
        self.g = g

    def __str__(self):
        return "Vasicek process with parameters {speed}, {mean}, and {volatility} on [0, {T}].".format(
            T=str(self.T), speed=str(self.theta), mean=str(self.mu), volatility=str(self.sigma))

    def _process_expectation(self):
        return self.initial * np.exp((-1.0) * self.theta * self.times) + self.mu * (
                np.ones(len(self.times)) - np.exp((-1.0) * self.theta * self.times))

    def process_expectation(self):
        expectations = self._process_expectation()
        return expectations

    def _process_variance(self):
        variances = (self.sigma ** 2) * (1.0 / (2.0 * self.theta)) * (
                np.ones(len(self.times)) - np.exp(-2.0 * self.theta * self.times))
        return variances

    def process_variance(self):
        variances = self._process_variance()
        return variances

    def _process_stds(self):
        stds = np.sqrt(self.process_variance())
        return stds

    def process_stds(self):
        stds = self._process_stds()
        return stds

    def get_marginal(self, t):
        mu_x = self.initial * np.exp(-1.0 * self.theta * t) + self.mu * (1.0 - np.exp(-1.0 * self.theta * t))
        variance_x = (self.sigma ** 2) * (1.0 / (2.0 * self.theta)) * (1.0 - np.exp(-2.0 * self.theta * t))
        sigma_x = np.sqrt(variance_x)
        marginal = norm(loc=mu_x, scale=sigma_x)

        return marginal
