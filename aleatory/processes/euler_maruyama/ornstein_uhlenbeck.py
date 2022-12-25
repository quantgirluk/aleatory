"""
Ornstein-Uhlenbeck Process
"""
from aleatory.processes.euler_maruyama.vasicek import Vasicek


class OUProcess(Vasicek):
    r"""
    Ornstein–Uhlenbeck Process

    .. image:: _static/ornstein–uhlenbeck_process_drawn.png

    An Ornstein–Uhlenbeck process :math:`X = \{X : t \geq  0\}` is characterised by the following
    Stochastic Differential Equation

    .. math::

      dX_t = -\theta X_t dt + \sigma X_t dW_t, \ \ \ \ \forall t\in (0,T],

    with initial condition :math:`X_0 = x_0`, where

    - :math:`\theta` is the speed of reversion
    - :math:`\sigma>0` is the instantaneous volatility
    - :math:`W_t` is a standard Brownian Motion.


    Each :math:`X_t` follows a normal distribution.


    :param float theta: the parameter :math:`\theta` in the above SDE
    :param float sigma: the parameter :math:`\sigma>0` in the above SDE
    :param float initial: the initial condition :math:`x_0` in the above SDE
    :param float T: the right hand endpoint of the time interval :math:`[0,T]`
        for the process
    :param numpy.random.Generator rng: a custom random number generator

    """

    def __init__(self, theta=1.0, sigma=0.5, initial=1.0, T=1.0, rng=None):
        super().__init__(theta=theta, mu=0.0, sigma=sigma, initial=initial, T=T, rng=rng)
        self.n = 1.0
        self.dt = 1.0 * self.T / self.n
        self.times = None
        self.name = "Ornstein–Uhlenbeck process"

    def __str__(self):
        return "Ornstein–Uhlenbeck process with parameters {speed}, and {volatility} on [0, {T}].".format(
            T=str(self.T), speed=str(self.theta), volatility=str(self.sigma))