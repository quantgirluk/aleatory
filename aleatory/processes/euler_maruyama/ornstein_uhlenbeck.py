"""
Ornstein-Uhlenbeck Process
"""

from aleatory.processes.euler_maruyama.vasicek import Vasicek


class OUProcess(Vasicek):
    r"""
    Ornstein–Uhlenbeck (OU) Process
    ================================

    .. image:: ../_static/ornstein–uhlenbeck_process_drawn.png

    Notes
    -----

    An Ornstein–Uhlenbeck process :math:`X = \{X : t \geq  0\}` is characterised by the following
    Stochastic Differential Equation

    .. math::

      dX_t = -\theta X_t dt + \sigma X_t dW_t, \ \ \ \ \forall t\in (0,T],

    with initial condition :math:`X_0 = x_0`, where

    - :math:`\theta` is the speed of reversion
    - :math:`\sigma>0` is the instantaneous volatility
    - :math:`W_t` is a standard Brownian Motion.

    Constructor, Methods, and Attributes
    ------------------------------------
    """

    def __init__(self, theta=1.0, sigma=0.5, initial=1.0, T=1.0, rng=None):
        """
        :param float theta: the parameter :math:`\theta` in the above SDE
        :param float sigma: the parameter :math:`\sigma>0` in the above SDE
        :param float initial: the initial condition :math:`x_0` in the above SDE
        :param float T: the right hand endpoint of the time interval :math:`[0,T]`
            for the process
        :param numpy.random.Generator rng: a custom random number generator
        """
        super().__init__(
            theta=theta, mu=0.0, sigma=sigma, initial=initial, T=T, rng=rng
        )
        self.n = 1.0
        self.dt = 1.0 * self.T / self.n
        self.times = None
        self.name = f"Ornstein–Uhlenbeck process $X(\\theta={self.theta}, \\sigma={self.sigma})$ on $[0,{self.T}]$"

    def __str__(self):
        return "Ornstein–Uhlenbeck process with parameters {speed}, and {volatility} on [0, {T}].".format(
            T=str(self.T), speed=str(self.theta), volatility=str(self.sigma)
        )


# if __name__ == "__main__":
#
#     import matplotlib.pyplot as plt
#
#     qs = "https://raw.githubusercontent.com/quantgirluk/matplotlib-stylesheets/main/quant-pastel-light.mplstyle"
#     plt.style.use(qs)
#
#     p = OUProcess()
#     p.plot(n=200, N=5, figsize=(12, 7), style=qs)
#
#     for p, cm in [
#         (OUProcess(), "twilight"),
#         (
#             OUProcess(
#                 theta=5.0,
#                 sigma=1.0,
#                 initial=1.0,
#                 T=4.0,
#             ),
#             "PiYG",
#         ),
#         (
#             OUProcess(
#                 theta=1.0,
#                 sigma=1.0,
#                 initial=5.0,
#                 T=1.0,
#             ),
#             "viridis",
#         ),
#         (
#             OUProcess(
#                 theta=0.5,
#                 sigma=2.0,
#                 initial=3.0,
#                 T=5.0,
#             ),
#             "Blues",
#         ),
#     ]:
#         p.draw(n=400, N=200, figsize=(12, 7), style=qs, colormap=cm, envelope=True)
