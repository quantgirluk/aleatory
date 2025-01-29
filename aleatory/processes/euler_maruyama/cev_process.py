"""
Constant Elasticity Variance Process
"""

from aleatory.processes import BrownianMotion, Vasicek, GBM
from aleatory.processes.euler_maruyama.ckls_process_generic import CKLSProcessGeneric


class CEVProcess(CKLSProcessGeneric):
    r"""
    Constant Elasticity of Variance (CEV) process
    =============================================

    .. image:: ../_static/cev_process_drawn.png

    Notes
    -----

    A CEV process  :math:`X = \{X : t \geq  0\}` is characterised by the following
    Stochastic Differential Equation

    .. math::

      dX_t = \mu X_t dt + \sigma X_t^{\gamma} dW_t, \ \ \ \ \forall t\in (0,T],

    with initial condition :math:`X_0 = x_0`, where

    - :math:`\mu` is the drift
    - :math:`\sigma>0` is the scale of the volatility
    - :math:`\gamma\geq 0` is the elasticity term
    - :math:`W_t` is a standard Brownian Motion.

    Constructor, Methods, and Attributes
    ------------------------------------
    """

    def __new__(cls, *args, **kwargs):
        r"""
        :param float mu: the parameter :math:`\mu` in the above SDE
        :param float sigma: the parameter :math:`\sigma>0` in the above SDE
        :param float gamma: the parameter :math:`\gamma` in the above SDE
        :param float initial: the initial condition :math:`x_0` in the above SDE
        :param float T: the right hand endpoint of the time interval :math:`[0,T]`
            for the process
        :param numpy.random.Generator rng: a custom random number generator
        """
        mu = kwargs["mu"] if "mu" in kwargs else 0.5
        gamma = kwargs["gamma"] if "gamma" in kwargs else 1.5
        sigma = kwargs["sigma"] if "sigma" in kwargs else 0.1
        initial = kwargs["initial"] if "initial" in kwargs else 1.0
        T = kwargs["T"] if "T" in kwargs else 1.0
        rng = kwargs["rng"] if "rng" in kwargs else None
        if mu == 0.0 and gamma == 0.0:
            return BrownianMotion(scale=sigma, T=T, rng=rng)
        elif gamma == 0 and mu < 0:
            theta = -1.0 * mu
            return Vasicek(
                theta=theta, mu=0.0, sigma=sigma, initial=initial, T=T, rng=rng
            )
        elif gamma == 1.0:
            return GBM(drift=mu, volatility=sigma, initial=initial, T=T, rng=rng)
        else:
            return CKLSProcessGeneric(
                alpha=0.0,
                beta=mu,
                sigma=sigma,
                gamma=gamma,
                initial=initial,
                T=T,
                rng=rng,
            )

    def __str__(self):
        return "CEV process with parameters gamma={gamma}, drift={drift}, and vol={volatility} on [0, {T}].".format(
            T=str(self.T),
            gamma=str(self.gamma),
            drift=str(self.mu),
            volatility=str(self.sigma),
        )


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#
#     qs = "https://raw.githubusercontent.com/quantgirluk/matplotlib-stylesheets/main/quant-pastel-light.mplstyle"
#     plt.style.use(qs)
#
#     p1 = CEVProcess()
#     p2 = CEVProcess(mu=1.0, gamma=1.0, sigma=0.5, initial=2.0, T=1.0)
#     p3 = CEVProcess(mu=1.0, gamma=0.75, sigma=1.0, initial=2.0, T=1.0)
#     p4 = CEVProcess(mu=1.0, gamma=1.25, sigma=0.5, initial=3.0, T=1.0)
#     p5 = CEVProcess(mu=-1.0, gamma=2.0, sigma=0.25, initial=3.0, T=0.25)
#     p6 = CEVProcess(mu=-1.0, gamma=0.0, sigma=0.25, initial=3.0, T=1.0)
#     for p, cm in [
#         (p1, "twilight"),
#         (p2, "PuBuGn"),
#         (p3, "copper"),
#         (p4, "RdBu"),
#         (p5, "Purples"),
#         (p6, "Oranges"),
#     ]:
#
#         p.draw(
#             n=500,
#             N=300,
#             figsize=(12, 7),
#             style=qs,
#             colormap=cm,
#             envelope=False,
#         )
#
#     p1.plot(n=500, N=10, figsize=(12, 7), style=qs)
#     p1.draw(n=500, N=300, figsize=(12, 7), style=qs, envelope=True)
