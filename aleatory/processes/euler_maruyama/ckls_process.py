"""Chan-Karolyi-Longstaff-Sanders (CKLS) process"""

from aleatory.processes.euler_maruyama.ckls_process_generic import CKLSProcessGeneric
from aleatory.processes import BrownianMotion, Vasicek, CIRProcess, GBM


class CKLSProcess(CKLSProcessGeneric):
    r"""
    Chan-Karolyi-Longstaff-Sanders (CKLS) process
    =============================================

    .. image:: ../_static/ckls_process_draw.png

    Notes
    -----

    A CKLS process  :math:`X = \{X : t \geq  0\}` is characterised by the following
    Stochastic Differential Equation

    .. math::

      dX_t = (\alpha  + \beta X_t) dt + \sigma X_t^{\gamma} dW_t, \ \ \ \ \forall t\in (0,T],

    with initial condition :math:`X_0 = x_0`, where

    - :math:`\alpha \in \mathbb{R}`
    - :math:`\beta \in \mathbb{R}`
    - :math:`\sigma>0` is the scale of the volatility
    - :math:`\gamma\geq 0` is the elasticity term
    - :math:`W_t` is a standard Brownian Motion.

    References
    ----------

    - CHAN, K.C., KAROLYI, G.A., LONGSTAFF, F.A. and SANDERS, A.B. (1992),
    An Empirical Comparison of Alternative Models of the Short-Term Interest Rate. The Journal of Finance,
    47: 1209-1227. https://doi.org/10.1111/j.1540-6261.1992.tb04011.x

    Constructor, Methods, and Attributes
    ------------------------------------

    """

    def __new__(cls, *args, **kwargs):
        alpha = kwargs["alpha"] if "alpha" in kwargs else 0.5
        beta = kwargs["beta"] if "beta" in kwargs else 0.5
        sigma = kwargs["sigma"] if "sigma" in kwargs else 0.1
        gamma = kwargs["gamma"] if "gamma" in kwargs else 1.5
        initial = kwargs["initial"] if "initial" in kwargs else 1.0
        T = kwargs["T"] if "T" in kwargs else 1.0
        rng = kwargs["rng"] if "rng" in kwargs else None
        if beta == 0.0 and gamma == 0:
            return BrownianMotion(drift=alpha, scale=sigma, T=T, rng=rng)
        elif gamma == 0 and beta < 0:
            theta = -1.0 * beta
            mu = -1.0 * alpha / beta
            return Vasicek(
                theta=theta, mu=mu, sigma=sigma, initial=initial, T=T, rng=rng
            )
        elif gamma == 0.5:
            theta = -1.0 * beta
            mu = -1.0 * alpha / beta
            return CIRProcess(
                theta=theta, mu=mu, sigma=sigma, initial=initial, T=T, rng=rng
            )
        elif alpha == 0.0 and gamma == 1.0:
            return GBM(drift=beta, volatility=sigma, initial=initial, T=T, rng=rng)
        else:
            return CKLSProcessGeneric(
                alpha=alpha,
                beta=beta,
                sigma=sigma,
                gamma=gamma,
                initial=initial,
                T=T,
                rng=rng,
            )
