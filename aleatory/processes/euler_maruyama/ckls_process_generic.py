"""
Chan-Karolyi-Longstaff-Sanders (CKLS) process generic
"""
from aleatory.processes.base_eu import SPEulerMaruyama
import numpy as np
from aleatory.utils.utils import draw_paths


class CKLSProcessGeneric(SPEulerMaruyama):
    r"""
    Chan-Karolyi-Longstaff-Sanders (CKLS) process

    .. image:: _static/ckls_process_draw.png


    A CKLS process  :math:`X = \{X : t \geq  0\}` is characterised by the following
    Stochastic Differential Equation

    .. math::

      dX_t = (\alpha  + \beta X_t) dt + \sigma X_t^{\gamma} dW_t, \ \ \ \ \forall t\in (0,T],

    with initial condition :math:`X_0 = x_0`, where

    - :math:`\alpha \in \mathbb{R}`
    - :math:`\beata \in \mathbb{R}`
    - :math:`\sigma>0` is the scale of the volatility
    - :math:`\gamma\geq 0` is the elasticity term
    - :math:`W_t` is a standard Brownian Motion.

    Reference: CHAN, K.C., KAROLYI, G.A., LONGSTAFF, F.A. and SANDERS, A.B. (1992),
    An Empirical Comparison of Alternative Models of the Short-Term Interest Rate. The Journal of Finance,
    47: 1209-1227. https://doi.org/10.1111/j.1540-6261.1992.tb04011.x

    :param float alpha: the parameter :math:`\alpha` in the above SDE
    :param float beta: the parameter :math:`\beta` in the above SDE
    :param float sigma: the parameter :math:`\sigma>0` in the above SDE
    :param float gamma: the parameter :math:`\gamma` in the above SDE
    :param float initial: the initial condition :math:`x_0` in the above SDE
    :param float T: the right hand endpoint of the time interval :math:`[0,T]`
        for the process
    :param numpy.random.Generator rng: a custom random number generator
    """

    def __init__(self, alpha=0.5, beta=0.5, sigma=0.1, gamma=1.0, initial=1.0, T=1.0, rng=None):
        super().__init__(T=T, rng=rng)
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.initial = initial
        self.n = 1.0
        self.dt = 1.0 * self.T / self.n
        self.times = np.arange(0.0, self.T + self.dt, self.dt)
        self.name = "Constant Elasticity Variance (CEV) process" if alpha==0.0 else "Chan-Karolyi-Longstaff-Sanders (CKLS) Process"
        self.paths = None
        self._marginals = None

        def f(x, _):
            return self.beta + self.alpha * np.exp(-1.0 * x) - 0.5 * (self.sigma ** 2) * np.exp(
                2.0 * (self.gamma - 1.0) * x)

        def g(x, _):
            return self.sigma * np.exp((self.gamma - 1.0) * x)

        self.f = f
        self.g = g

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = value

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        if value < 0:
            raise ValueError("sigma cannot be negative")
        self._sigma = value

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        if value < 0:
            raise ValueError("gamma cannot be negative")
        self._gamma = value

    def __str__(self):
        if self.alpha == 0.0:
            return ("CKLS process with parameters alpha={alpha}, beta={beta},sigma={sigma}, gamma={gamma}, "
                    "and initial={initial} on [0, {T}]. Note that this is a CEV process.").format(
                T=str(self.T), gamma=str(self.gamma), alpha=str(self.alpha), beta=str(self.beta), sigma=str(self.sigma),
                initial=str(self.initial))

        return ("CKLS process with parameters alpha={alpha}, beta={beta},sigma={sigma}, gamma={gamma}, "
             "and initial={initial} on [0, {T}].").format(
                T=str(self.T), gamma=str(self.gamma), alpha=str(self.alpha), beta=str(self.beta), sigma=str(self.sigma),
                initial=str(self.initial))


    def draw(self, n, N, marginal=True, envelope=False, title=None, **fig_kw):
        self.simulate(n, N)
        expectations = self.estimate_expectations()

        if envelope:
            lower = self.estimate_quantiles(0.005)
            upper = self.estimate_quantiles(0.995)
        else:
            lower = None
            upper = None

        chart_title = title if title else self.name

        if marginal:
            fig = draw_paths(times=self.times, paths=self.paths, N=N, title=chart_title, KDE=True, marginal=marginal,
                             expectations=expectations, envelope=envelope, lower=lower, upper=upper,
                             **fig_kw)
        else:
            fig = draw_paths(times=self.times, paths=self.paths, N=N, title=chart_title,
                             expectations=expectations, marginal=marginal, **fig_kw)

        return fig

    def sample(self, n):
        return self._sample_em_process(n, log=True)
