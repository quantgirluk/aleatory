"""
Constant Elasticity Variance Process
"""
from aleatory.processes.base import SPEulerMaruyama
import numpy as np
from aleatory.utils.utils import draw_paths


class CEVProcess(SPEulerMaruyama):
    r"""
    CEV or constant elasticity of variance process

    .. image:: _static/cev_process_drawn.png


    A CEV process  :math:`X = \{X : t \geq  0\}` is characterised by the following
    Stochastic Differential Equation

    .. math::

      dX_t = \mu X_t dt + \sigma X_t^{\gamma} dW_t, \ \ \ \ \forall t\in (0,T],

    with initial condition :math:`X_0 = x_0`, where

    - :math:`\mu` is the drift
    - :math:`\sigma>0` is the scale of the volatility
    - :math:`\gamma\geq 0` is the elasticity term
    - :math:`W_t` is a standard Brownian Motion.


    :param float mu: the parameter :math:`\mu` in the above SDE
    :param float sigma: the parameter :math:`\sigma>0` in the above SDE
    :param float gamma: the parameter :math:`\gamma` in the above SDE
    :param float initial: the initial condition :math:`x_0` in the above SDE
    :param float T: the right hand endpoint of the time interval :math:`[0,T]`
        for the process
    :param numpy.random.Generator rng: a custom random number generator
    """

    def __init__(self, gamma=1.5, mu=0.5, sigma=0.1, initial=1.0, T=1.0, rng=None):
        super().__init__(T=T, rng=rng)
        self.gamma = gamma
        self.mu = mu
        self.sigma = sigma
        self.initial = initial
        self.n = 1.0
        self.dt = 1.0 * self.T / self.n
        self.times = np.arange(0.0, self.T + self.dt, self.dt)
        self.name = "CEV Process"
        self.paths = None
        self._marginals = None

        def f(x, _):
            return self.mu - 0.5 * (self.sigma ** 2) * np.exp(2.0 * (self.gamma - 1.0) * x)
            # return self.mu * x

        def g(x, _):
            return self.sigma * np.exp((self.gamma - 1.0) * x)
            # return self.sigma * (x ** self.gamma)

        self.f = f
        self.g = g

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
        return "CEV process with parameters {gamma}, {drift}, and {volatility} on [0, {T}].".format(
            T=str(self.T), gamma=str(self.gamma), drift=str(self.mu), volatility=str(self.sigma))

    def _get_empirical_marginal_samples(self):
        if self.paths is None:
            self.simulate(self.n, self.N)

        empirical_marginal_samples = np.array(self.paths).transpose()
        return empirical_marginal_samples

    def get_marginal(self, t):
        pass

    def estimate_expectations(self):
        if self._marginals is None:
            self._marginals = self._get_empirical_marginal_samples()

        empirical_means = [np.mean(m) for m in self._marginals]
        return empirical_means

    def estimate_variances(self):
        if self._marginals is None:
            self._marginals = self._get_empirical_marginal_samples()
        empirical_vars = [np.var(m) for m in self._marginals]
        return empirical_vars

    def estimate_stds(self):
        variances = self.estimate_variances()
        stds = [np.sqrt(var) for var in variances]
        return stds

    def estimate_quantiles(self, q):
        if self._marginals is None:
            self._marginals = self._get_empirical_marginal_samples()
        empirical_quantiles = [np.quantile(m, q) for m in self._marginals]
        return empirical_quantiles

    def _process_expectation(self):
        return self.estimate_expectations()

    def process_expectation(self):
        return self._process_expectation()

    def _process_variance(self):
        return self.estimate_variances()

    def process_variance(self):
        return self._process_variance()

    def _process_stds(self):
        stds = np.sqrt(self.process_variance())
        return stds

    def process_stds(self):
        stds = self._process_stds()
        return stds

    def draw(self, n, N, marginal=True, envelope=False, **fig_kw):
        self.simulate(n, N)
        expectations = self.estimate_expectations()

        if envelope:
            lower = self.estimate_quantiles(0.005)
            upper = self.estimate_quantiles(0.995)
        else:
            lower = None
            upper = None

        if marginal:
            figsize = (12, 6)
            fig = draw_paths(times=self.times, paths=self.paths, N=N, KDE=True, name=self.name, marginal=marginal,
                             expectations=expectations, envelope=envelope, lower=lower, upper=upper, figsize=figsize,
                             **fig_kw)
        else:
            figsize = (9.5, 6)
            fig = draw_paths(times=self.times, paths=self.paths, N=N, name=self.name,
                             expectations=expectations, marginal=marginal, figsize=figsize, **fig_kw)

        return fig

    def sample(self, n):
        return self._sample_em_process(n, log=True)
