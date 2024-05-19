from aleatory.processes.euler_maruyama.ckls_process_generic import CKLSProcessGeneric
from aleatory.processes import BrownianMotion, Vasicek, CIRProcess, GBM


class CKLSProcess(CKLSProcessGeneric):
    def __new__(cls, *args, **kwargs):
        alpha = kwargs['alpha'] if  'alpha' in kwargs else 0.5
        beta = kwargs['beta'] if 'beta' in kwargs else 0.5
        sigma = kwargs['sigma'] if 'sigma' in kwargs else 0.1
        gamma = kwargs['gamma'] if 'gamma' in kwargs else 1.5
        initial = kwargs['initial'] if 'initial' in kwargs else 1.0
        T = kwargs['T'] if 'T' in kwargs else 1.0
        rng = kwargs['rng'] if 'rng' in kwargs else None
        if beta == 0.0 and gamma == 0:
            return BrownianMotion(drift=alpha, scale=sigma, T=T, rng=rng)
        elif gamma == 0 and beta<0:
            theta = -1.0 * beta
            mu = -1.0 * alpha / beta
            return Vasicek(theta=theta, mu=mu, sigma=sigma, initial=initial, T=T, rng=rng)
        elif gamma == 0.5:
            theta = -1.0 * beta
            mu = -1.0 * alpha / beta
            return CIRProcess(theta=theta, mu=mu, sigma=sigma, initial=initial, T=T, rng=rng)
        elif alpha == 0.0 and gamma == 1.0:
            return GBM(drift=beta, volatility=sigma, initial=initial, T=T, rng=rng)
        else:
            return CKLSProcessGeneric(alpha=alpha, beta=beta, sigma=sigma, gamma=gamma, initial=initial, T=T, rng=rng)

    # def __init__(self, alpha=0.5, beta=0.5, sigma=0.1, gamma=1.0, initial=1.0, T=1.0, rng=None):
    #     super().__init__(alpha, beta, sigma, gamma, initial, T, rng)

    # def __str__(self):
    #     return "CKLS process with parameters alpha={alpha}, beta={beta}, sigma={sigma}, gamma={gamma}, initial={initial} on [0, {T}].".format(
    #         T=str(self.T), gamma=str(self.gamma), alpha=str(self.alpha), beta=str(self.beta), sigma=str(self.sigma),
    #         initial=self.initial)
