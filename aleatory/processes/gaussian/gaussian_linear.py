from aleatory.processes.gaussian.gaussian_processes_base import GaussianSigma
from aleatory.utils.kernels import linear_kernel, linear_kernel_diag


class GPLinear(GaussianSigma):
    r"""
    Gaussian Process with Linear Kernel
    ===================================

    A centered Gaussian Process with covariance function given by

    .. math::

        K(t, s) = \sigma^2 ts, \ \ \ \ t, s \in [0,T]

    """

    def __init__(self, sigma=1.0, T=1.0, rng=None):
        super().__init__(sigma=sigma, T=T, rng=rng)
        self.name = f"Linear GP ($\\sigma$={sigma:.2f})"
        self.short_name = f"Linear GP"

    def covariance_function(self, times):
        return linear_kernel(times, sigma=self.sigma)

    def variance_function(self, times):
        return linear_kernel_diag(times, sigma=self.sigma)
