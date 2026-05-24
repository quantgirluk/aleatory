from aleatory.processes.gaussian.gaussian_processes_base import GaussianSigma
from aleatory.utils.kernels import constant_kernel, constant_kernel_diag


class GPConstant(GaussianSigma):
    r"""
    GP with Constant Kernel
    =======================

    A centered Gaussian Process with covariance function given by

    .. math::

        K(t, s) = \sigma^2, \ \ \ \ t, s \in [0,T]

    """

    def __init__(self, sigma=1.0, T=1.0, rng=None):
        super().__init__(sigma=sigma, T=T, rng=rng)
        self.name = f"Constant GP ($\\sigma$={sigma:.2f})"
        self.short_name = f"Constant GP"

    def covariance_function(self, times):
        return constant_kernel(times, sigma=self.sigma)

    def variance_function(self, times):
        return constant_kernel_diag(times, sigma=self.sigma)
