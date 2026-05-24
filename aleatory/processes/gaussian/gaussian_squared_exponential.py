from aleatory.processes.gaussian.gaussian_processes_base import GaussianLengthScaleSigma
from aleatory.utils.kernels import (
    squared_exponential_kernel,
    squared_exponential_kernel_diag,
)


class GPSquaredExponential(GaussianLengthScaleSigma):
    r"""
    Gaussian Process with Squared Exponential Kernel
    ================================================

    This process is identical to the Gaussian Process with RBF kernel, as the Squared Exponential kernel is just
    another name for the RBF kernel. The covariance function is given by

    .. math::

        K(t, s) = \sigma^2 \exp\left(-\frac{(t - s)^2}{2l^2}\right) \ \ \ \ \ t, s \in [0,T]


    """

    def __init__(self, length_scale=1.0, sigma=1.0, T=1.0, rng=None):
        super().__init__(length_scale=length_scale, sigma=sigma, T=T, rng=rng)
        self.name = (
            f"Squared Exponential GP (l={length_scale:.2f}, $\\sigma$={sigma:.2f})"
        )
        self.short_name = f"Squared Exponential GP"

    def covariance_function(self, times):
        return squared_exponential_kernel(
            times, length_scale=self.length_scale, sigma=self.sigma
        )

    def variance_function(self, times):
        return squared_exponential_kernel_diag(
            times, length_scale=self.length_scale, sigma=self.sigma
        )
