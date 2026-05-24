from aleatory.processes.gaussian.gaussian_processes_base import GaussianThreeParameter

from aleatory.utils.kernels import (
    matern_kernel,
    matern_kernel_diag,
)


class GPMatern(GaussianThreeParameter):
    r"""
    Gaussian Process with Matern Kernel
    ===================================

    Notes
    -----
    A Gaussian Process with Matern kernel is a centered Gaussian Process with covariance function given by

    .. math::

        K(t, s) = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left( \sqrt{2\nu} \frac{|t - s|}{l} \right)^\nu K_\nu\left( \sqrt{2\nu} \frac{|t - s|}{l} \right)

    where :math:`l` is the length scale parameter, :math:`\sigma` is the scale parameter, :math:`\nu` is the smoothness parameter, and :math:`K_{\nu}` is the modified Bessel function of the second kind.

    """

    def __init__(self, length_scale=1.0, sigma=1.0, nu=1.5, T=1.0, rng=None):
        super().__init__(length_scale=length_scale, sigma=sigma, nu=nu, T=T, rng=rng)
        self.name = (
            f"Matern GP (l={length_scale:.2f}, $\\sigma$={sigma:.2f}, $\\nu$={nu:.2f})"
        )
        self.short_name = f"Matern GP"

    def covariance_function(self, times):
        return matern_kernel(
            times, length_scale=self.length_scale, sigma=self.sigma, nu=self.nu
        )

    def variance_function(self, times):
        return matern_kernel_diag(
            times, length_scale=self.length_scale, sigma=self.sigma, nu=self.nu
        )
