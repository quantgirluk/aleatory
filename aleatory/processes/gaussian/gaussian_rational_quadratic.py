from aleatory.processes.gaussian.gaussian_processes_base import GaussianLengthScaleSigma
from aleatory.utils.kernels import (
    rational_quadratic_kernel,
    rational_quadratic_kernel_diag,
)


class GPRationalQuadratic(GaussianLengthScaleSigma):
    r"""
    Gaussian Process with Rational Quadratic Kernel
    ================================================

    This process is a Gaussian Process with a rational quadratic kernel. The covariance function is given by

    .. math::

        K(t, s) = \sigma^2 \left(1 + \frac{(t - s)^2}{2\alpha l^2}\right)^{-\alpha} \ \ \ \ \ t, s \in [0,T]

    where :math:`l` is the length scale and :math:`\alpha` is the shape parameter.

    """

    def __init__(self, length_scale=1.0, sigma=1.0, alpha=1.0, T=1.0, rng=None):
        super().__init__(length_scale=length_scale, sigma=sigma, T=T, rng=rng)
        self.alpha = alpha
        self.name = (
            f"Rational Quadratic GP (l={length_scale:.2f}, $\\sigma$={sigma:.2f}, "
            f"$\\alpha$={alpha:.2f})"
        )
        self.short_name = f"Rational Quadratic GP"

    def covariance_function(self, times):
        return rational_quadratic_kernel(
            times, length_scale=self.length_scale, sigma=self.sigma, alpha=self.alpha
        )

    def variance_function(self, times):
        return rational_quadratic_kernel_diag(
            times, length_scale=self.length_scale, sigma=self.sigma, alpha=self.alpha
        )
