"""
Gaussian Process with Radial Basis Function (RBF) Kernel
"""

from aleatory.processes.gaussian.gaussian_processes_base import GaussianLengthScaleSigma

from aleatory.utils.kernels import (
    RBF_kernel,
    RBF_kernel_diag,
)


class GPRBF(GaussianLengthScaleSigma):
    r"""
    Gaussian Process with Radial Basis Function (RBF) Kernel
    =========================================================

    A Gaussian Process with RBF kernel is a centered Gaussian Process with covariance function given by

    .. math::

        K(t, s) = \sigma^2 \exp\left(-\frac{(t - s)^2}{2l^2}\right)

    where :math:`l` is the length scale parameter and :math:`\sigma` is the scale parameter.


    Notes
    -----


    Examples
    --------
    .. highlight:: python
    .. code-block:: python

        from aleatory.processes import GPRBF
        process = GPRBF(length_scale=0.3, sigma=1.0, T=1.0)
        fig = process.plot_paths_and_kernel(n=100, N=5, matrix_shape=True)
        fig.show()

    .. code-block:: python

        from aleatory.processes import GPRBF
        process = GPRBF(length_scale=0.3, sigma=1.0, T=1.0)
        fig = process.draw(n=100, N=200, figsize=(12, 7))
        fig.show()

    """

    def __init__(self, length_scale=1.0, sigma=1.0, T=1.0, rng=None):
        """
        :param double length_scale: the length scale parameter :math:`l` in the above covariance function
        :param double sigma: the scale parameter :math:`\\sigma` in the above covariance function
        :param double T: the endpoint of the time interval :math:`[0,T]` over which the process is defined
        :param rng: random number generator for reproducibility
        """
        super().__init__(length_scale=length_scale, sigma=sigma, T=T, rng=rng)
        self.name = f"RBF(l={length_scale:.2f}, $\\sigma$={sigma:.2f})"
        self.short_name = f"RBF"

    def covariance_function(self, times):
        return RBF_kernel(times, length_scale=self.length_scale, sigma=self.sigma)

    def variance_function(self, times):
        return RBF_kernel_diag(times, length_scale=self.length_scale, sigma=self.sigma)
