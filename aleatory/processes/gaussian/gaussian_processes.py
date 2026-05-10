from aleatory.processes.gaussian.gp_base import (
    GaussianSigma,
    GaussianLengthScaleSigma,
    GaussianThreeParameter,
)

import aleatory.utils.kernels as kernels


class GPWhiteNoise(GaussianSigma):
    r"""
    Gaussian Process White Noise
    ============================


    A centered Gaussian Process with covariance function given by

    .. math::
        K(t, s) = \\sigma^2 \\delta(t - s)

    where :math:`\\delta` is the Dirac delta function.

    Notes
    -----

    - The sample paths of this process are almost surely not continuous, and are not functions in the classical sense, but rather distributions.

    Examples
    --------

    .. highlight:: python
    .. code-block:: python

        from aleatory.processes import GPWhiteNoise
        process = GPWhiteNoise(sigma=1.0, T=1.0)
        fig = process.plot_paths_and_kernel(n=100, N=5, matrix_shape=True)
        fig.show()

    .. code-block:: python

        from aleatory.processes import GPWhiteNoise
        process = GPWhiteNoise(sigma=1.0, T=1.0)
        fig = process.draw(n=100, N=200, figsize=(12, 7))
        fig.show()
    """

    def __init__(self, sigma=1.0, T=1.0):
        super().__init__(sigma=sigma, T=T)
        self.name = f"White Noise ($\\sigma$={sigma:.2f})"
        self.short_name = f"White Noise"

    def covariance_function(self, times):
        return kernels.white_noise_kernel(times, sigma=self.sigma)

    def variance_function(self, times):
        return kernels.white_noise_kernel_diag(times, sigma=self.sigma)


class GPLinear(GaussianSigma):

    def __init__(self, sigma=1.0, T=1.0):
        super().__init__(sigma=sigma, T=T)
        self.name = f"Linear GP ($\\sigma$={sigma:.2f})"
        self.short_name = f"Linear GP"

    def covariance_function(self, times):
        return kernels.linear_kernel(times, sigma=self.sigma)

    def variance_function(self, times):
        return kernels.linear_kernel_diag(times, sigma=self.sigma)


class GPConstant(GaussianSigma):

    def __init__(self, sigma=1.0, T=1.0):
        super().__init__(sigma=sigma, T=T)
        self.name = f"Constant GP ($\\sigma$={sigma:.2f})"
        self.short_name = f"Constant GP"

    def covariance_function(self, times):
        return kernels.constant_kernel(times, sigma=self.sigma)

    def variance_function(self, times):
        return kernels.constant_kernel_diag(times, sigma=self.sigma)


class GPRBF(GaussianLengthScaleSigma):
    r"""
    Gaussian Process with Radial Basis Function (RBF) Kernel
    =========================================================

    Notes
    -----

    A Gaussian Process with RBF kernel is a centered Gaussian Process with covariance function given by

    .. math::
        K(t, s) = \\sigma^2 \\exp\\left(-\\frac{(t - s)^2}{2l^2}\\right)

    where :math:`l` is the length scale parameter and :math:`\\sigma` is the scale parameter.

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

    def __init__(self, length_scale=1.0, sigma=1.0, T=1.0):
        super().__init__(length_scale=length_scale, sigma=sigma, T=T)
        self.name = f"RBF(l={length_scale:.2f}, $\\sigma$={sigma:.2f})"
        self.short_name = f"RBF"

    def covariance_function(self, times):
        return kernels.RBF_kernel(
            times, length_scale=self.length_scale, sigma=self.sigma
        )

    def variance_function(self, times):
        return kernels.RBF_kernel_diag(
            times, length_scale=self.length_scale, sigma=self.sigma
        )


class GPSquaredExponential(GaussianLengthScaleSigma):

    def __init__(self, length_scale=1.0, sigma=1.0, T=1.0):
        super().__init__(length_scale=length_scale, sigma=sigma, T=T)
        self.name = (
            f"Squared Exponential GP (l={length_scale:.2f}, $\\sigma$={sigma:.2f})"
        )
        self.short_name = f"Squared Exponential GP"

    def covariance_function(self, times):
        return kernels.squared_exponential_kernel(
            times, length_scale=self.length_scale, sigma=self.sigma
        )

    def variance_function(self, times):
        return kernels.squared_exponential_kernel_diag(
            times, length_scale=self.length_scale, sigma=self.sigma
        )


class GPMatern(GaussianThreeParameter):
    r""" "
    Gaussian Process with Matern Kernel
    ===================================

    Notes
    -----
    A Gaussian Process with Matern kernel is a centered Gaussian Process with covariance function given by

    .. math::
        K(t, s) = \\sigma^2 \\frac{2^{1-\\nu}}{\\Gamma(\\nu)} \\left(\\sqrt{2\\nu} \\frac{|t - s|}{l}\\right)^{\\!\\!\\!\\!\\!\\!\\!\\!\\!\\!\\!} K_{\\nu}\\left(\\sqrt{2\\nu} \\frac{|t - s|}{l}\\right)

    where :math:`l` is the length scale parameter, :math:`\\sigma` is the scale parameter, :math:`\\nu` is the smoothness parameter, and :math:`K_{\\nu}` is the modified Bessel function of the second kind.

    """

    def __init__(self, length_scale=1.0, sigma=1.0, nu=1.5, T=1.0):
        super().__init__(length_scale=length_scale, sigma=sigma, nu=nu, T=T)
        self.name = (
            f"Matern GP (l={length_scale:.2f}, $\\sigma$={sigma:.2f}, $\\nu$={nu:.2f})"
        )
        self.short_name = f"Matern GP"

    def covariance_function(self, times):
        return kernels.matern_kernel(
            times, length_scale=self.length_scale, sigma=self.sigma, nu=self.nu
        )

    def variance_function(self, times):
        return kernels.matern_kernel_diag(
            times, length_scale=self.length_scale, sigma=self.sigma, nu=self.nu
        )


class GPPeriodic(GaussianThreeParameter):

    def __init__(self, length_scale=1.0, sigma=1.0, period=1.0, T=1.0):
        super().__init__(length_scale=length_scale, sigma=sigma, nu=period, T=T)
        self.name = (
            f"Periodic GP (l={length_scale:.2f}, $\\sigma$={sigma:.2f}, p={period:.2f})"
        )
        self.short_name = f"Periodic GP"

    def covariance_function(self, times):
        return kernels.periodic_kernel(
            times, length_scale=self.length_scale, sigma=self.sigma, period=self.nu
        )

    def variance_function(self, times):
        return kernels.periodic_kernel_diag(
            times, length_scale=self.length_scale, sigma=self.sigma, period=self.nu
        )


if __name__ == "__main__":
    import math
    import matplotlib.pyplot as plt
    import numpy as np

    mystyle = "https://raw.githubusercontent.com/quantgirluk/matplotlib-stylesheets/main/quant-pastel-light.mplstyle"
    plt.style.use(mystyle)

    processes = [
        GPWhiteNoise(sigma=1.0, T=1.0),
        GPLinear(sigma=1.0, T=1.0),
        GPConstant(sigma=1, T=1.0),
        GPRBF(length_scale=0.3, sigma=1.0, T=1.0),
        GPSquaredExponential(length_scale=0.3, sigma=1.0, T=1.0),
        GPMatern(length_scale=0.3, sigma=1.0, nu=1.5, T=1.0),
        GPPeriodic(length_scale=0.3, sigma=1.0, period=0.5, T=1.0),
    ]

    for g in processes:

        # g.plot_paths_and_kernel(n=100, N=5)
        g.plot_paths_and_kernel(n=100, N=5, matrix_shape=True)
        # g.plot(n=100, N=100)
        # g.draw(n=100, N=100)
        g.plot_covariance()
        g.plot_kernel3d()
        # g.plot_mean_function()
        g.plot_mean_variance(times=np.linspace(0, 1.0, 100))

#     def brownian_cov(t):
#         return np.minimum.outer(t, t)

#     def mean_function(t):
#         return np.zeros_like(t)

#     # def covariance_function(t):
#     #     return np.array([[math.exp(-abs(t[i] - t[j])) for j in range(len(t))] for i in range(len(t))])

#     def covariance_function(t):
#         return brownian_cov(t)

#     # gp = GaussianProcess(mean=mean_function, covariance=covariance_function)
#     # gp.plot(n=100, N=5)
#     # # gp.plot_covariance(np.linspace(0, 10, 100))
#     # # gp.plot_covariance_function(n=100)
#     # gp.plot_paths_covariance(n=100, N=5)


#     test = GaussianRBF(length_scale=1.0, sigma=1.0)
#     test.make_widget()

#     # test.plot_paths_covariance(n=100, N=5)
#     # interact(test.plot_paths_covariance, n=5, N=widgets.IntSlider(min=1, max=20, step=1, value=5))
