from aleatory.processes.gaussian.gaussian_processes_base import (
    GaussianSigma,
)

from aleatory.utils.kernels import white_noise_kernel, white_noise_kernel_diag


class WhiteNoise(GaussianSigma):
    r"""
    Gaussian Process White Noise
    ============================


    A centered Gaussian Process with covariance function given by

    .. math::
        K(t, s) = \sigma^2 \delta(t - s)

    where :math:`\delta` is the Dirac delta function.

    Notes
    -----

    - The sample paths of this process are almost surely not continuous, and are not functions in the classical sense, but rather distributions.

    Examples
    --------

    .. highlight:: python
    .. code-block:: python

        from aleatory.processes import WhiteNoise
        process = WhiteNoise(sigma=1.0, T=1.0)
        fig = process.plot_paths_and_kernel(n=100, N=5, matrix_shape=True)
        fig.show()

    .. code-block:: python

        from aleatory.processes import WhiteNoise
        process = WhiteNoise(sigma=1.0, T=1.0)
        fig = process.draw(n=100, N=200, figsize=(12, 7))
        fig.show()
    """

    def __init__(self, sigma=1.0, T=1.0, rng=None):
        super().__init__(sigma=sigma, T=T, rng=rng)
        self.name = f"White Noise ($\\sigma$={sigma:.2f})"
        self.short_name = f"White Noise"

    def covariance_function(self, times):
        return white_noise_kernel(times, sigma=self.sigma)

    def variance_function(self, times):
        return white_noise_kernel_diag(times, sigma=self.sigma)


# if __name__ == "__main__":
#     import math
#     import matplotlib.pyplot as plt
#     import numpy as np

#     mystyle = "https://raw.githubusercontent.com/quantgirluk/matplotlib-stylesheets/main/quant-pastel-light.mplstyle"
#     plt.style.use(mystyle)

#     processes = [
#         # WhiteNoise(sigma=1.0, T=1.0),
#         # GPLinear(sigma=1.0, T=1.0),
#         # GPConstant(sigma=1, T=1.0),
#         # GPRBF(length_scale=0.3, sigma=1.0, T=1.0),
#         GPSquaredExponential(length_scale=0.3, sigma=1.0, T=1.0),
#         GPMatern(length_scale=0.3, sigma=1.0, nu=1.5, T=1.0),
#         # GPPeriodic(length_scale=0.3, sigma=1.0, period=0.5, T=1.0),
#     ]

#     for g in processes:

#         # g.plot_paths_and_kernel(n=100, N=5)
#         # g.plot_paths_and_kernel(n=100, N=5, matrix_shape=True)
#         # g.plot_paths_and_kernel(n=100, N=5)

#         # g.plot(n=100, N=100)
#         # g.draw(n=100, N=100)
#         g.plot_covariance()
#         g.plot_kernel()
#         # g.plot_kernel3d()
#         # g.plot_mean_function()
#         # g.plot_mean_variance(times=np.linspace(0, 1.0, 100))
