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

    def __init__(self, sigma=1.0, c=0.0, sigma_b=0.0, T=1.0, rng=None):
        super().__init__(sigma=sigma, T=T, rng=rng)
        self.c = c
        self.sigma_b = sigma_b
        self.name = f"Linear GP ($\\sigma$={sigma:.2f}, c={c:.2f}, $\\sigma_b$={sigma_b:.2f}, T={T:.2f})"
        self.short_name = f"Linear GP"

    def covariance_function(self, times):
        return linear_kernel(times, sigma=self.sigma, c=self.c, sigma_b=self.sigma_b)

    def variance_function(self, times):
        return linear_kernel_diag(
            times, sigma=self.sigma, c=self.c, sigma_b=self.sigma_b
        )


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    print("Running GPLinear example...")
    process = GPLinear(sigma=1.0, T=5.0)
    fig = process.plot_paths_and_kernel(n=100, N=5, matrix_shape=False)
    fig.show()

    # fig = process.draw(n=100, N=200, figsize=(12, 7))
    # fig.show()
