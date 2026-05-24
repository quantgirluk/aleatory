from aleatory.processes.gaussian.gaussian_processes_base import GaussianThreeParameter
from aleatory.utils.kernels import (
    periodic_kernel,
    periodic_kernel_diag,
)


class GPPeriodic(GaussianThreeParameter):

    def __init__(self, length_scale=1.0, sigma=1.0, period=1.0, T=1.0, rng=None):
        super().__init__(
            length_scale=length_scale, sigma=sigma, nu=period, T=T, rng=rng
        )
        self.name = (
            f"Periodic GP (l={length_scale:.2f}, $\\sigma$={sigma:.2f}, p={period:.2f})"
        )
        self.short_name = f"Periodic GP"

    def covariance_function(self, times):
        return periodic_kernel(
            times, length_scale=self.length_scale, sigma=self.sigma, period=self.nu
        )

    def variance_function(self, times):
        return periodic_kernel_diag(
            times, length_scale=self.length_scale, sigma=self.sigma, period=self.nu
        )
