from numpy import c_

from aleatory.processes.gaussian.gaussian_processes_base import GaussianSigma
from aleatory.utils.kernels import linear_kernel, linear_kernel_diag
import ipywidgets as widgets
from ipywidgets import interact

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
        process_name = f"Linear GP ($\\sigma$={sigma:.2f})"
        if c != 0.0 and sigma_b == 0.0:
            process_name = f"Linear GP ($\\sigma$={sigma:.2f}, c={c:.2f})"
        if c != 0.0 and sigma_b != 0.0:
            process_name = f"Linear GP ($\\sigma$={sigma:.2f}, c={c:.2f}, $\\sigma_b$={sigma_b:.2f})"
        if c == 0.0 and sigma_b != 0.0:
            process_name = f"Linear GP ($\\sigma$={sigma:.2f}, $\\sigma_b$={sigma_b:.2f})"
            
        self.name = process_name
        self.short_name = f"Linear GP"

    def covariance_function(self, times):
        return linear_kernel(times, sigma=self.sigma, c=self.c, sigma_b=self.sigma_b)

    def variance_function(self, times):
        return linear_kernel_diag(
            times, sigma=self.sigma, c=self.c, sigma_b=self.sigma_b
        )
    
    def make_widget(self, cmap="coolwarm", matrix_shape=False, **plot_kwargs):

        sigma_slider = widgets.FloatSlider(
            value=self.sigma, min=0.1, max=3.0, step=0.10, description="Sigma"
        )

        c_slider = widgets.FloatSlider(
            value=self.c, min=-3.0, max=3.0, step=0.25, description="c"
        )

        sigma_b_slider = widgets.FloatSlider(
            value=self.sigma_b, min=0.1, max=3.0, step=0.10, description="Sigma_b"
        )

        n_samples_slider = self._widget_n_samples_slider()

        def update(sigma=self.sigma, c=self.c, sigma_b=self.sigma_b, n_samples=5):
            self.sigma = sigma
            self.c = c
            self.sigma_b = sigma_b
            self._widget_plot(
                n_samples=n_samples,
                title=f"GP ($\\sigma$={sigma:.2f}, c={c:.2f}, $\\sigma_b$={sigma_b:.2f})",
                cmap=cmap,
                matrix_shape=matrix_shape,
                plot_kwargs=plot_kwargs,
            )

        widget = interact(update, sigma=sigma_slider, 
                          c=c_slider, sigma_b=sigma_b_slider, n_samples=n_samples_slider)
        return widget
    


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    print("Running GPLinear example...")
    process = GPLinear(sigma=1.0, c=1.0, T=5.0)
    fig = process.plot_paths_and_kernel(n=100, N=5, matrix_shape=False, figsize=(12, 7))
    fig.show()

    # fig = process.draw(n=100, N=200, figsize=(12, 7))
    # fig.show()
