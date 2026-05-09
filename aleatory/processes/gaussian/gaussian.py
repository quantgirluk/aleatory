"""Gaussian processes"""

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interactive
import kernels as kernels

from scipy.stats import norm

from aleatory.processes.base_analytical import SPAnalytical, SPAnalyticalMarginals

def check_positive_definite(matrix):
    """ Check if a matrix is positive definite """
    eigenvalues = np.linalg.eigvals(matrix)
    return np.all(eigenvalues > 0)


class GaussianProcess(SPAnalyticalMarginals):

    """ 
    A Gaussian Process is a collection of random variables, for which any finite number of which have a joint Gaussian distribution. 
    It is fully specified by its mean function and covariance function (kernel).
    """

    def __init__(self, mean_function, covariance_function, variance_function=None, T=1.0, rng=None):
        super().__init__(T=T, rng=rng)
        self.mean = mean_function
        self.covariance = covariance_function
        self.variance = variance_function
        self.kernel = covariance_function
        self.name = "GaussianProcess"
        self.short_name = "GP"
        self.paths = None
        self.times = None


    def sample_at(self, times):
        return self._sample_at(times)
    
    def _sample_at(self, times):
        mean_evaluated = self.mean(times)
        covariance_evaluated = self.covariance(times)
        return np.random.multivariate_normal(mean_evaluated, covariance_evaluated)
    
    def sample(self, n, T=None):
        if T is None:
            T = self.T
        times = np.linspace(0, T, n)
        return self.sample_at(times)
    

    def simulate(self, n, N, T=None):
        if T is None:
            T = self.T
        times = np.linspace(0, T, n)
        paths = np.random.multivariate_normal(self.mean(times), self.covariance(times), size=N)
        self.times = times
        self.paths = paths
        return paths
    

    def _process_expectation(self, times=None):
        if times is None:
            times = self.times
        return self.mean(times)
    

    def _process_variance(self, times=None):
        if times is None:
            times = self.times
        if self.variance is not None:
            return np.asarray(self.variance(times))
        return np.diag(self.covariance(times))
    
    def _process_stds(self, times=None):
        if times is None:
            times = self.times
        return np.sqrt(self._process_variance(times))
    
    def get_marginal(self, time):
        expectation = self.mean(time)
        if self.variance is not None:
            variance = np.asarray(self.variance(np.array([time]))).reshape(-1)[0]
        else:
            variance = self.covariance(np.array([time]))[0, 0]
        return norm(loc=expectation, scale=np.sqrt(variance))
                    
    def draw(
        self, n, N, T=None, marginal=True, envelope=False, type="3sigma", title=None, **fig_kw
    ):
        """
        Simulates and plots paths/trajectories from the instanced stochastic process.

        Produces different kind of visualisation illustrating the following elements:

        - times versus process values as lines
        - the expectation of the process across time
        - histogram showing the empirical marginal distribution :math:`X_T` (optional when ``marginal = True``)
        - probability density function of the marginal distribution :math:`X_T` (optional when ``marginal = True``)
        - envelope of confidence intervals across time (optional when ``envelope = True``)

        :param int n: number of steps in each path
        :param int N: number of paths to simulate
        :param float T: the endpoint of the time interval [0,T] over which the process is defined. If not passed, it defaults to the value of T passed in the constructor.
        :param bool marginal:  defaults to True
        :param bool envelope:   defaults to False
        :param str type:   defaults to  '3sigma'
        :param str title:  to be used to customise plot title. If not passed, the title defaults to the name of the process.
        :return:
        """

        if type == "3sigma":
            return self._draw_3sigmastyle(
                n=n, N=N, T=T, marginal=marginal, envelope=envelope, title=title, **fig_kw
            )
        elif type == "qq":
            return self._draw_qqstyle(
                n, N, T=T, marginal=marginal, envelope=envelope, title=title, **fig_kw
            )
        else:
            raise ValueError
    
    def plot_mean_variance(self, times, **fig_kw):
        return super()._plot_mean_variance(times, process_label="B", **fig_kw)

    

    # def plot(self, n, N, T=None, title=None):
    #     paths = self.simulate(n, N, T)
    #     if T is None:
    #         T = self.T
    #     times = np.linspace(0, T, n)
    #     for i in range(N):
    #         plt.plot(times, paths[i])
    #     if title is None:
    #         title = "Gaussian Process"
    #     plt.title(title)
    #     plt.xlabel("Time")
    #     plt.ylabel("Value")
    #     plt.show()

    def plot_covariance(self, times=None, cmap='coolwarm',matrix_shape=True,  title=None, cbar_label='Covariance'):
        if times is None:
            times = np.linspace(0, self.T, 100)
        covariance_matrix = self.covariance(times)
        title = title if title else f"{self.name} \nCovariance Matrix"
        if matrix_shape:
            origin= 'upper'
            my_extent = [times[0], times[-1], times[-1], times[0]]
        else:            
            origin = 'lower'
            my_extent = [times[0], times[-1], times[0], times[-1]]

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(covariance_matrix, cmap=cmap, interpolation='none', origin=origin, extent=my_extent)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)
        ax.set_title(title)
        ax.set_xlabel("t")
        ax.set_ylabel("s")
        plt.show()

        return fig

    def plot_kernel(self, times=None, cmap='coolwarm', matrix_shape=False, title=None, cbar_label='Kernel K(t, s)'):
        if title is None:
            title = f"{self.name} \nKernel Function"
        self.plot_covariance(times, cmap=cmap, matrix_shape=matrix_shape,title=title, cbar_label=cbar_label)

    def plot_mean_function(self, T=None, n=None):
        if T is None:
            T = self.T
        if n is None:
            n = int(100*T)
        times = np.linspace(0, T, n)
        mean_values = self.mean(times)
        plt.plot(times, mean_values)
        plt.title("Mean Function")
        plt.xlabel("Time")
        plt.ylabel("Mean")
        plt.show()

    def plot_paths_and_kernel(self, n, N, T=None, cmap='coolwarm', matrix_shape=False, title=None):
        if T is None:
            T = self.T
        paths = self.simulate(n, N, T)
        times = np.linspace(0, T, n)
        K = self.covariance(times)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for i in range(N):
            axes[0].plot(times, paths[i])
        axes[0].set_title("Simulated Paths")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Value")

        if matrix_shape:
            origin = 'upper'
            extent = [times[0], times[-1], times[-1], times[0]]
        else:            
            origin = 'lower'
            extent = [times[0], times[-1], times[0], times[-1]]
        
        axes[1].imshow(K, cmap=cmap, interpolation='none', origin=origin, extent=extent)
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=axes[1])
        cbar.set_label('K(t, s)')
        axes[1].set_title("Kernel")
        axes[1].set_xlabel("t")
        axes[1].set_ylabel("s")

        if title:
            fig.suptitle(title)
        else:
            fig.suptitle(f"{self.name}")
        plt.tight_layout()
        plt.show()
        return fig


class GaussianSigma(GaussianProcess):
    
    def __init__(self, sigma=1.0, T=1.0):
        super().__init__(
            mean_function=lambda t: np.zeros_like(t),
            covariance_function=self.covariance_function,
            variance_function=self.variance_function,
            T=T,
        )
        self.sigma = sigma
        self.name = f"Gaussian Process ($\\sigma$={sigma:.2f})"
        self.short_name = f"GP"

    def variance_function(self, times):
        covariance_matrix = self.covariance_function(times)
        return np.diag(covariance_matrix)

    def make_widget(self, matrix_shape=False, cmap='coolwarm'):

        sigma_slider = widgets.FloatSlider(value=self.sigma, min=0.25, max=3.0, step=0.25, description='Sigma')
        n_samples_slider = widgets.IntSlider(value=5, min=2, max=10, step=1, description='Samples')

        def update(sigma=self.sigma, n_samples=5):
            self.sigma = sigma
            self.plot_paths_and_kernel(n=100, N=n_samples, T=self.T, title=f"GP ($\\sigma$={sigma:.2f})", cmap=cmap, matrix_shape=matrix_shape)

        widget = interact(update, sigma=sigma_slider, n_samples=n_samples_slider)
        return widget 

class GaussianLengthScaleSigma(GaussianProcess):
    
    def __init__(self, length_scale=1.0, sigma=1.0, T=1.0):
        super().__init__(
            mean_function=lambda t: np.zeros_like(t),
            covariance_function=self.covariance_function,
            variance_function=self.variance_function,
            T=T,
        )
        self.length_scale = length_scale
        self.sigma = sigma
        self.name = f"Gaussian Process (l={length_scale:.2f}, $\\sigma$={sigma:.2f})"
        self.short_name = f"GP"

    def variance_function(self, times):
        covariance_matrix = self.covariance_function(times)
        return np.diag(covariance_matrix)


    def make_widget(self, matrix_shape=False,cmap='coolwarm'):
        length_slider = widgets.FloatSlider(value=self.length_scale, min=0.1, max=1.0, step=0.1, description='Length Scale')
        sigma_slider = widgets.FloatSlider(value=self.sigma, min=0.25, max=3.0, step=0.25, description='Sigma')
        n_samples_slider = widgets.IntSlider(value=5, min=2, max=10, step=1, description='Samples')

        def update(length_scale, sigma, n_samples=5):
            self.length_scale = length_scale
            self.sigma = sigma
            self.plot_paths_and_kernel(n=100, N=n_samples, T=self.T, title=f"{self.short_name} (l={length_scale:.2f}, $\\sigma$={sigma:.2f})", cmap=cmap, matrix_shape=matrix_shape)

        widget = interact(update, length_scale=length_slider, sigma=sigma_slider, n_samples=n_samples_slider)
        return widget  


class GaussianThreeParameter(GaussianProcess):
    
    def __init__(self, length_scale=1.0, sigma=1.0, nu=1.5, T=1.0):
        super().__init__(
            mean_function=lambda t: np.zeros_like(t),
            covariance_function=self.covariance_function,
            variance_function=self.variance_function,
            T=T,
        )
        self.length_scale = length_scale
        self.sigma = sigma
        self.nu = nu
        self.name = f"Gaussian Process (l={length_scale:.2f}, $\\sigma$={sigma:.2f}, $\\nu$={nu:.2f})"
        self.short_name = f"GP"

    def variance_function(self, times):
        covariance_matrix = self.covariance_function(times)
        return np.diag(covariance_matrix)

    def make_widget(self, matrix_shape=False, cmap='coolwarm'):
        length_slider = widgets.FloatSlider(value=self.length_scale, min=0.1, max=1.0, step=0.1, description='Length Scale')
        sigma_slider = widgets.FloatSlider(value=self.sigma, min=0.25, max=3.0, step=0.25, description='Sigma')
        nu_slider = widgets.FloatSlider(value=self.nu, min=0.5, max=2.5, step=0.5, description='Nu')
        n_samples_slider = widgets.IntSlider(value=5, min=2, max=10, step=1, description='Samples')

        def update(length_scale, sigma, nu, n_samples):
            self.length_scale = length_scale
            self.sigma = sigma
            self.nu = nu
            self.plot_paths_and_kernel(n=100, N=n_samples, T=self.T, title=f"{self.short_name} (l={length_scale:.2f}, $\\sigma$={sigma:.2f}, $\\nu$={nu:.2f})", cmap=cmap, matrix_shape=matrix_shape)

        widget = interact(update, length_scale=length_slider, sigma=sigma_slider, nu=nu_slider, n_samples=n_samples_slider)
        return widget


class GPWhiteNoise(GaussianSigma):
    
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
    
    def __init__(self, length_scale=1.0, sigma=1.0, T=1.0):
        super().__init__(length_scale=length_scale, sigma=sigma, T=T)
        self.name = f"RBF(l={length_scale:.2f}, $\\sigma$={sigma:.2f})"
        self.short_name = f"RBF"    

    def covariance_function(self, times):
        return kernels.RBF_kernel(times, length_scale=self.length_scale, sigma=self.sigma)

    def variance_function(self, times):
        return kernels.RBF_kernel_diag(times, length_scale=self.length_scale, sigma=self.sigma)


class GPSquaredExponential(GaussianLengthScaleSigma):
    
    def __init__(self, length_scale=1.0, sigma=1.0, T=1.0):
        super().__init__(length_scale=length_scale, sigma=sigma, T=T)
        self.name = f"Squared Exponential GP (l={length_scale:.2f}, $\\sigma$={sigma:.2f})"
        self.short_name = f"Squared Exponential GP"

    def covariance_function(self, times):
        return kernels.squared_exponential_kernel(times, length_scale=self.length_scale, sigma=self.sigma)

    def variance_function(self, times):
        return kernels.squared_exponential_kernel_diag(times, length_scale=self.length_scale, sigma=self.sigma)


class GPMatern(GaussianThreeParameter):
    
    def __init__(self, length_scale=1.0, sigma=1.0, nu=1.5, T=1.0):
        super().__init__(length_scale=length_scale, sigma=sigma, nu=nu, T=T)
        self.name = f"Matern GP (l={length_scale:.2f}, $\\sigma$={sigma:.2f}, $\\nu$={nu:.2f})"
        self.short_name = f"Matern GP"

    def covariance_function(self, times):
        return kernels.matern_kernel(times, length_scale=self.length_scale, sigma=self.sigma, nu=self.nu)

    def variance_function(self, times):
        return kernels.matern_kernel_diag(times, length_scale=self.length_scale, sigma=self.sigma, nu=self.nu)
    

class GPPeriodic(GaussianThreeParameter):
    
    def __init__(self, length_scale=1.0, sigma=1.0, period=1.0, T=1.0):
        super().__init__(length_scale=length_scale, sigma=sigma, nu=period, T=T)
        self.name = f"Periodic GP (l={length_scale:.2f}, $\\sigma$={sigma:.2f}, p={period:.2f})"
        self.short_name = f"Periodic GP"

    def covariance_function(self, times):
        return kernels.periodic_kernel(times, length_scale=self.length_scale, sigma=self.sigma, period=self.nu)

    def variance_function(self, times):
        return kernels.periodic_kernel_diag(times, length_scale=self.length_scale, sigma=self.sigma, period=self.nu)
    

if __name__ == "__main__":
    import math

    mystyle = "https://raw.githubusercontent.com/quantgirluk/matplotlib-stylesheets/main/quant-pastel-light.mplstyle"
    plt.style.use(mystyle)

    processes = [GPWhiteNoise(sigma=1.0, T=1.0), GPLinear(sigma=1.0, T=1.0), 
                 GPConstant(sigma=1, T=1.0), GPRBF(length_scale=0.3, sigma=1.0, T=1.0), 
                 GPSquaredExponential(length_scale=0.3, sigma=1.0, T=1.0), 
                 GPMatern(length_scale=0.3, sigma=1.0, nu=1.5, T=1.0), 
                 GPPeriodic(length_scale=0.3, sigma=1.0, period=0.5, T=1.0)]

    for g in processes:

        # g.plot_paths_and_kernel(n=100, N=5)
        g.plot_paths_and_kernel(n=100, N=5, matrix_shape=True)
        g.plot(n=100, N=100)
        g.draw(n=100, N=100)
        g.plot_covariance()
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

