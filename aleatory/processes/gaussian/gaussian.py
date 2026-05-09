"""Gaussian processes"""

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interactive
import kernels as kernels

def check_positive_definite(matrix):
    """ Check if a matrix is positive definite """
    eigenvalues = np.linalg.eigvals(matrix)
    return np.all(eigenvalues > 0)


class GaussianProcess:

    """ 
    A Gaussian Process is a collection of random variables, for which any finite number of which have a joint Gaussian distribution. 
    It is fully specified by its mean function and covariance function (kernel).
    """

    def __init__(self, mean_function, covariance_function, T=1.0):
        self.T = T
        self.mean = mean_function
        self.covariance = covariance_function
        self.kernel = covariance_function
        self.name = "GaussianProcess"
        self.short_name = "GP"

    
    def sample_at(self, times):
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
        return paths
    

    def plot(self, n, N, T=None, title=None):
        paths = self.simulate(n, N, T)
        if T is None:
            T = self.T
        times = np.linspace(0, T, n)
        for i in range(N):
            plt.plot(times, paths[i])
        if title is None:
            title = "Gaussian Process"
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.show()

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
        super().__init__(mean_function=lambda t: np.zeros_like(t), covariance_function=self.covariance_function, T=T)
        self.sigma = sigma
        self.name = f"Gaussian Process ($\\sigma$={sigma:.2f})"
        self.short_name = f"GP"

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
        super().__init__(mean_function=lambda t: np.zeros_like(t), covariance_function=self.covariance_function, T=T)
        self.length_scale = length_scale
        self.sigma = sigma
        self.name = f"Gaussian Process (l={length_scale:.2f}, $\\sigma$={sigma:.2f})"
        self.short_name = f"GP"


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
        super().__init__(mean_function=lambda t: np.zeros_like(t), covariance_function=self.covariance_function, T=T)
        self.length_scale = length_scale
        self.sigma = sigma
        self.nu = nu
        self.name = f"Gaussian Process (l={length_scale:.2f}, $\\sigma$={sigma:.2f}, $\\nu$={nu:.2f})"
        self.short_name = f"GP"

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


class GPLinear(GaussianSigma):
    
    def __init__(self, sigma=1.0, T=1.0):
        super().__init__(sigma=sigma, T=T)
        self.name = f"Linear GP ($\\sigma$={sigma:.2f})"
        self.short_name = f"Linear GP"  

    def covariance_function(self, times):
        return kernels.linear_kernel(times, sigma=self.sigma)


class GPConstant(GaussianSigma):
    
    def __init__(self, sigma=1.0, T=1.0):
        super().__init__(sigma=sigma, T=T)
        self.name = f"Constant GP ($\\sigma$={sigma:.2f})"
        self.short_name = f"Constant GP"

    def covariance_function(self, times):
        return kernels.constant_kernel(times, sigma=self.sigma)


class GPRBF(GaussianLengthScaleSigma):
    
    def __init__(self, length_scale=1.0, sigma=1.0, T=1.0):
        super().__init__(length_scale=length_scale, sigma=sigma, T=T)
        self.name = f"RBF(l={length_scale:.2f}, $\\sigma$={sigma:.2f})"
        self.short_name = f"RBF"    

    def covariance_function(self, times):
        return kernels.RBF_kernel(times, length_scale=self.length_scale, sigma=self.sigma)


class GPSquaredExponential(GaussianLengthScaleSigma):
    
    def __init__(self, length_scale=1.0, sigma=1.0, T=1.0):
        super().__init__(length_scale=length_scale, sigma=sigma, T=T)
        self.name = f"Squared Exponential GP (l={length_scale:.2f}, $\\sigma$={sigma:.2f})"
        self.short_name = f"Squared Exponential GP"

    def covariance_function(self, times):
        return kernels.squared_exponential_kernel(times, length_scale=self.length_scale, sigma=self.sigma)


class GPMatern(GaussianThreeParameter):
    
    def __init__(self, length_scale=1.0, sigma=1.0, nu=1.5, T=1.0):
        super().__init__(length_scale=length_scale, sigma=sigma, nu=nu, T=T)
        self.name = f"Matern GP (l={length_scale:.2f}, $\\sigma$={sigma:.2f}, $\\nu$={nu:.2f})"
        self.short_name = f"Matern GP"

    def covariance_function(self, times):
        return kernels.matern_kernel(times, length_scale=self.length_scale, sigma=self.sigma, nu=self.nu)
    

class GPPeriodic(GaussianThreeParameter):
    
    def __init__(self, length_scale=1.0, sigma=1.0, period=1.0, T=1.0):
        super().__init__(length_scale=length_scale, sigma=sigma, nu=period, T=T)
        self.name = f"Periodic GP (l={length_scale:.2f}, $\\sigma$={sigma:.2f}, p={period:.2f})"
        self.short_name = f"Periodic GP"

    def covariance_function(self, times):
        return kernels.periodic_kernel(times, length_scale=self.length_scale, sigma=self.sigma, period=self.nu)
    

if __name__ == "__main__":
    import math

    mystyle = "https://raw.githubusercontent.com/quantgirluk/matplotlib-stylesheets/main/quant-pastel-light.mplstyle"
    plt.style.use(mystyle)

    g = GPRBF(length_scale=0.1, sigma=1.0, T=1.0)

    # g.plot_covariance()
    # # g.plot_covariance(times=np.linspace(0, 2, 200), title="RBF Kernel", cmap='viridis', matrix_shape=True)

    # # g.plot_covariance(times=np.linspace(0, 1, 100), title="RBF Kernel", cmap='viridis', matrix_shape=False)

    # g.plot_kernel()
    # g.plot_kernel(times=np.linspace(0, 1, 100), cmap='viridis',)

    g.plot_paths_and_kernel(n=100, N=5)
    g.plot_paths_and_kernel(n=100, N=5, matrix_shape=True)

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


