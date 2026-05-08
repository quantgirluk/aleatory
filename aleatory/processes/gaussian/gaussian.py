"""Gaussian processes"""

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interactive
import kernels as kernels

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

    def plot_covariance_at(self, times, cmap='coolwarm', title="Covariance Function"):
        covariance_matrix = self.covariance_at(times)
        plt.imshow(covariance_matrix, cmap=cmap, interpolation='none', origin='lower')
        plt.colorbar()
        plt.title(title)
        plt.xlabel("Time Index")
        plt.ylabel("Time Index")
        plt.show()

    def plot_kernel_at(self, times):
        self.plot_covariance_at(times, title="Covariance Kernel")

    def plot_covariance_function(self, T, n):
        times = np.linspace(0, T, n)
        self.plot_covariance_at(times)

    def plot_kernel(self, T, n):
        self.plot_covariance_function(T, n)

    def plot_mean_function(self, T, n):
        times = np.linspace(0, T, n)
        mean_values = self.mean(times)
        plt.plot(times, mean_values)
        plt.title("Mean Function")
        plt.xlabel("Time")
        plt.ylabel("Mean")
        plt.show()

    def plot_paths_and_kernel(self, n, N, T=None, cmap='coolwarm', title=None):
        
        if T is None:
            T = self.T

        paths = self.simulate(n, N, T)
        times = np.linspace(0, T, n)
        K = self.covariance(times)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for i in range(N):
            axes[0].plot(times, paths[i])
        axes[0].set_title("Gaussian Process Paths")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Value")

        axes[1].imshow(K, cmap=cmap, interpolation='none', origin='lower', extent=[0, T, 0, T])
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=axes[1])
        cbar.set_label('Covariance')
        axes[1].set_title("Kernel")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Time")

        if title:
            fig.suptitle(title)
        else:
            fig.suptitle(f"{self.name} Paths and Kernel/Covariance Function")

        plt.tight_layout()
        plt.show()
        return fig


class GaussianSigma(GaussianProcess):
    
    def __init__(self, sigma=1.0, T=1.0):
        super().__init__(mean_function=lambda t: np.zeros_like(t), covariance_function=self.covariance_function, T=T)
        self.sigma = sigma
        self.name = f"Gaussian Process (sigma={sigma:.2f})"

    def make_widget(self):

        sigma_slider = widgets.FloatSlider(value=self.sigma, min=0.25, max=3.0, step=0.25, description='Sigma')
        n_samples_slider = widgets.IntSlider(value=5, min=2, max=10, step=1, description='Samples')

        def update(sigma=self.sigma, n_samples=5):
            self.sigma = sigma
            self.plot_paths_and_kernel(n=100, N=n_samples, T=self.T, title=f"GP (sigma={sigma:.2f})")

        widget = interact(update, sigma=sigma_slider, n_samples=n_samples_slider)
        return widget 

class GaussianLengthScaleSigma(GaussianProcess):
    
    def __init__(self, length_scale=1.0, sigma=1.0, T=1.0):
        super().__init__(mean_function=lambda t: np.zeros_like(t), covariance_function=self.covariance_function, T=T)
        self.length_scale = length_scale
        self.sigma = sigma
        self.name = f"Gaussian Process (length_scale={length_scale:.2f}, sigma={sigma:.2f})"


    def make_widget(self):
        length_slider = widgets.FloatSlider(value=self.length_scale, min=0.1, max=1.0, step=0.1, description='Length Scale')
        sigma_slider = widgets.FloatSlider(value=self.sigma, min=0.25, max=3.0, step=0.25, description='Sigma')
        n_samples_slider = widgets.IntSlider(value=5, min=2, max=10, step=1, description='Samples')

        def update(length_scale, sigma, n_samples=5):
            self.length_scale = length_scale
            self.sigma = sigma
            self.plot_paths_and_kernel(n=100, N=n_samples, T=self.T, title=f"{self.name} (length_scale={length_scale:.2f}, sigma={sigma:.2f})")

        widget = interact(update, length_scale=length_slider, sigma=sigma_slider, n_samples=n_samples_slider)
        return widget  


class GaussianThreeParameter(GaussianProcess):
    
    def __init__(self, length_scale=1.0, sigma=1.0, nu=1.5, T=1.0):
        super().__init__(mean_function=lambda t: np.zeros_like(t), covariance_function=self.covariance_function, T=T)
        self.length_scale = length_scale
        self.sigma = sigma
        self.nu = nu
        self.name = f"Gaussian Process (length_scale={length_scale:.2f}, sigma={sigma:.2f}, nu={nu:.2f})"

    def make_widget(self):
        length_slider = widgets.FloatSlider(value=self.length_scale, min=0.1, max=1.0, step=0.1, description='Length Scale')
        sigma_slider = widgets.FloatSlider(value=self.sigma, min=0.25, max=3.0, step=0.25, description='Sigma')
        nu_slider = widgets.FloatSlider(value=self.nu, min=0.5, max=2.5, step=0.5, description='Nu')
        n_samples_slider = widgets.IntSlider(value=5, min=2, max=10, step=1, description='Samples')

        def update(length_scale, sigma, nu, n_samples):
            self.length_scale = length_scale
            self.sigma = sigma
            self.nu = nu
            self.plot_paths_and_kernel(n=100, N=n_samples, T=self.T, title=f"{self.name} (length_scale={length_scale:.2f}, sigma={sigma:.2f}, nu={nu:.2f})")

        widget = interact(update, length_scale=length_slider, sigma=sigma_slider, nu=nu_slider, n_samples=n_samples_slider)
        return widget


    

class GaussianWhiteNoise(GaussianSigma):
    
    def __init__(self, sigma=1.0, T=1.0):
        super().__init__(sigma=sigma, T=T)
        self.name = f"White Noise GP (sigma={sigma:.2f})"

    def covariance_function(self, times):
        return kernels.white_noise_kernel(times, sigma=self.sigma)


class GaussianLinear(GaussianSigma):
    
    def __init__(self, sigma=1.0, T=1.0):
        super().__init__(sigma=sigma, T=T)
        self.name = f"Linear GP (sigma={sigma:.2f})"

    def covariance_function(self, times):
        return kernels.linear_kernel(times, sigma=self.sigma)


class GaussianConstant(GaussianSigma):
    
    def __init__(self, sigma=1.0, T=1.0):
        super().__init__(sigma=sigma, T=T)
        self.name = f"Constant GP (sigma={sigma:.2f})"

    def covariance_function(self, times):
        return kernels.constant_kernel(times, sigma=self.sigma)


class GaussianRBF(GaussianLengthScaleSigma):
    
    def __init__(self, length_scale=1.0, sigma=1.0, T=1.0):
        super().__init__(length_scale=length_scale, sigma=sigma, T=T)
        self.name = f"RBF Gaussian Process (length_scale={length_scale:.2f}, sigma={sigma:.2f})"

    def covariance_function(self, times):
        return kernels.RBF_kernel(times, length_scale=self.length_scale, sigma=self.sigma)


class GaussianSquaredExponential(GaussianLengthScaleSigma):
    
    def __init__(self, length_scale=1.0, sigma=1.0, T=1.0):
        super().__init__(length_scale=length_scale, sigma=sigma, T=T)
        self.name = f"Squared Exponential GP (length_scale={length_scale:.2f}, sigma={sigma:.2f})"

    def covariance_function(self, times):
        return kernels.squared_exponential_kernel(times, length_scale=self.length_scale, sigma=self.sigma)


class GaussianMatern(GaussianThreeParameter):
    
    def __init__(self, length_scale=1.0, sigma=1.0, nu=1.5, T=1.0):
        super().__init__(length_scale=length_scale, sigma=sigma, nu=nu, T=T)
        self.name = f"Matern GP (length_scale={length_scale:.2f}, sigma={sigma:.2f}, nu={nu:.2f})"

    def covariance_function(self, times):
        return kernels.matern_kernel(times, length_scale=self.length_scale, sigma=self.sigma, nu=self.nu)
    

class GaussianPeriodic(GaussianThreeParameter):
    
    def __init__(self, length_scale=1.0, sigma=1.0, period=1.0, T=1.0):
        super().__init__(length_scale=length_scale, sigma=sigma, nu=period, T=T)
        self.name = f"Periodic GP (length_scale={length_scale:.2f}, sigma={sigma:.2f}, period={period:.2f})"

    def covariance_function(self, times):
        return kernels.periodic_kernel(times, length_scale=self.length_scale, sigma=self.sigma, period=self.nu)
    
    





# if __name__ == "__main__":
#     import math

#     mystyle = "https://raw.githubusercontent.com/quantgirluk/matplotlib-stylesheets/main/quant-pastel-light.mplstyle"
#     plt.style.use(mystyle)

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


