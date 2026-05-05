"""Gaussian processes"""

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interactive

class GaussianProcess:

    def __init__(self, mean, covariance, T=1.0):
        self.T = T
        self.mean = mean
        self.covariance = covariance


    def covariance_at(self, times):
        return self.covariance(times)
    
    def sample_at(self, times):
        mean = self.mean(times)
        covariance = self.covariance_at(times)
        return np.random.multivariate_normal(mean, covariance)
    
    def sample(self, n, T=None):
        if T is None:
            T = self.T
        times = np.linspace(0, T, n)
        return self.sample_at(times)
    

    def simulate(self, n, N, T=None):
        if T is None:
            T = self.T
        times = np.linspace(0, T, n)
        paths = np.random.multivariate_normal(self.mean(times), self.covariance_at(times), size=N)
        return paths
    

    def plot(self, n, N, T=None):
        paths = self.simulate(n, N, T)
        if T is None:
            T = self.T
        times = np.linspace(0, T, n)
        for i in range(N):
            plt.plot(times, paths[i])
        plt.title("Gaussian Process")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.show()

    def plot_covariance_at(self, times, cmap='coolwarm', title="Covariance Matrix"):
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
        covariance_values = [self.covariance_at([t, t])[0, 1] for t in times]
        plt.plot(times, covariance_values)
        plt.title("Covariance Function")
        plt.xlabel("Time")
        plt.ylabel("Covariance")
        plt.show()

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
        K = self.covariance_at(times)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for i in range(N):
            axes[0].plot(times, paths[i])
        axes[0].set_title("Gaussian Process Paths")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Value")

        axes[1].imshow(K, cmap=cmap, interpolation='none', origin='lower', extent=[0, T, 0, T])
        axes[1].set_title("Covariance Matrix")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Time")

        if title:
            fig.suptitle(title)
        else:
            fig.suptitle("Gaussian Process Paths and Covariance Matrix")

        plt.tight_layout()
        plt.show()
        return fig
    

class GaussianRBF(GaussianProcess):
    
    def __init__(self, length_scale=1.0, sigma=1.0, T=1.0):
        self.length_scale = length_scale
        self.sigma = sigma
        super().__init__(mean=lambda t: np.zeros_like(t), covariance=self.rbf_covariance, T=T)

    def rbf_covariance(self, times):
        pairwise_dists = np.subtract.outer(times, times)**2
        return self.sigma * np.exp(-0.5 * pairwise_dists / self.length_scale**2)
    
    def make_widget(self):
        length_slider = widgets.FloatSlider(value=self.length_scale, min=0.1, max=1.0, step=0.1, description='Length Scale')
        sigma_slider = widgets.FloatSlider(value=self.sigma, min=0.25, max=3.0, step=0.25, description='Sigma')
        n_samples_slider = widgets.IntSlider(value=5, min=1, max=10, step=1, description='Samples')

        def update(length_scale, sigma, n_samples):
            self.length_scale = length_scale
            self.sigma = sigma
            self.plot_paths_and_kernel(n=200, N=n_samples, title=f"RBF GP (length_scale={length_scale:.2f}, sigma={sigma:.2f})")
            # self.plot_gp(times=np.linspace(0, self.T, 100), n_samples=n_samples,
            #              title=f"RBF GP (length_scale={length_scale:.2f}, sigma={sigma:.2f})")

        widget = interact(update, length_scale=length_slider, sigma=sigma_slider, n_samples=n_samples_slider)
        return widget

class GaussianWhiteNoise(GaussianProcess):
    
    def __init__(self, sigma=1.0, T=1.0):
        self.sigma = sigma
        super().__init__(mean=lambda t: np.zeros_like(t), covariance=self.white_noise_covariance, T=T)

    def white_noise_covariance(self, times):
        return self.sigma * np.eye(len(times))
        




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


