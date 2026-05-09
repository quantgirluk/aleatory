""" Common Covariance Functions/Kernels for Gaussian Processes """

import numpy as np

def constant_kernel(times, sigma=1.0):
    return sigma**2 * np.ones((len(times), len(times)))
    
def linear_kernel(times, sigma=1.0):
    return sigma**2 * np.outer(times, times)

def squared_exponential_kernel(times, length_scale=1.0, sigma=1.0):
    sqdist = np.subtract.outer(times, times)**2
    return sigma**2 * np.exp(-0.5 * sqdist / length_scale**2)

def periodic_kernel(times, length_scale=1.0, sigma=1.0, period=1.0):
    pairwise_dists = np.subtract.outer(times, times)**2
    return sigma**2 * np.exp(-2 * np.sin(np.pi * pairwise_dists / period)**2 / length_scale**2)

def RBF_kernel(times, length_scale=1.0, sigma=1.0):
    return squared_exponential_kernel(times, length_scale=length_scale, sigma=sigma)

def white_noise_kernel(times, sigma=1.0):
    return sigma**2 * np.eye(len(times))

def brownian_kernel(times, sigma=1.0):
    return sigma**2 * np.minimum.outer(times, times)

def ournstein_uhlenbeck_kernel(times, theta=1.0, sigma=1.0):
    s = times[:, None]
    t_ = times[None, :]
    m = np.minimum(s, t_)    
    res = (sigma**2 / (2 * theta)) * (np.exp(-np.abs(s - t_)) - np.exp(-(s + t_)))
    return res

def matern_kernel(times, length_scale=1.0, sigma=1.0, nu=1.5):
    from scipy.spatial.distance import pdist, squareform
    pairwise_dists = squareform(pdist(times[:, None], metric='euclidean'))
    if nu == 0.5:
        return sigma**2 * np.exp(-pairwise_dists / length_scale)
    elif nu == 1.5:
        sqrt3 = np.sqrt(3)
        return sigma**2 * (1 + sqrt3 * pairwise_dists / length_scale) * np.exp(-sqrt3 * pairwise_dists / length_scale)
    elif nu == 2.5:
        sqrt5 = np.sqrt(5)
        return sigma**2 * (1 + sqrt5 * pairwise_dists / length_scale + 5 * pairwise_dists**2 / (3 * length_scale**2)) * np.exp(-sqrt5 * pairwise_dists / length_scale)
    else:
        raise ValueError("Unsupported nu value. Use 0.5, 1.5, or 2.5.")

def rational_quadratic_kernel(times, length_scale=1.0, sigma=1.0, alpha=1.0):
    pairwise_dists = np.subtract.outer(times, times)**2
    return sigma**2 * (1 + pairwise_dists / (2 * alpha * length_scale**2))**(-alpha)


if __name__ == "__main__":

    test = constant_kernel(times=np.linspace(0, 1, 5), sigma=1.0)
    print(test)