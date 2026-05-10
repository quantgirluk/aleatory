"""Common Covariance Functions/Kernels for Gaussian Processes"""

import numpy as np


def _as_array_1d(times):
    return np.asarray(times)


def constant_kernel(times, sigma=1.0):
    return sigma**2 * np.ones((len(times), len(times)))


def constant_kernel_diag(times, sigma=1.0):
    times = _as_array_1d(times)
    return sigma**2 * np.ones_like(times, dtype=float)


def linear_kernel(times, sigma=1.0):
    return sigma**2 * np.outer(times, times)


def linear_kernel_diag(times, sigma=1.0):
    times = _as_array_1d(times)
    return sigma**2 * times**2


def squared_exponential_kernel(times, length_scale=1.0, sigma=1.0):
    sqdist = np.subtract.outer(times, times) ** 2
    return sigma**2 * np.exp(-0.5 * sqdist / length_scale**2)


def squared_exponential_kernel_diag(times, length_scale=1.0, sigma=1.0):
    times = _as_array_1d(times)
    return sigma**2 * np.ones_like(times, dtype=float)


def periodic_kernel(times, length_scale=1.0, sigma=1.0, period=1.0):
    pairwise_dists = np.subtract.outer(times, times) ** 2
    return sigma**2 * np.exp(
        -2 * np.sin(np.pi * pairwise_dists / period) ** 2 / length_scale**2
    )


def periodic_kernel_diag(times, length_scale=1.0, sigma=1.0, period=1.0):
    times = _as_array_1d(times)
    return sigma**2 * np.ones_like(times, dtype=float)


def RBF_kernel(times, length_scale=1.0, sigma=1.0):
    return squared_exponential_kernel(times, length_scale=length_scale, sigma=sigma)


def RBF_kernel_diag(times, length_scale=1.0, sigma=1.0):
    return squared_exponential_kernel_diag(
        times, length_scale=length_scale, sigma=sigma
    )


def white_noise_kernel(times, sigma=1.0):
    return sigma**2 * np.eye(len(times))


def white_noise_kernel_diag(times, sigma=1.0):
    times = _as_array_1d(times)
    return sigma**2 * np.ones_like(times, dtype=float)


def brownian_kernel(times, sigma=1.0):
    return sigma**2 * np.minimum.outer(times, times)


def brownian_kernel_diag(times, sigma=1.0):
    times = _as_array_1d(times)
    return sigma**2 * times


def ournstein_uhlenbeck_kernel(times, theta=1.0, sigma=1.0):
    s = times[:, None]
    t_ = times[None, :]
    m = np.minimum(s, t_)
    res = (sigma**2 / (2 * theta)) * (np.exp(-np.abs(s - t_)) - np.exp(-(s + t_)))
    return res


def ournstein_uhlenbeck_kernel_diag(times, theta=1.0, sigma=1.0):
    times = _as_array_1d(times)
    return (sigma**2 / (2 * theta)) * (1.0 - np.exp(-2.0 * times))


def matern_kernel(times, length_scale=1.0, sigma=1.0, nu=1.5):
    r"""
    Matérn covariance function for time-indexed inputs.

    Parameters
    ----------
    times : array-like, shape (n,)
        Time points (e.g. np.linspace(0, T, n)).
    length_scale : float
        Length scale parameter ℓ (controls correlation decay).
    sigma : float
        Output scale (variance = sigma^2).
    nu : float
        Smoothness parameter.

    Returns
    -------
    K : ndarray, shape (n, n)
        Covariance matrix K(s, t).
    """
    import numpy as np
    from scipy.special import kv, gamma

    times = np.asarray(times)

    # Pairwise time differences |t_i - t_j|
    d = np.abs(times[:, None] - times[None, :])

    # --- Fast special cases ---

    if nu == 0.5:
        # Orsteins-Uhlenbeck (nu=0.5)
        return sigma**2 * np.exp(-d / length_scale)
    elif nu == 1.5:
        # Matern 3/2 (nu=1.5)
        r = np.sqrt(3) * d / length_scale
        return sigma**2 * (1 + r) * np.exp(-r)
    elif nu == 2.5:
        # Matern 5/2 (nu=2.5)
        r = np.sqrt(5) * d / length_scale
        return sigma**2 * (1 + r + r**2 / 3) * np.exp(-r)
    elif np.isinf(nu):
        # Gaussian (squared exponential) kernel as nu -> infinity
        return sigma**2 * np.exp(-(d**2) / (2 * length_scale**2))

    # --- General case ---
    else:
        scaled_d = np.sqrt(2 * nu) * d / length_scale

        # Avoid numerical issues at zero
        scaled_d = np.where(scaled_d == 0, 1e-12, scaled_d)

        K = sigma**2 * (2 ** (1 - nu) / gamma(nu)) * (scaled_d**nu) * kv(nu, scaled_d)

        # Fix diagonal explicitly
        np.fill_diagonal(K, sigma**2)

        return K


def matern_kernel_diag(times, length_scale=1.0, sigma=1.0, nu=1.5):
    r"""
    Diagonal of the Matérn covariance function for time-indexed inputs.

    Parameters
    ----------
    times : array-like, shape (n,)
        Time points (e.g. np.linspace(0, T, n)).
    length_scale : float
        Length scale parameter ℓ (controls correlation decay).
    sigma : float
        Output scale (variance = sigma^2).
    nu : float
        Smoothness parameter.

    Returns
    -------
    K_diag : ndarray, shape (n,)
        Diagonal of the covariance matrix K(s, t).
    """
    times = _as_array_1d(times)
    return sigma**2 * np.ones_like(times, dtype=float)


def rational_quadratic_kernel(times, length_scale=1.0, sigma=1.0, alpha=1.0):
    pairwise_dists = np.subtract.outer(times, times) ** 2
    return sigma**2 * (1 + pairwise_dists / (2 * alpha * length_scale**2)) ** (-alpha)


def rational_quadratic_kernel_diag(times, length_scale=1.0, sigma=1.0, alpha=1.0):
    times = _as_array_1d(times)
    return sigma**2 * np.ones_like(times, dtype=float)


if __name__ == "__main__":

    test = constant_kernel(times=np.linspace(0, 1, 5), sigma=1.0)
    print(test)
