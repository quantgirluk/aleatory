from scipy.stats import rv_continuous
import numpy as np
from scipy.special import kv, gamma


def _vg_log_pdf(x, r, theta, sigma):

    f1 = -np.log(sigma * np.sqrt(np.pi) * gamma(0.5 * r))
    f2 = (theta * x) / sigma**2
    f3 = (0.5 * (r - 1.0)) * np.log(np.abs(x) / (2.0 * np.sqrt(theta**2 + sigma**2)))
    f4 = np.log(
        kv(0.5 * (r - 1), (np.sqrt(theta**2 + sigma**2) * np.abs(x)) / sigma**2)
    )

    value = f1 + f2 + f3 + f4
    return value


def _vg_pdf(x, r, theta, sigma):
    f1 = -np.log(sigma * np.sqrt(np.pi) * gamma(0.5 * r))
    f2 = (theta * x) / sigma**2
    # f30 = (np.abs(x) / (2.0 * np.sqrt(theta**2 + sigma**2))) ** (0.5 * (r - 1.0))
    f3 = (0.5 * (r - 1.0)) * (
        np.log(np.abs(x)) - np.log((2.0 * np.sqrt(theta**2 + sigma**2)))
    )
    special_part = kv(
        0.5 * (r - 1.0), (np.sqrt(theta**2 + sigma**2) * np.abs(x)) / sigma**2
    )
    # f4 = np.log(special_part)
    # value = np.exp(f1 + f2 + f3 + f4)
    value = np.exp(f1 + f2 + f3) * special_part
    return value


# noinspection PyMethodOverriding
class vg_gen(rv_continuous):
    r"""A variance-gamma continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `vg` is:

    .. math::

        f(x, k, \lambda) =  \exp(-(x^2 + \lambda^2)/2)
            (x/\lambda)^{k/2}  \lambda I_{(k/2)-1}(\lambda x)

    for :math:`x >= 0`, :math:`k > 0` and :math:`\lambda \ge 0`.
    :math:`k` specifies the degrees of freedom (denoted ``df`` in the
    implementation) and :math:`\lambda` is the non-centrality parameter
    (denoted ``nc`` in the implementation). :math:`I_\nu` denotes the
    modified Bessel function of first order of degree :math:`\nu`
    (`scipy.special.iv`).

    `ncx` takes ``df`` and ``nc`` as shape parameters.

    %(after_notes)s

    %(example)s

    """

    def _argcheck(self, r, theta, sigma):
        return (r > 0) & (sigma > 0)

    def _rvs(self, r, theta, sigma, size=None, random_state=None):
        s = random_state.gamma(shape=0.5 * r, scale=2.0, size=size)
        z = random_state.normal(loc=0.0, scale=1.0, size=size)
        vg_samples = theta * s + sigma * np.sqrt(s) * z
        return vg_samples

    def _logpdf(self, x, r, theta, sigma):
        values = _vg_log_pdf(x, r, theta, sigma)
        return values

    def _pdf(self, x, r, theta, sigma):
        values = _vg_pdf(x, r, theta, sigma)
        return values

    def _stats(self, r, theta, sigma):
        mu = r * theta
        mu2 = r * (sigma**2 + 2.0 * theta**2)
        variance = mu2
        mu3 = 2.0 * r * theta * (3.0 * sigma**2 + 4.0 * theta**2)
        skewness = mu3 / (mu2 ** (3.0 / 2.0))
        mu4 = (
            3.0
            * r
            * (
                (r + 2.0) * sigma**4
                + (4.0 * r + 16.0) * ((theta * sigma) ** 2 + theta**4)
            )
        )
        kurtosis = mu4 / (mu2**2)

        return mu, variance, skewness, kurtosis


vg = vg_gen(name="vg")


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    pars = {"r": 0.5, "theta": 1.0, "sigma": 1.0}
    f = vg(r=pars["r"], theta=pars["theta"], sigma=pars["sigma"])
    sample = f.rvs(size=500)
    a = -2.0
    b = 2.0
    xs = np.linspace(a, b, 500)
    ys = f.pdf(xs)
    plt.hist(sample, density=True, bins=50)
    plt.plot(xs, ys)
    plt.show()
