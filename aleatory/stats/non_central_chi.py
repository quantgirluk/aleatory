from scipy.stats import rv_continuous, chi, ncx2

import numpy as np
from scipy.special import xlogy, ive, eval_genlaguerre


def _lazywhere(cond, arrays, f, fillvalue=None, f2=None):
    """
    np.where(cond, x, fillvalue) always evaluates x even where cond is False.
    This one only evaluates f(arr1[cond], arr2[cond], ...).

    Examples
    --------
    # >>> a, b = np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])
    # >>> def f(a, b):
    # ...     return a*b
    # >>> _lazywhere(a > 2, (a, b), f, np.nan)
    array([ nan,  nan,  21.,  32.])

    Notice, it assumes that all `arrays` are of the same shape, or can be
    broadcasted together.

    """
    cond = np.asarray(cond)
    if fillvalue is None:
        if f2 is None:
            raise ValueError("One of (fillvalue, f2) must be given.")
        else:
            fillvalue = np.nan
    else:
        if f2 is not None:
            raise ValueError("Only one of (fillvalue, f2) can be given.")

    args = np.broadcast_arrays(cond, *arrays)
    cond, arrays = args[0], args[1:]
    temp = tuple(np.extract(cond, arr) for arr in arrays)
    tcode = np.mintypecode([a.dtype.char for a in arrays])
    out = np.full(np.shape(arrays[0]), fill_value=fillvalue, dtype=tcode)
    np.place(out, cond, f(*temp))
    if f2 is not None:
        temp = tuple(np.extract(~cond, arr) for arr in arrays)
        np.place(out, ~cond, f2(*temp))

    return out


def _ncx_log_pdf(x, df, nc):
    # We use (x**2 + nc**2)/2 = (x - nc)**2/2  + x*nc, and include the
    # factor of exp(-x*nc) into the ive function to improve numerical
    # stability at large values of x.
    # if nc == 0:
    #     value = chi.logpdf(x, df)
    #     return value

    df2 = df / 2.0 - 1.0
    res = np.log(nc) + xlogy(df / 2.0, x / nc) - 0.5 * (x - nc) ** 2
    corr = ive(df2, x * nc)
    # Return res + np.log(corr) avoiding np.log(0)
    value = res + np.log(corr)
    return value


def _ncx_pdf(x, df, nc):
    # if nc == 0:
    #     value = chi.logpdf(x, df)
    #     return value

    df2 = df / 2.0 - 1.0
    res = np.log(nc) + xlogy(df / 2.0, x / nc) - 0.5 * (x - nc) ** 2
    corr = ive(df2, x * nc)
    value = np.exp(res) * corr
    return value


# noinspection PyMethodOverriding
class ncx_gen(rv_continuous):

    def _argcheck(self, df, nc):
        return (df > 0) & (nc >= 0)

    def _rvs(self, df, nc, size=None, random_state=None):
        ncx2_samples = random_state.noncentral_chisquare(df, nc, size)
        return np.sqrt(ncx2_samples)

    def _logpdf(self, x, df, nc):
        cond = np.ones_like(x, dtype=bool) & (nc != 0)
        return _lazywhere(cond, (x, df, nc), f=_ncx_log_pdf, f2=chi.logpdf)

    def _pdf(self, x, df, nc):
        cond = np.ones_like(x, dtype=bool) & (nc != 0)
        return _lazywhere(cond, (x, df, nc), f=_ncx_pdf, f2=chi.pdf)

    def _ppf(self, x, df, nc):
        return np.sqrt(ncx2.ppf(x, df, nc))

    def _stats(self, df, nc):
        alpha = (df / 2.0) - 1.0
        mu = np.sqrt(np.pi / 2.0) * eval_genlaguerre(0.5, alpha, -0.5 * nc ** 2)
        variance = df + nc ** 2 - mu ** 2
        m3 = 3.0 * np.sqrt(np.pi / 2.0) * eval_genlaguerre(1.5, alpha, -0.5 * nc ** 2)
        skewness = (m3 - 3.0 * mu * variance - mu ** 3) / (np.sqrt(variance) ** 3)
        m4 = (df + nc ** 2) ** 2 + 2.0 * (df + 2.0 * nc ** 2)
        kurtosis = m4 / (variance ** 2)

        return (mu,
                variance,
                skewness,
                kurtosis)


ncx = ncx_gen(a=0.0, name='ncx')
#
# import matplotlib.pyplot as plt
#
# marginal = ncx(df=4.0, nc=1.0)
# x = np.linspace(0, 5, 100)
#
# plt.plot(x, ncx.pdf(x, df=4., nc=1.0))
# plt.plot(x, marginal.pdf(x))
# plt.show()
#
# plt.plot(x, ncx.cdf(x, df=4., nc=1.0))
# plt.plot(x, marginal.cdf(x))
# plt.show()
#
# x = np.linspace(0.001, 0.999, 100)
# plt.plot(x, ncx.ppf(x, df=4., nc=1.0))
# plt.plot(x, marginal.ppf(x))
# plt.show()
#
# sample = marginal.rvs(size=10000)
# plt.hist(sample, density=True, bins=70)
# x = np.linspace(0, 5, 100)
# plt.plot(x, marginal.pdf(x))
# plt.show()
