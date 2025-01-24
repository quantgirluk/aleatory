from scipy.stats import rv_continuous, chi, ncx2, chi2
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
    type_code = np.mintypecode([a.dtype.char for a in arrays])
    out = np.full(np.shape(arrays[0]), fill_value=fillvalue, dtype=type_code)
    np.place(out, cond, f(*temp))
    if f2 is not None:
        temp = tuple(np.extract(~cond, arr) for arr in arrays)
        np.place(out, ~cond, f2(*temp))

    return out


def _ncx_log_pdf(x, df, nc):
    # We use (x**2 + nc**2)/2 = (x - nc)**2/2  + x*nc, and include the
    # factor of exp(-x*nc) into the ive function to improve numerical
    # stability at large values of x.

    df2 = df / 2.0 - 1.0
    res = np.log(nc) + xlogy(df / 2.0, x / nc) - 0.5 * (x - nc) ** 2
    corr = ive(df2, x * nc)
    value = res + np.log(corr)
    return value


def _ncx_pdf(x, df, nc):
    df2 = df / 2.0 - 1.0
    res = np.log(nc) + xlogy(df / 2.0, x / nc) - 0.5 * (x - nc) ** 2
    corr = ive(df2, x * nc)
    value = np.exp(res) * corr
    return value


# noinspection PyMethodOverriding
class ncx_gen(rv_continuous):
    r"""A non-central chi continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `ncx` is:

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

    def _argcheck(self, df, nc):
        return (df > 0) & (nc >= 0)

    def _rvs(self, df, nc, size=None, random_state=None):
        ncx2_samples = random_state.noncentral_chisquare(df, nc**2, size)
        return np.sqrt(ncx2_samples)

    def _ppf(self, q, df, nc):
        cond = np.ones_like(q, dtype=bool) & (nc != 0)
        values = _lazywhere(cond, (q, df, nc**2), f=ncx2.ppf, f2=chi2.ppf)
        return np.sqrt(values)

    def _logpdf(self, x, df, nc):
        cond = np.ones_like(x, dtype=bool) & (nc != 0)
        return _lazywhere(cond, (x, df, nc), f=_ncx_log_pdf, f2=chi.logpdf)

    def _pdf(self, x, df, nc):
        cond = np.ones_like(x, dtype=bool) & (nc != 0)
        return _lazywhere(cond, (x, df, nc), f=_ncx_pdf, f2=chi.pdf)

    def _stats(self, df, nc):
        alpha = (df / 2.0) - 1.0
        mu = np.sqrt(np.pi / 2.0) * eval_genlaguerre(0.5, alpha, -0.5 * nc**2)
        variance = df + nc**2 - mu**2
        m3 = 3.0 * np.sqrt(np.pi / 2.0) * eval_genlaguerre(1.5, alpha, -0.5 * nc**2)
        skewness = (m3 - 3.0 * mu * variance - mu**3) / (np.sqrt(variance) ** 3)
        m4 = (df + nc**2) ** 2 + 2.0 * (df + 2.0 * nc**2)
        kurtosis = m4 / (variance**2)

        return (mu, variance, skewness, kurtosis)


ncx = ncx_gen(a=0.0, name="ncx")
