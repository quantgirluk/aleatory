from scipy.stats import rv_continuous
import numpy as np
from scipy.special import xlogy, ive


def _ncx_log_pdf(x, df, nc):
    # We use (x**2 + nc**2)/2 = (x - nc)**2/2  + x*nc, and include the
    # factor of exp(-x*nc) into the ive function to improve numerical
    # stability at large values of x.
    df2 = df / 2.0 - 1.0
    res = np.log(nc) + xlogy(df / 2.0, x / nc) - 0.5 * (x - nc) ** 2
    corr = ive(df2, x * nc)
    # Return res + np.log(corr) avoiding np.log(0)
    return res + np.log(corr)


def _ncx_pdf(x, df, nc):
    df2 = df / 2.0 - 1.0
    res = np.log(nc) + xlogy(df / 2.0, x / nc) - 0.5 * (x - nc) ** 2
    corr = ive(df2, x * nc)
    return np.exp(res) * corr


# noinspection PyMethodOverriding
class ncx(rv_continuous):

    def __init__(self, df: float, nc: float) -> None:
        super().__init__()
        self.df = df
        self.nc = nc

    def _argcheck(self, df, nc):
        return (df > 0) & (nc >= 0)

    def _rvs(self, *args, size=None, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
            ncx2_samples = random_state.noncentral_chisquare(df=self.df, nc=self.nc, size=size)
            return np.sqrt(ncx2_samples)

    def _logpdf(self, x, df, nc):
        value = _ncx_log_pdf(x, df, nc)
        return value

    def _pdf(self, x, df, nc):
        value = _ncx_pdf(x, df, nc)

        return value
