"""Mixed Poisson Process"""

import numpy as np


from aleatory.processes.base import BaseProcess
from aleatory.utils.utils import check_positive_number, check_positive_integer
from aleatory.utils.plotters import plot_poisson, draw_poisson_like


class MixedPoissonProcess(BaseProcess):
    r"""
    Mixed Poisson Process
    =====================

    .. image:: ../_static/mixed_poisson_draw.png

    Notes
    -----

    In probability theory, a mixed Poisson process is a special point process that is a generalization of a
    Poisson process. Mixed Poisson processes are simple example for Cox processes.

    A Mixed Poisson process (MPP) :math:`\{N(t), t\in [0,\infty)\}` is a counting process with counting distribution of the form:

    .. math::
        P(N(t)= n) = \int_0^{\infty} \frac{1}{n!} e^{-\lambda t} (\lambda t)^n d \Lambda(\lambda), \qquad n\in \mathbb{N},

    where :math:`\Lambda` is the structure distribution given by

    .. math::
        \Lambda(\lambda) = P(\Lambda \leq \lambda)

    with :math:`\Lambda(0)=0`. This type of distribution is known as mixed Poisson distribution which gives the name to the processes.

    Constructor, Methods, and Attributes
    ------------------------------------
    """

    def __init__(self, intensity, intensity_args=None, intensity_kwargs=None, rng=None):
        """
        :parameter callable intensity: a callable function which defines the structure distribution :math:`\Lambda`
        :parameter intensity_args: the arguments to be passed to the intensity function
        :parameter intensity_kwargs: the keyword arguments to be passed to the intensity function
        :parameter numpy.random.Generator rng: a custom random number generator
        """
        super().__init__(rng=rng)
        self.intensity = intensity
        self.intensity_args = intensity_args if intensity_args is not None else {}
        self.intensity_kwargs = intensity_kwargs if intensity_kwargs is not None else {}
        self.name = "Mixed Poisson Process"
        self.T = None
        self.N = None
        self.paths = None

    def __str__(self):
        return "Mixed Poisson Process"

    def __repr__(self):
        return "MixedPoissonProcess"

    @property
    def intensity(self):
        return self._intensity

    @intensity.setter
    def intensity(self, value):
        if not callable(value):
            raise ValueError("intensity must be a callable function")
        self._intensity = value

    def _sample_rate(self):
        rate = self.intensity(*self.intensity_args, **self.intensity_kwargs)
        return rate

    def _sample_poisson_process(self, jumps=None, T=None):
        exp_mean = 1.0 / self._sample_rate()
        if jumps is not None and T is not None:
            raise ValueError("Only one must be provided either jumps or T")
        elif jumps:
            check_positive_integer(jumps)
            exponential_times = self.rng.exponential(exp_mean, size=jumps)
            arrival_times = np.cumsum(exponential_times)
            arrival_times = np.insert(arrival_times, 0, [0])
            return arrival_times
        elif T:
            check_positive_number(T, "Time")
            t = 0.0
            arrival_times = [0.0]
            while t < T:
                t += self.rng.exponential(scale=exp_mean)
                arrival_times.append(t)
            return np.array(arrival_times)

    def sample(self, jumps=None, T=None):
        return self._sample_poisson_process(jumps=jumps, T=T)

    def simulate(self, N, jumps=None, T=None):
        """
        Simulate paths/trajectories from the instanced stochastic process.
        It requires either the number of jumps (`jumps`) or the  time (`T`)
        for the simulation to end.

        - If `jumps` is provided, the function returns :math:`N` paths each one with exactly that number of jumps.

        - If `T` is provided, the function returns :math:`N` paths over the time :math:`[0,T]`. Note that in this case each path can have a different number of jumps.

        :param int N: number of paths to simulate
        :param int jumps: number of jumps
        :param float T: time T
        :return: list with N paths (each one is a numpy array of size n)

        """
        self.N = N
        self.paths = [self.sample(jumps=jumps, T=T) for _ in range(N)]
        return self.paths

    def plot(self, N, jumps=None, T=None, title=None, **fig_kwargs):

        paths = self.simulate(N, jumps=jumps, T=T)
        plot_title = title if title else self.name

        return plot_poisson(
            jumps=jumps, T=T, paths=paths, title=plot_title, **fig_kwargs
        )

    def draw(
        self,
        N,
        T=None,
        mode="steps",
        style="seaborn-v0_8-whitegrid",
        colormap="RdYlBu_r",
        envelope=True,
        marginal=True,
        colorspos=None,
        title=None,
        **fig_kw,
    ):

        title = title if title else self.name
        self.simulate(N, T=T)
        paths = self.paths

        times = np.linspace(0.0, T, 200)

        if hasattr(self, "marginal_expectation"):
            expectations = self.marginal_expectation(times)
        else:
            expectations = None

        if hasattr(self, "get_marginal"):
            marginalT = self.get_marginal(T)
            marginals = [self.get_marginal(ti) for ti in times]
            lower = [m.ppf(0.005) for m in marginals]
            upper = [m.ppf(0.9995) for m in marginals]
        else:
            marginalT = None
            lower = None
            upper = None

        fig = draw_poisson_like(
            T,
            paths,
            marginalT=marginalT,
            expectations=expectations,
            envelope=envelope,
            lower=lower,
            upper=upper,
            style=style,
            colormap=colormap,
            marginal=marginal,
            mode=mode,
            colorspos=colorspos,
            title=title,
            **fig_kw,
        )

        return fig


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from scipy.stats import gamma, chi2

    def intensity_gamma(a=1.0):
        g = gamma(a=a)
        return g.rvs()

    p1 = MixedPoissonProcess(intensity=intensity_gamma)
    t1 = "Mixed Poisson Process with $\\Lambda \sim \Gamma(1.0, 1.0)$"
    p2 = MixedPoissonProcess(intensity=intensity_gamma, intensity_kwargs={"a": 3.0})
    t2 = "Mixed Poisson Process with $\\Lambda \sim \Gamma(3.0, 1.0)$"

    def intensity_chi2(df=3.0):
        rv = chi2(df=df)
        return rv.rvs()

    p3 = MixedPoissonProcess(intensity=intensity_chi2)
    t3 = "Mixed Poisson Process with $\\Lambda \sim \\chi^2(3.0)$"
    p4 = MixedPoissonProcess(intensity=intensity_chi2, intensity_kwargs={"df": 20})
    t4 = "Mixed Poisson Process with $\\Lambda \sim \\chi^2(20.0)$"

    qs = "https://raw.githubusercontent.com/quantgirluk/matplotlib-stylesheets/main/quant-pastel-light.mplstyle"
    plt.style.use(qs)

    for p, cm, t in [
        (p1, "terrain", t1),
        (p2, "RdPu", t2),
        (p3, "plasma", t3),
        (p4, "Blues", t4),
    ]:

        p.draw(
            N=300,
            T=5.0,
            figsize=(12, 7),
            style=qs,
            colormap=cm,
            envelope=False,
            title=t,
        )
#
#     p1.plot(N=10, T=10, figsize=(12, 7), style=qs, title=t1)
