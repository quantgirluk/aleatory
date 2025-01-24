"""Poisson Process"""

import numpy as np

from aleatory.processes.base import BaseProcess
from aleatory.utils.plotters import plot_poisson, draw_poisson_like


class InhomogeneousPoissonProcess(BaseProcess):
    r"""
    Inhomogeneous Poisson Process
    =============================

    .. image:: ../_static/poisson_non_homogeneous_draw.png


    Notes
    -----

    An inhomogeneous Poisson point process is a type of random mathematical object that consists of points randomly
    located on a mathematical space with the essential feature that the points occur independently of one another.

    More precisely, let :math:`\lambda(t):[0,\infty) \mapsto :[0,\infty)` be an integrable function. A inhomogeneou (or
    non-homogeneous) Poisson process :math:`\{N(t) : t\geq 0\}` with intensity rate :math:`\lambda(t)`, is defined by the
    following properties:

    - :math:`N(0)=0`,
    - :math:`N(t)` has independent increments,
    - :math:`N(t)` has a Poisson distribution with parameter :math:`\Lambda (t) = \int_0^t \lambda(s)ds`, for each :math:`t> 0`.

    Constructor, Methods, and Attributes
    ------------------------------------

    """

    def __init__(self, intensity, rng=None):
        """
        :parameter callable intensity: a callable object which defines the intensity of the Poisson process
        :parameter numpy.random.Generator rng: a custom random number generator
        """
        super().__init__(rng=rng)
        self.name = "Inhomogeneous Poisson Process"
        self.intensity = intensity
        self.T = None
        self.N = None
        self.paths = None
        self.max_lambda = None

    def __str__(self):
        return f"Inhomogeneous Poisson Process"

    def __repr__(self):
        return "Inhomogeneous Poisson Process"

    @property
    def intensity(self):
        return self._intensity

    @intensity.setter
    def intensity(self, value):
        if not callable(value):
            raise ValueError("The Intensity Function must be a callable")
        self._intensity = value

    def _estimate_max_lambda(self):
        # Estimate max_lambda numerically

        if self.max_lambda:
            return self.max_lambda
        else:
            t_grid = np.linspace(0, self.T, 500)
            max_lambda = np.max(self.intensity(t_grid))
            self.max_lambda = max_lambda
            return max_lambda

    def _sample_in_poisson_process(self):
        """
        Simulate a non-homogeneous Poisson process (NHPP) on the interval [0, T].
        Returns:
            np.ndarray
                The simulated event times.
        """
        max_lambda = self._estimate_max_lambda()

        event_times = [0.0]
        t = 0.0
        while t < self.T:
            # Step 1: Simulate candidate inter-arrival time from homogeneous Poisson process
            u = self.rng.uniform()
            candidate_interarrival = -np.log(u) / max_lambda
            t += candidate_interarrival

            # if t >= self.T:
            #     break

            # Step 2: Accept or reject with probability \lambda(t) / max_lambda
            acceptance_prob = self.intensity(t) / max_lambda
            u2 = self.rng.uniform()
            if u2 < acceptance_prob:
                event_times.append(t)

        return np.array(event_times)

    def sample(self, T):
        self.T = T
        sample = self._sample_in_poisson_process()
        self.max_lambda = None
        return sample

    def simulate(self, N, T):
        """
        Simulate paths/trajectories from the instanced stochastic process. The function returns :math:`N` paths
        over the time :math:`[0,T]`. Note that in this case each path can have a different number of jumps.

        :param int N: number of paths to simulate
        :param float T: length of the time interval to simulate
        :return: list with N paths (each one is a numpy array of size n)

        """
        self.N = N
        self.T = T
        self.paths = [self._sample_in_poisson_process() for _ in range(N)]
        return self.paths

    def plot(self, N, T=10.0, title=None, **fig_kwargs):

        paths = self.simulate(N, T=T)
        plot_title = title if title else self.name

        return plot_poisson(T=T, paths=paths, title=plot_title, **fig_kwargs)

    def draw(
        self,
        N,
        T=10.0,
        style="seaborn-v0_8-whitegrid",
        colormap="RdYlBu_r",
        mode="steps",
        title=None,
        **fig_kw,
    ):

        title = title if title else self.name
        self.simulate(N=N, T=T)
        paths = self.paths
        fig = draw_poisson_like(
            T, paths, title=title, style=style, colormap=colormap, mode=mode, **fig_kw
        )
        return fig


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    def myfunction1(s):
        return s

    p1 = InhomogeneousPoissonProcess(intensity=myfunction1)
    title1 = f"Inhomogeneous Poisson Process $\\lambda(t)=t$"

    def myfunction2(s):
        return s**2

    p2 = InhomogeneousPoissonProcess(intensity=myfunction2)
    title2 = f"Inhomogeneous Poisson Process $\\lambda(t)=t^2$"

    def myfunction3(s):
        return 3 * np.exp(s)  # Example: periodic intensity
        # return 3 * np.sin(2 * np.pi * s)  # Example: periodic intensity

    p3 = InhomogeneousPoissonProcess(intensity=myfunction3)
    title3 = f"Inhomogeneous Poisson Process $\\lambda(t)=3 e^t$"

    def myfunction4(s):
        return 5 + 2.0 * np.sin(2 * np.pi * s)  # Example: periodic intensity

    p4 = InhomogeneousPoissonProcess(intensity=myfunction4)
    title4 = f"Inhomogeneous Poisson Process $\\lambda(t)=5 + 2\\sin(2\\pi t)$"

    def myfunction5(s):
        return (1.0 / (s + 1.0)) + 3.0  # Example: periodic intensity

    p5 = InhomogeneousPoissonProcess(intensity=myfunction5)
    title5 = "Inhomogeneous Poisson Process $\\lambda(t)=\\frac{1}{t+10} + 3$"

    qs = "https://raw.githubusercontent.com/quantgirluk/matplotlib-stylesheets/main/quant-pastel-light.mplstyle"

    plt.style.use(qs)

    for p, cm, t in [
        (p1, "twilight", title1),
        (p2, "RdPu", title2),
        (p3, "viridis", title3),
        (p4, "Blues", title4),
        (p5, "OrRd", title5),
    ]:

        p.draw(
            N=100,
            T=5.0,
            figsize=(12, 7),
            style=qs,
            colormap=cm,
            envelope=False,
            title=t,
        )
#
#     p1.plot(N=5, T=5.0, figsize=(12, 7), style=qs, title=title1)
