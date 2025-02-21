import numpy as np
from aleatory.processes.jump.poisson_nonhomogeneous import (
    InhomogeneousPoissonProcess,
)


class HawkesProcess(InhomogeneousPoissonProcess):
    r"""
    Hawkes process
    ==============

    .. image:: ../_static/hawkes_draw.png

    Notes
    -----

    A Hawkes process is a counting process :math:`N(t)` whose conditional intensity process is given by

    .. math::
        \lambda^{\ast}(t) = \mu + \sum_{T_i <t} \phi(t-T_i), \qquad t\geq 0,

    where

    - :math:`\mu>0` is the baseline intensity (rate of events in the absence of previous events) or background arrival rate

    - :math:`\phi: \mathbb{R}^{+}\rightarrow \mathbb{R}`, is the excitation function or triggering kernel, a non-negative function representing the influence of past events, and

    - :math:`T_i`, are the times of prior events.

    In particular, we assume

    .. math::
        \phi(t) = \alpha \exp(-\beta t)

    i.e. an exponentially decaying excitation function.


    Constructor, Methods, and Attributes
    ------------------------------------

    """

    def __init__(self, mu=1.0, alpha=1.0, beta=1.0, rng=None):
        r"""
        :parameter double mu: the baseline intensity
        :parameter double alpha: the :math:`\alpha >0` in the excitation function above
        :parameter double beta: the :math:`\beta >0` in the excitation function above
        :parameter double beta: the :math:`\beta >0` in the excitation function above
        :parameter numpy.random.Generator rng: a custom random number generator
        """

        def constant():
            return 1.0

        super().__init__(rng=rng, intensity=constant)
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.name = (
            f"Hawkes Process $N(t)$ with conditional Intensity process\n"
            f"$\\lambda_t^{{\star}} = {self.mu} + \sum_{{T_i < t}} {self.alpha}\exp(-{self.beta}(t-T_i)) $\n"
        )
        self.T = None
        self.N = None
        self.paths = None

    def __str__(self):
        return f"Hawkes Process with intensity rate"

    def __repr__(self):
        return f"Hawkes Process"

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        if value < 0:
            raise ValueError("mu must be positive")
        self._mu = value

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if value < 0:
            raise ValueError("alpha must be positive")
        self._alpha = value

    def _sample_hawkes_process(self, T):
        """
        Simulate a uni-variate Hawkes process using Ogata's thinning method.

        Parameters:
            T (float): Time horizon for the simulation.

        Returns:
            events (list): List of event times.
        """
        # Initialize variables
        events = []  # List to store event times
        t = 0  # Current time
        lambda_t = self.mu  # Initial intensity

        while t < T:
            # Step 1: Generate candidate point from an upper-bound homogeneous Poisson process
            u = np.random.uniform()
            t = t - np.log(u) / lambda_t  # Inverse transform sampling

            # Break if the new event time exceeds T
            if t > T:
                break

            # Step 2: Calculate the conditional intensity at the proposed time
            lambda_star = self.mu + sum(
                self.alpha * np.exp(-self.beta * (t - ti)) for ti in events
            )

            # Step 3: Accept or reject the point based on the ratio
            u2 = np.random.uniform()
            if u2 <= lambda_star / lambda_t:
                # Accept the event
                events.append(t)
                lambda_t = lambda_star  # Update the intensity
            else:
                # Reject the event; keep the current intensity
                lambda_t = lambda_star

        return events

    def sample(self, T=None):
        return self._sample_hawkes_process(T=T)

    def simulate(self, N, T=10.0):
        """
        Simulate paths/trajectories from the instanced stochastic process.
        The function returns :math:`N` paths over the time :math:`[0,T]`. Note
        each path can have a different number of jumps.

        :param int N: number of paths to simulate
        :param float T: time horizon for the simulation
        :return: list with N paths (each one is a numpy array of size n)

        """
        self.N = N
        self.paths = [self.sample(T=T) for _ in range(N)]
        return self.paths


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    qs = "https://raw.githubusercontent.com/quantgirluk/matplotlib-stylesheets/main/quant-pastel-light.mplstyle"
    plt.style.use(qs)

    p = HawkesProcess(mu=0.5, alpha=1.0, beta=2.0)
    p.draw(
        N=200,
        T=10.0,
        figsize=(12, 7),
        style=qs,
        colormap="terrain",
        mode="steps+points",
        envelope=False,
    )


#     p2 = HawkesProcess(mu=1.0, alpha=1.0, beta=2.0)
#     p3 = HawkesProcess(mu=1.0, alpha=0.5, beta=0.5)
#     p4 = HawkesProcess(mu=0.25, alpha=1.0, beta=0.5)
#
#     for p, cm in [
#         (p1, "terrain"),
#         (p2, "RdPu"),
#         (p3, "plasma"),
#         (p4, "Blues"),
#     ]:
#
#         p.draw(
#             N=200,
#             T=10.0,
#             figsize=(12, 7),
#             style=qs,
#             colormap=cm,
#             mode="steps+points",
#             envelope=False,
#         )
#
#     p1.plot(N=10, T=10, figsize=(12, 7), style=qs)

# for alpha in [0.5, 1.0, 2.0]:
#     p = HawkesProcess(mu=0.5, alpha=1.0, beta=2.0)
#     # p.plot(N=100, T=10.0, figsize=(10, 6), style=qs)
#     p.draw(N=200, T=10.0, figsize=(12, 6), style=qs)
# p = HawkesProcess(mu=0.5, alpha=1.0, beta=0.5)
# p.plot(N=10, T=10.0, figsize=(10, 6), style=qs)
