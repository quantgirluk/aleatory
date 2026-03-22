import numpy as np

from aleatory.processes.base import StochasticProcess
from matplotlib import pyplot as plt

class Heston(StochasticProcess):
    """
    Heston stochastic volatility model.

    dS_t = mu S_t dt + sqrt(v_t) S_t dW_1
    dv_t = kappa(theta - v_t) dt + sigma sqrt(v_t) dW_2

    corr(dW_1, dW_2) = rho
    """

    def __init__(self, rho=0.0, mu=1.0, kappa=1.0, theta=0.04, sigma=0.2, v0=0.04, s0=1.0, T=1.0):
        super().__init__(T=T)
        self.mu = mu
        self.rho = rho
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.v0 = v0
        self.S0 = s0
        self.name = f"Heston Model with $\\rho={self.rho}$"
        self.n = None
        self.times = None
        self.S_paths = None
        self.v_paths = None

    def simulate(self, n, N):
        self.n_paths = N
        self.steps = n
        self.dt = self.T / n
        self.times = np.linspace(0, self.T, n + 1)

        S = np.zeros((self.n_paths, self.steps + 1))
        v = np.zeros((self.n_paths, self.steps + 1))

        S[:, 0] = self.S0
        v[:, 0] = self.v0

        for t in range(1, self.steps + 1):

            z1 = np.random.normal(size=self.n_paths)
            z2 = np.random.normal(size=self.n_paths)

            # Correlated Brownian motions
            W1 = z1
            W2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2

            vt_prev = np.maximum(v[:, t-1], 0)

            v[:, t] = (
                v[:, t-1]
                + self.kappa * (self.theta - vt_prev) * self.dt
                + self.sigma * np.sqrt(vt_prev * self.dt) * W2
            )

            v[:, t] = np.maximum(v[:, t], 0)

            S[:, t] = (
                S[:, t-1]
                * np.exp(
                    (self.mu - 0.5 * vt_prev) * self.dt
                    + np.sqrt(vt_prev * self.dt) * W1
                )
            )

        self.S_paths = S
        self.v_paths = v
        return S, v

    def plot(self, n=100, N=10, title=None, suptitle=None, **fig_kw):
        """Visualise simulated price and variance paths."""

        self.simulate(n, N)
        chart_suptitle = suptitle if suptitle is not None else self.name

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Price paths
        for s_path in self.S_paths:
            axes[0].plot(self.times, s_path)

        axes[0].set_title("Heston Model - Asset Price Paths")
        axes[0].set_ylabel("Price")

        # Variance paths
        for v_path in self.v_paths:
            axes[1].plot(self.times, v_path)

        axes[1].set_title("Variance Paths")
        axes[1].set_ylabel("Variance")
        axes[1].set_xlabel("Time")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    p = Heston(rho=-0.5, kappa=2.0, theta=0.04, sigma=0.3, v0=0.04, s0=1.0, T=1.0)
    p.plot(n=200, N=100, title="Heston Model Simulation", suptitle="Heston Model with $\\rho=-0.5$")