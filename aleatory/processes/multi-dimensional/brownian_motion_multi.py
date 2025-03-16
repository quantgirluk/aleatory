import numpy as np

from aleatory.processes.base import StochasticProcess
from aleatory.utils.utils import get_times


class MultiDimensionalStochasticProcess(StochasticProcess):
    def __init__(self, initial=0.0, name=None, T=1.0, rng=None):
        super().__init__(T=T, rng=rng)
        self.initial = initial
        self.name = name
        self.n = None
        self.times = None
        self.N = None
        self.paths = None


class MultiDimensionalBM(MultiDimensionalStochasticProcess):
    def __init__(self, d, correlation=None, T=1.0, rng=None):
        super().__init__(T=T, rng=rng)
        self.d = d
        self.correlation = correlation if correlation is not None else np.eye(d)
        self.correlated_case = (
            True if not np.allclose(self.correlation, np.eye(self.d)) else False
        )

    def _sample(self, n):
        dt = self.T / n  # Time step size

        # Generate increments from a normal distribution
        dW = np.sqrt(dt) * np.random.randn(n - 1, self.d)

        if self.correlated_case:
            L = np.linalg.cholesky(self.correlation)  # Cholesky decomposition
            dW = dW @ L.T  # Apply transformation to introduce correlation

        L = np.linalg.cholesky(self.correlation)  # Cholesky decomposition
        dW = dW @ L.T  # Introduce correlation

        # Initialize Brownian motion paths
        W = np.zeros((n, self.d))
        W[1:] = np.cumsum(dW, axis=0)

        return W

    def sample(self, n):
        self.n = n
        sample = self._sample(n)
        self.times = get_times(self.T, self.n)
        return sample

    def simulate(self, n, N):

        self.times = get_times(self.T, self.n)  # Time grid
        correlation_matrix = self.correlation
        dt = self.T / n  # Time step size

        # Generate independent Brownian increments for all paths
        dW = np.sqrt(dt) * np.random.randn(N, n - 1, self.d)

        if self.correlated_case:
            L = np.linalg.cholesky(correlation_matrix)  # Cholesky decomposition
            dW = dW @ L.T

        # Compute cumulative sum to obtain Brownian paths
        W = np.zeros((N, n, self.d))
        W[:, 1:, :] = np.cumsum(dW, axis=1)

        return W


# p = MultiDimensionalBM(d=2, T=1)
# s = p.sample(n=20)
# sim = p.simulate(n=20, N=5)
