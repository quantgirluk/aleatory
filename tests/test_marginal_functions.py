from aleatory.processes import BrownianMotion
import numpy as np


class TestMarginalFunctions:

    def test_marginals_BM(self, drift=1.0, scale=2.0):
        grid_times = np.linspace(0, 1.0, 100)

        process = BrownianMotion(drift=drift, scale=scale)
        process.sample(n=100)
        means = process.marginal_expectation(grid_times)
        variances = process.marginal_variance(grid_times)

        for m, t in zip(means, grid_times):
            assert m == drift * t

        for v, t in zip(variances, grid_times):
            assert v == (scale ** 2) * t

        for m, t in zip(means, process.times):
            assert m == drift * t
