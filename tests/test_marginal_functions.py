from aleatory.processes import BrownianMotion
import numpy as np
import pytest


class TestMarginalFunctions:

    def test_marginals_BM(self,drif=1.0, scale=2.0):
        grid_times = np.linspace(0, 1.0, 100)
        process = BrownianMotion(drift=drif, scale=scale)
        means = process.marginal_expectation(grid_times)
        variances = process.marginal_variance(grid_times)

        for m, t in zip(means, grid_times):
            assert m == drif*t

        for v, t in zip(variances, grid_times):
            assert v == (scale ** 2) * t

        process.draw(n=100, N=50)
        means = process.marginal_expectation()
        for m, t in zip(means, process.times):
            assert m == drif*t

