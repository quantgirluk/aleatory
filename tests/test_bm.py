import unittest

import matplotlib.pyplot as plt

from aleatory.processes import BrownianMotion

import numpy as np


def draw_mean_variance(process, T):
    ts = np.linspace(0.1, T, T)
    means = process.marginal_expectation(ts)
    variances = process.marginal_variance(ts)
    fig, (ax1, ax2,) = plt.subplots(1, 2, figsize=(9, 4))

    ax1.plot(ts, means, lw=1.5, color='black', label='$E[X_t]$')
    ax1.set_xlabel('t')
    ax1.legend()
    ax2.plot(ts, variances, lw=1.5, color='red', label='$Var[X_t]$')
    ax2.set_xlabel('t')
    ax2.legend()
    fig.suptitle(
        'Expectation and Variance of $X_t$', size=12)
    plt.show()


class testBM(unittest.TestCase):
    vis = False

    @unittest.skipIf(not vis, "No Visualisation Required")
    def test_bm_cases(self):
        process = BrownianMotion()
        process.draw(n=100, N=100)

    @unittest.skipIf(not vis, "No Visualisation Required")
    def test_bm_exp_var_chart(self):
        bm = BrownianMotion()
        bma = BrownianMotion(drift=1.0, scale=1.0, initial=1.0)

        for process in [bm, bma]:
            draw_mean_variance(process, 100)

    def test_bm_expectation(self):
        process = BrownianMotion(initial=1.0, drift=2.0, scale=1.0)
        process.sample(n=100)
        exp1 = process.process_expectation()
        ts = process.times
        exp2 = process.marginal_expectation(ts)
        for e1, e2 in zip(exp1, exp2):
            assert (e1 == e2)

    @unittest.skipIf(not vis, "No Visualisation Required")
    def test_bm_sample_at(self):
        process = BrownianMotion(initial=1.0, drift=0., scale=1.0)
        sample = process.sample_at(times=[1., 2., 3., 4., 5., 10, 20, 30])
        print(sample)
        plt.plot(process.times, sample)
        plt.show()
