import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np

from aleatory.processes import BESProcess, BESQProcess
import math
import scipy.special as sp
import unittest


def bessel_marginal_formula(initial, x, t, dim):
    nu = (dim / 2.0) - 1.0
    special_term = sp.iv(nu, (initial * x) / t)
    result = (x / t) * ((x / initial) ** nu) * math.exp(-1.0 * (initial ** 2 + x ** 2) / (2.0 * t)) * special_term

    return result


def besq_marginal_formula(initial, x, t, dim):
    nu = (dim / 2.0) - 1.0
    special_term = sp.iv(nu, math.sqrt(initial * x) / t)
    result = (1.0 / (2.0 * t)) * ((x / initial) ** (nu / 2)) * math.exp(-1.0 * ((initial + x) / (2 * t))) * special_term

    return result


class TestBesselProcesses(unittest.TestCase):

    def test_process(self, dim=2, initial=1.0, T=1.0):
        for p in [BESProcess(dim=dim, initial=initial, T=T),
                  BESQProcess(dim=dim, initial=initial, T=T)]:
            sam = p.sample(n=100)
            self.assertEqual(len(sam), 100)
            self.assertEqual(sam[0], initial)

    # def test_process_simulate(self, dim=2, initial=1.0, T=1.0):
    #     for p in [
    #         BESProcess(dim=dim, initial=initial, T=T),
    #         BESQProcess(dim=dim, initial=initial, T=T)
    #     ]:
    #         simulation = p.simulate(n=100, N=100)
    #         last_points = [path[-1] for path in simulation]
    #         marginal = p.get_marginal(T)
    #
    #         plt.hist(last_points, density=True)
    #         x = np.linspace(marginal.ppf(0.001), marginal.ppf(0.999), 200)
    #         plt.plot(x, marginal.pdf(x))
    #         plt.show()

    def test_charts_simple(self, dim=2, initial=0.0, T=1.0):
        for p in [BESProcess(dim=dim, initial=initial, T=T),
                  BESQProcess(dim=dim, initial=initial, T=T)
                  ]:
            fig = p.plot(n=100, N=100, figsize=(12, 6))
            assert (isinstance(fig, matplotlib.figure.Figure))
            fig = p.draw(n=100, N=100, figsize=(12, 6))
            assert (isinstance(fig, matplotlib.figure.Figure))
            p.draw(n=100, N=100, envelope=True, orientation='vertical', figsize=(12, 6))

    def test_charts_complex(self, dim=2.5, initial=2.0, T=1.0):
        for p in [BESProcess(dim=dim, initial=initial, T=T)]:
            p.draw(n=100, N=100, envelope=True, figsize=(10, 6))
            # p.draw(n=200, N=200, envelope=True, orientation='vertical', figsize=(14, 6))

    def test_bessel_marginal(self, dim=2.5, initial=2.0, t=0.5, vis=False):

        for p, formula in [
            (BESProcess(dim=dim, initial=initial), bessel_marginal_formula),
            (BESQProcess(dim=dim, initial=initial), besq_marginal_formula)
        ]:
            X_1 = p.get_marginal(t=t)
            xs = np.linspace(0.001, X_1.ppf(0.99999), 200)

            values1 = [formula(initial, x, t, dim) for x in xs]
            values2 = [X_1.pdf(x) for x in xs]

            for (v1, v2) in zip(values1, values2):
                self.assertAlmostEqual(v1, v2)

            if vis:
                plt.plot(xs, values1, '-', lw=1.5, alpha=0.75, label=f'$pdf$ for $t$={t:.2f}')
                plt.plot(xs, values2, '-', lw=1.5, alpha=0.75, label=f'Formula')
                q05 = X_1.ppf(0.05)
                q95 = X_1.ppf(0.95)
                plt.axvline(q05)
                plt.axvline(q95)
                plt.axvline(X_1.mean(), color="red")
                plt.legend()
                plt.title(p.name + ' Marginal pdf')
                plt.show()

    def test_bessel_expectation(self, dim=2.5, initial=1.0, vis=False):

        times = np.linspace(0.01, 1, 100)

        for p in [BESProcess(dim=dim, initial=initial),
                  BESQProcess(dim=dim, initial=initial)]:

            marginals = [p.get_marginal(t=t) for t in times]
            means = [X.mean() for X in marginals]
            variances1 = [X.var() for X in marginals]
            expectations = [p.marginal_expectation(t) for t in times]
            variances2 = [p.marginal_variance(t) for t in times]

            for (m, e) in zip(means, expectations):
                self.assertAlmostEqual(m, e)

            for (v1, v2) in zip(variances1, variances2):
                self.assertAlmostEqual(v1, v2)

            if vis:
                plt.plot(times, means, label=p.name + ' marginal + mean')
                plt.plot(times, expectations, label='formula')
                plt.legend()
                plt.show()

                plt.plot(times, variances1, label=p.name + ' marginal + variance')
                plt.plot(times, variances2, label='formula')
                plt.legend()
                plt.show()
