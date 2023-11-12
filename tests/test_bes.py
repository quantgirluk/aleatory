import matplotlib.pyplot as plt
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

    def test_charts_simple(self, dim=2, initial=0.0, T=1.0):
        for p in [BESProcess(dim=dim, initial=initial, T=T),
                  BESQProcess(dim=dim, initial=initial, T=T)]:
            p.draw(n=100, N=100, figsize=(10, 6))
            p.draw(n=100, N=100, envelope=True, orientation='vertical', figsize=(12, 6))

    def test_charts_complex(self, dim=1.5, initial=2.0, T=1.0):
        for p in [BESProcess(dim=dim, initial=initial, T=T)]:
            p.draw(n=100, N=100, figsize=(12, 6))
            p.draw(n=100, N=100, envelope=True, orientation='vertical', figsize=(12, 6))

    def test_bessel_marginal(self, dim=2.5, initial=1.0, t=1.5):
        p = BESProcess(dim=dim, initial=initial)
        X_1 = p.get_marginal(t=t)
        xs = np.linspace(0.001, np.sqrt(X_1.ppf(0.999)), 200)
        test = [bessel_marginal_formula(initial, x, t, dim) for x in xs]
        mar = [X_1.pdf(x ** 2) * 2.0 * x for x in xs]

        for (t, m) in zip(test, mar):
            self.assertAlmostEqual(t, m)

        # plt.plot(xs, mar, '-', lw=1.5, alpha=0.75, label=f'$pdf$ for $t$={t:.2f}')
        # plt.plot(xs, test, '-', lw=1.5, alpha=0.75, label=f'Formula')
        # plt.legend()
        # plt.show()
        # alpha = (dim / 2.0) - 1.0
        # nc = (initial ** 2) / t

    def test_besq_marginal(self, dim=2.5, initial=1.0, t=1.5):
        p = BESQProcess(dim=dim, initial=initial)
        X_1 = p.get_marginal(t=t)
        xs = np.linspace(0, X_1.ppf(0.99), 200)
        test = [besq_marginal_formula(initial, x, t, dim) for x in xs]
        mar = X_1.pdf(xs)

        for (t, m) in zip(test, mar):
            self.assertAlmostEqual(t, m)

        plt.plot(xs, mar, '-', lw=1.5, alpha=0.75, label=f'$pdf$ for $t$={t:.2f}')
        plt.plot(xs, test, '-', lw=1.5, alpha=0.75, label=f'Formula')
        plt.legend()
        plt.show()


def test_bessel_expectations(T=100.0, initial=5.0, dim=3.5, n=200):
    times = np.linspace(0, T, n)
    times = times[1:]
    alpha = (dim / 2.0) - 1.0
    nc = (initial ** 2) / times
    expectations = math.sqrt(math.pi / 2.0) * sp.eval_genlaguerre(0.5, alpha, ((-1.0 / 2.0) * nc)) * np.sqrt(times)
    variances = dim * times + initial ** 2 - expectations ** 2
    stds = np.sqrt(variances)
    plt.plot(times, expectations, label="expectations")
    plt.plot(times, stds, label="stds")
    plt.legend()
    plt.show()
