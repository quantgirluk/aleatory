import matplotlib.pyplot as plt
import numpy as np

from aleatory.processes import BESProcess, BESQProcess
import math
import scipy.special as sp



def test_integer(dim=10):
    p = BESProcess(dim=dim)

    # sam = p.sample(n=10)
    # sim = p.simulate(n=10, N=10)
    p.draw(n=100, N=100)


def test_non_integer(dim=3.5):
    p = BESProcess(dim=dim)

    sam = p.sample(n=100)
    times = np.linspace(0, 1.0, 100)
    plt.plot(times, sam)
    plt.show()

    # sim = p.simulate(n=10, N=10)
    p.draw(n=100, N=200)


def test_non_integer_besq(dim=4.5):
    p = BESQProcess(dim=dim)
    p.draw(n=100, N=200)
    p.plot(n=100, N=5)


def test_initial_points(initial=5.0, dim=4.5):
    p = BESProcess(dim=dim, initial=initial)
    # p = BESQProcess(dim=dim, initial=initial, T=10)
    p.draw(n=100, N=100, envelope=False, orientation='vertical')
    p.draw(n=100, N=100, envelope=False, orientation='horizontal')
    p.draw(n=100, N=100, envelope=True)
    # p.draw(n=100, N=100, envelope=False, marginal=False)


def test_initial_points_float(initial=2.0, dim=4.5):
    p = BESProcess(dim=dim, initial=initial)
    # p = BESQProcess(dim=dim, initial=initial)
    p.draw(n=100, N=100)
    x1 = p.get_marginal(t=1)
    m1 = x1.mean()
    m2 = p.marginal_expectation(times=[1.0])
    diff = m1 - m2[0]


def bessel_marginal_formula(initial, x, t, dim):
    nu = (dim / 2.0) - 1.0
    spterm = sp.iv(nu, (initial * x) / t)
    result = (x / t) * ((x / initial) ** nu) * math.exp(-1.0 * (initial ** 2 + x ** 2) / (2.0 * t)) * spterm

    return result

def test_bessel_marginal(initial=2.0, dim=3, t=0.5):
    p = BESProcess(dim=dim, initial=initial)
    X_1 = p.get_marginal(t=t)
    xs = np.linspace(0.001, np.sqrt(X_1.ppf(0.999)), 200)
    test = [bessel_marginal_formula(initial, x, t, dim) for x in xs]
    mar = [X_1.pdf(x**2)*2.0*x for x in xs]
    plt.plot(xs, mar, '-', lw=1.5, alpha=0.75, label=f'$t$={t:.2f}')
    plt.plot(xs, test, '-', lw=1.5, alpha=0.75, label=f'TEST')
    alpha = (dim / 2.0) - 1.0
    nc = (initial ** 2) / t

    expectation = math.sqrt(math.pi / 2.0) * sp.eval_genlaguerre(0.5, alpha, ((-1.0 / 2.0) * nc))*np.sqrt(t)
    plt.axvline(expectation)

    plt.title(f'$X_{t}$ pdf')
    plt.legend()
    plt.show()


def test_bessel_expectations(T=100.0, initial=5.0, dim=3.5, n=200):
    times = np.linspace(0, T, n)
    times = times[1:]
    alpha = (dim / 2.0) - 1.0
    nc = (initial ** 2) / times
    expectations = math.sqrt(math.pi / 2.0) * sp.eval_genlaguerre(0.5, alpha, ((-1.0 / 2.0) * nc))*np.sqrt(times)
    variances = dim*times + initial**2 - expectations ** 2
    stds = np.sqrt(variances)
    plt.plot(times, expectations, label="expectations")
    plt.plot(times, stds, label="stds")
    plt.legend()
    plt.show()


def besq_marginal_formula(initial, x, t, dim):
    nu = (dim / 2.0) - 1.0
    spterm = sp.iv(nu, math.sqrt(initial * x)/t)
    result = (1.0 / (2.0*t)) * ((x / initial) ** (nu / 2)) * math.exp(-1.0 * ((initial + x) / (2 * t))) * spterm

    return result

def test_besq_marginal(initial=3.0, dim=9, t=5):
    p = BESQProcess(dim=dim, initial=initial)
    X_1 = p.get_marginal(t=t)
    xs = np.linspace(0, X_1.ppf(0.9999), 200)
    test = [besq_marginal_formula(initial, x, t, dim) for x in xs]
    plt.plot(xs, X_1.pdf(xs), '-', lw=1.5, alpha=0.75, label=f'$t$={t:.2f}')
    plt.plot(xs, test, '-', lw=1.5, alpha=0.75, label=f'Formula')
    plt.title(f'$X_{t}$ pdf')
    plt.legend()
    plt.show()
