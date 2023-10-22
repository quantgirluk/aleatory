import matplotlib.pyplot as plt
import numpy as np

from aleatory.processes import BESProcess, BESQProcess


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

    # sam = p.sample(n=100)
    # times = np.linspace(0, 1.0, 100)
    # plt.plot(times,sam)
    # plt.show()

    # sim = p.simulate(n=10, N=10)
    p.draw(n=100, N=200)
    p.plot(n=100, N=5)
