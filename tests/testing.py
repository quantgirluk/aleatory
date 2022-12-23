import unittest
import matplotlib.pyplot as plt
import numpy as np
from aleatory.processes import BrownianMotion, GBM, Vasicek, OUProcess, CIRProcess, CEVProcess
from parameterized import parameterized, parameterized_class


def test_sample(self):
    sample = self.process.sample(self.n)
    times = self.process.times
    self.assertEqual(len(times), len(self.grid_times))
    for (t1, t2) in zip(times, self.grid_times):
        self.assertEqual(t1, t2)
    plt.plot(times, sample)
    plt.show()


def test_sample_at(self):
    sample = self.process.sample_at(self.times_given)
    times = self.process.times
    for (t1, t2) in zip(times, self.times_given):
        self.assertEqual(t1, t2)
    self.assertEqual(len(times), len(sample))
    plt.plot(times, sample)
    plt.show()
    # def setUp(self) -> None:
    #     self.N = 100
    #     self.n = 100
    #     self.T = 1.0
    #     self.times_given = np.linspace(0, 1.0, 100, endpoint=True)
    #     self.grid_times = np.linspace(0, self.T, self.n)


class TestProcesses(unittest.TestCase):
    bm = BrownianMotion()
    bmd = BrownianMotion(drift=-1.0, scale=0.5)
    gbm = GBM(drift=1.0, volatility=0.5)
    vasicek = Vasicek(theta=1.5, mu=1.0, sigma=0.6, initial=4.0)
    ouprocess = OUProcess(theta=1.5, sigma=0.6, initial=4.0)
    cirprocess = CIRProcess(theta=0.06, mu=0.01, sigma=0.009)
    cev = CEVProcess(gamma=0.5, mu=1.50, sigma=0.6, initial=1.0)

    @parameterized.expand([
        [bm],
        [bmd], [gbm],
        [vasicek], [ouprocess], [cirprocess],
        [cev]
    ])
    def test_charts(self, process):
        process.plot(n=100, N=5)
        process.draw(n=100, N=200, envelope=False)
        # process.draw(n=100, N=200, envelope=True)
        # process.draw(n=100, N=200, marginal=False)
        # process.draw(n=100, N=200, marginal=False, envelope=True)

if __name__ == '__main__':
    unittest.main(verbosity=2)
