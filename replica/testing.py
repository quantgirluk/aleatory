import unittest

import matplotlib.pyplot as plt
import numpy as np

from brownian_motion import BrownianMotion
from brownian_motion_paths import BrownianPaths
from gaussian import Gaussian
from geometric_brownian_motion import GeometricBrownianMotion
from geometric_brownian_motion_paths import GBMPaths
from ou_process import OUProcess


# my_process = Gaussian()
# n =200
#
# print(my_process.sample_at(times=[1,2,3, 4.5]))
# plt.plot(my_process.times(n=n), my_process.sample(n=n))
# plt.show()

# n = 200
# for k in range(2):
#     my_brownian = BrownianMotion(drift=-1.0, scale=2.0, T=5.0)
#     sample = my_brownian.sample(n=n)
#     print(type(sample))
#     times = my_brownian.times
#     plt.plot(times, sample)
# plt.show()

class TestProcesses(unittest.TestCase):
    def test_Gaussian(self):
        process = Gaussian()
        n = 100
        times = process.times(n)
        sample = process.sample(n)

        check_times = np.linspace(0, 1, n)
        for (t1, t2) in zip(times, check_times):
            self.assertEqual(t1, t2)
        plt.plot(times, sample)
        plt.show()

        process = Gaussian()
        times = np.arange(1, 11, 1)
        sample = process.sample_at(times)
        print(process._times)

        plt.plot(times, sample)
        plt.show()

    def test_BrownianMotion(self):
        my_times = np.linspace(0, 5, 100, endpoint=True)
        for k in range(10):
            my_brownian = BrownianMotion(drift=-1.0, scale=2.0)
            sample = my_brownian.sample_at(my_times)
            plt.plot(my_times, sample)
        plt.show()
        self.assertEqual(len(my_times), len(my_brownian.times))
        for t, s in zip(my_times, my_brownian.times):
            self.assertEqual(t, s)

    def test_GeometricBrownianMotion(self):
        my_times = np.linspace(0, 1, 100, endpoint=True)
        for k in range(20):
            my_process = GeometricBrownianMotion(drift=2.0, volatility=0.5)
            sample = my_process.sample_at(my_times)
            plt.plot(my_times, sample)
        plt.show()

    def test_OUProcess(self):
        process = OUProcess(T=10.0, theta=0.7, mu=1.50, sigma=0.06)
        n = 100

        for k in range(200):
            sample = process.sample(n)
            times = process._times
            plt.plot(times, sample)
        plt.show()


class TestPaths(unittest.TestCase):
    def test_BrownianPaths(self):
        my_times = np.linspace(0, 1, 100, endpoint=True)
        BMP = BrownianPaths(N=100, times=my_times, drift=4.0, scale=1.5)
        BMP.plot()
        BMP.draw()

    def test_GBMPaths(self):
        my_times = np.linspace(0, 1, 250, endpoint=True)
        GBMP = GBMPaths(N=100, drift=.4, volatility=0.2, initial=1.0, times=my_times)
        GBMP.plot()
        GBMP.draw()


if __name__ == '__main__':
    unittest.main(verbosity=2)
