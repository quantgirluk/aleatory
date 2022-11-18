import unittest

import matplotlib.pyplot as plt
import numpy as np

from brownian_motion import BrownianMotion
from brownian_motion_paths import BrownianPaths
from gaussian import Gaussian
from geometric_brownian_motion import GeometricBrownianMotion
from geometric_brownian_motion_paths import GBMPaths
from ou_process import OUProcess
from ou_process_paths import OUProcessPaths
from cir_process import CIRProcess
from cir_process_paths import CIRProcessPaths

class TestProcesses(unittest.TestCase):
    def test_Gaussian(self):
        process = Gaussian()
        n = 100
        sample = process.sample(n)
        times = process.times
        check_times = np.linspace(0, 1, n)
        for (t1, t2) in zip(times, check_times):
            self.assertEqual(t1, t2)
        plt.plot(times, sample)
        plt.show()

        process = Gaussian()
        times = np.arange(1, 11, 1)
        sample = process.sample_at(times)
        plt.plot(times, sample)
        plt.show()

    def test_BrownianMotion(self):
        my_times = np.linspace(0, 5, 100, endpoint=True)
        process = BrownianMotion(drift=-1.0, scale=2.0)
        for k in range(10):
            sample = process.sample_at(my_times)
            plt.plot(my_times, sample)
        plt.show()
        self.assertEqual(len(my_times), len(process.times))
        for t, s in zip(my_times, process.times):
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
            times = process.times
            plt.plot(times, sample)
        plt.show()

    def test_CIRProcess(self):
        process = CIRProcess(T=1000.0, theta=0.06, mu=0.01, sigma=0.009)
        n = 1000

        for k in range(2):
            sample = process.sample(n)
            times = process.times
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
        GBMP = GBMPaths(N=200, drift=2.0, volatility=0.5, initial=1.0, times=my_times)
        GBMP.plot()
        GBMP.draw()

    def test_OUPaths(self):

        OUP = OUProcessPaths(N=100, n=200, theta=2.5, mu=1.50, sigma=0.6)
        OUP.plot()
        OUP.draw()

    def test_CIRPaths(self):

        OUP = CIRProcessPaths(N=100, n=10, theta=2.5, mu=1.50, sigma=0.6)
        OUP.plot()
        OUP.draw()


if __name__ == '__main__':
    unittest.main(verbosity=2)
