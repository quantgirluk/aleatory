import unittest
import numpy as np
from brownian_motion_paths import BrownianPaths
from geometric_brownian_motion_paths import GBMPaths
from geometric_brownian_motion import GeometricBrownianMotion
from gaussian import Gaussian
from brownian_motion import BrownianMotion
import matplotlib.pyplot as plt


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


class TestPaths(unittest.TestCase):
    def test_BrownianPaths(self):
        my_times = np.linspace(0, 1, 100, endpoint=True)
        BMP = BrownianPaths(N=200, times=my_times, drift=4.0, scale=1.5)
        BMP.plot()
        BMP.draw()

    def test_GBMPaths(self):
        my_times = np.linspace(0, 1, 100, endpoint=True)
        GBMP = GBMPaths(N=200, drift=1.0, volatility=0.25, initial=1.0, times=my_times)
        GBMP.plot()
        GBMP.draw()



if __name__ == '__main__':
    unittest.main(verbosity=2)
