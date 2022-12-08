import unittest
import matplotlib.pyplot as plt
import numpy as np
from replica.processes.exact_solution.brownian_motion import BrownianMotion
from replica.sp_paths.brownian_motion_paths import BrownianPaths
from replica.processes.exact_solution.gaussian import GaussianIncrements
from replica.processes.exact_solution.geometric_brownian_motion import GeometricBrownianMotion
from replica.sp_paths.geometric_brownian_motion_paths import GBMPaths
from replica.processes.euler_maruyama.ou_process import OUProcess
from replica.sp_paths.ou_process_paths import OUProcessPaths
from replica.processes.euler_maruyama.cir_process import CIRProcess
from replica.sp_paths.cir_process_paths import CIRProcessPaths
from replica.processes.euler_maruyama.cev_process import CEV_process
from replica.sp_paths.cev_process_paths import CEVProcessPaths


class Pars:

    def __init__(self, initial, drift, vol, theta, mu, sigma, T):
        self.initial = initial
        self.drift = drift
        self.vol = vol
        self.theta = theta
        self.vol = vol
        self.mu = mu
        self.sigma = sigma
        self.T = T


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


class TestProcess(unittest.TestCase):
    def setUp(self):
        self.n = 100
        self.T = 1.0
        self.process = None
        self.times_given = np.linspace(0, 1, 100, endpoint=True)
        self.grid_times = np.linspace(0, self.T, self.n)

    def test_Gaussian(self):
        self.process = GaussianIncrements(T=self.T)
        test_sample(self)
        test_sample_at(self)

    def test_BrownianMotion(self):
        self.process = BrownianMotion(T=self.T, drift=-1.0, scale=2.0)
        test_sample(self)
        test_sample_at(self)

    def test_GeometricBrownianMotion(self):
        self.process = GeometricBrownianMotion(T=self.T, drift=-2.0, volatility=0.5)
        test_sample(self)
        test_sample_at(self)

    def test_OUProcess(self):
        self.process = OUProcess(T=self.T, theta=0.7, mu=1.50, sigma=0.06, initial=4.0)
        test_sample(self)

    def test_CIRProcess(self):
        self.process = CIRProcess(T=self.T, theta=0.06, mu=0.01, sigma=0.009)
        test_sample(self)

    def test_CIRProcess_params(self):
        with self.assertRaises(ValueError):
            self.process = CIRProcess(T=self.T, theta=-0.06, mu=0.01, sigma=0.009)
        with self.assertRaises(ValueError):
            self.process = CIRProcess(T=self.T, theta=1.0, mu=1.0, sigma=3.0)

    def test_CEVProcess(self):
        self.process = CEV_process(T=self.T, gamma=1.0, mu=2.0, sigma=1.0)
        test_sample(self)

    def test_CEVProcess_params(self):
        with self.assertRaises(ValueError):
            self.process = CEV_process(T=self.T, gamma=-1.0, mu=2.0, sigma=1.0)
        with self.assertRaises(ValueError):
            self.process = CEV_process(T=self.T, gamma=1.0, mu=2.0, sigma=-1.0)


class TestPaths(unittest.TestCase):

    def setUp(self) -> None:
        self.N = 100
        self.n = 100
        self.T = 1.0
        self.times_given = np.linspace(0, 1.0, 100, endpoint=True)
        self.grid_times = np.linspace(0, self.T, self.n)

    def test_BrownianMotionPaths(self):
        BMP = BrownianPaths(N=100, times=self.times_given)
        BMP.plot()
        BMP.draw()
        BMP.draw_envelope()
        BMP.draw_envelope_std()

    def test_BrownianPaths(self):
        BMP = BrownianPaths(N=100, times=self.times_given, drift=2.0, scale=0.5)
        BMP.plot()
        BMP.draw()
        BMP.draw_envelope()
        BMP.draw_envelope_std()

    def test_GBMPaths(self):
        # GBM = GBMPaths(N=self.N, times=self.times_given, drift=-2.0, volatility=0.5, initial=2.0)
        # GBM.plot()
        # GBM.draw()
        # GBM.draw_envelope()
        GBM = GBMPaths(N=self.N, times=self.times_given, drift=1.0, volatility=0.5, initial=1.5)
        GBM.plot()
        GBM.draw()
        GBM.draw_envelope()

    def test_OUPaths(self):
        OUP = OUProcessPaths(N=self.N, n=self.n, theta=1.5, mu=1.0, sigma=0.6, initial=4.0)
        OUP.plot()
        OUP.draw()
        OUP.draw_envelope()

    def test_CIRPaths(self):
        CIR = CIRProcessPaths(N=self.N, n=self.n, theta=2.5, mu=1.50, sigma=0.6)
        # CIR = CIRProcessPaths(N=self.N, n=self.n, theta=0.5, mu=4.50, sigma=0.5, initial=1.0)
        CIR.plot()
        CIR.draw()
        CIR.draw_envelope()

    def test_CEVPaths(self):
        CEV = CEVProcessPaths(N=self.N, n=self.n, gamma=0.5, mu=1.50, sigma=0.6, initial=1.0)
        # CEV = CEVProcessPaths(N=self.N, n=self.n, gamma=1.0, mu=2.0, sigma=1.0, initial=1.0)
        CEV.plot()
        CEV.draw()


if __name__ == '__main__':
    unittest.main(verbosity=2)
