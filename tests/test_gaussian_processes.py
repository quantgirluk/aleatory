import matplotlib.figure
import numpy as np
import unittest

from aleatory.processes import WhiteNoise, GPLinear, GPConstant, GPRBF


class TestGaussianProcesses(unittest.TestCase):

    def test_gaussian_processes(self):
        processes = [
            WhiteNoise(sigma=1.0, T=1.0),
            GPLinear(sigma=1.0, T=1.0),
            GPConstant(sigma=1, T=1.0),
            GPRBF(length_scale=0.3, sigma=1.0, T=1.0),
        ]

        for g in processes:
            g.times = np.linspace(0, g.T, 100)
            sample = g.sample(n=100)
            assert len(sample) == 100
            assert isinstance(sample, np.ndarray)
            assert isinstance(g.covariance_function(g.times), np.ndarray)
            assert isinstance(g.variance_function(g.times), np.ndarray)

    def test_gaussian_processes_figures(self, skip=True):
        processes = [
            WhiteNoise(sigma=1.0, T=1.0),
            GPLinear(sigma=1.0, T=1.0),
            GPConstant(sigma=1, T=1.0),
            GPRBF(length_scale=0.3, sigma=1.0, T=1.0),
        ]

        for g in processes:
            fig = g.plot_paths_and_kernel(n=100, N=5)
            assert isinstance(fig, matplotlib.figure.Figure)

        for g in processes:
            fig = g.plot_kernel()
            assert isinstance(fig, matplotlib.figure.Figure)
