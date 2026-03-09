import unittest

from aleatory.processes import GammaProcess


class testGamma(unittest.TestCase):
    def test_T_updates_gamma_increments(self):
        process = GammaProcess(T=1.0)
        self.assertEqual(process.gamma_increments.T, 1.0)

        process.T = 2.5
        self.assertEqual(process.gamma_increments.T, 2.5)

    def test_simulate_T_updates_gamma_increments(self):
        process = GammaProcess(T=1.0)

        process.simulate(n=20, N=2, T=3.0)

        self.assertEqual(process.T, 3.0)
        self.assertEqual(process.gamma_increments.T, 3.0)
