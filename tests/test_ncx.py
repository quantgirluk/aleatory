import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from aleatory.stats import ncx

import unittest


class TestNCX(unittest.TestCase):
    df = 2.5
    nc = 2.0
    size = 500
    vis = False

    marginal = ncx(df=2.5, nc=2.0)

    @unittest.skipIf(not vis, "No Visualisation Required")
    def test_pdf(self):
        marginal = self.marginal
        x = np.linspace(0, marginal.ppf(0.9999))
        plt.plot(x, marginal.pdf(x))
        plt.axvline(marginal.ppf(0.05))
        plt.axvline(marginal.ppf(0.95))
        plt.axvline(marginal.mean(), color="red")
        plt.show()

    def test_cdf_ppf(self):
        marginal = self.marginal
        xs = np.linspace(0, marginal.ppf(0.9999))
        vals = ncx.ppf(marginal.cdf(xs), df=self.df, nc=self.nc)

        for (x, v) in zip(xs, vals):
            self.assertAlmostEqual(x, v)
        if self.vis:
            plt.plot(xs, vals)
            plt.plot(xs, xs)
            plt.show()

        qs = np.linspace(0, 1)
        vals = ncx.cdf(marginal.ppf(qs), df=self.df, nc=self.nc)
        for (q, v) in zip(qs, vals):
            self.assertAlmostEqual(q, v)
        if self.vis:
            plt.plot(qs, vals)
            plt.plot(qs, qs)
            plt.show()

    @unittest.skipIf(not vis, "No Visualisation Required")
    def test_sample(self):
        marginal = self.marginal
        sample_size = 5000
        bins = int(sqrt(sample_size))
        sample = marginal.rvs(size=sample_size)

        plt.hist(sample, density=True, bins=bins)
        x = np.linspace(0, 7, 100)
        plt.plot(x, marginal.pdf(x))
        plt.show()
