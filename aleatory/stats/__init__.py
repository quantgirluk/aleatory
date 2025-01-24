"""
=========================
Probability Distributions
=========================

.. currentmodule:: aleatory.stats


The :py:mod:`aleatory.stats` module contains the implementation of the following probability distributions:

.. autosummary::
   :toctree: stats/

   ncx      -- Non-central chi
   vg       -- Variance-Gamma

"""

from aleatory.stats.non_central_chi import ncx, ncx_gen
from aleatory.stats.variance_gamma import vg
