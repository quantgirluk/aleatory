"""
.. _statsref:

==============================================
Statistical functions (:mod:`aleatory.stats`)
==============================================

.. currentmodule:: aleatory.stats

This module contains the implementation of the following probability distributions:
    -  Non-Central Chi
    - Variance-Gamma

.. autosummary::
   :toctree: non_central_chi/

   ncx      -- Non-central chi
   vg       -- Variance-Gamma

"""

from aleatory.stats.non_central_chi import ncx, ncx_gen
from aleatory.stats.variance_gamma import vg
