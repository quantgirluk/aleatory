"""
==================
Gaussian Processes
==================

.. currentmodule:: aleatory.gaussian

The :py:mod:`aleatory.gaussian` module contains the implementation of the following Gaussian Processes:

.. autosummary::
   :toctree: gaussian/

    GPConstant       -- Gaussian Process with Constant Kernel
    GPLinear         -- Gaussian Process with Linear Kernel
    GPMatern         -- Gaussian Process with Matern Kernel
    GPPeriodic       -- Gaussian Process with Periodic Kernel
    GPRBF            -- Gaussian Process with Radial Basis Function (RBF) Kernel
    GPSquaredExponential -- Gaussian Process with Squared Exponential Kernel (Same as RBF)
    GPWhiteNoise     -- Gaussian Process with White Noise Kernel
"""

from aleatory.gaussian.gaussian_processes import (
    GPWhiteNoise,
    GPRBF,
    GPSquaredExponential,
    GPLinear,
    GPConstant,
    GPMatern,
    GPPeriodic,
)
