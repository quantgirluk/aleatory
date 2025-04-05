"""

.. currentmodule:: aleatory.processes

The :py:mod:`aleatory.processes` module provides classes for following stochastic processes

.. autosummary::
   :toctree: processes

   BrownianMotion               --  Brownian Motion
   GBM                          --  Geometric Brownian Motion
   BrownianBridge               --  Brownian Bridge
   BrownianExcursion            --  Brownian Excursion
   BrownianMeander              --  Brownian Meander
   BESProcess                   -- Bessel Process
   BESQProcess                  -- Squared Bessel Process
   OUProcess                    -- Ornstein-Uhlenbeck Process
   Vasicek                      --  Vasicek Process
   CEVProcess                   --  Constant Elasticity Variance Process
   CIRProcess                   -- Cox–Ingersoll–Ross (CIR) Process
   CKLSProcess                  -- Chan-Karolyi-Longstaff-Sanders (CKLS) process
   fBM                          -- fractional Brownian Motion
   GaltonWatson                 -- Galton-Watson Process
   GammaProcess                 -- Gamma Process
   GeneralRandomWalk            -- General Random Walk Process
   HawkesProcess                -- Hawkes Process
   InverseGaussian              -- Inverse Gaussian Process
   PoissonProcess               -- Poisson Process
   MixedPoissonProcess          -- Mixed Poisson Process
   InhomogeneousPoissonProcess -- Inhomogeneous Poisson Process
   RandomWalk                   -- Random Walk Process
   VarianceGammaProcess         -- Variance Gamma Process

"""

from aleatory.processes.analytical.brownian_motion import BrownianMotion
from aleatory.processes.analytical.geometric_brownian import GBM

from aleatory.processes.analytical.brownian_bridge import BrownianBridge
from aleatory.processes.analytical.brownian_excursion import BrownianExcursion
from aleatory.processes.analytical.brownian_meander import BrownianMeander


from aleatory.processes.analytical.bes import BESProcess
from aleatory.processes.analytical.besq import BESQProcess

from aleatory.processes.euler_maruyama.ornstein_uhlenbeck import OUProcess
from aleatory.processes.euler_maruyama.vasicek import Vasicek
from aleatory.processes.euler_maruyama.cev_process import CEVProcess
from aleatory.processes.euler_maruyama.cir_process import CIRProcess
from aleatory.processes.euler_maruyama.ckls_process import CKLSProcess


from aleatory.processes.fractional.fbm import fBM

from aleatory.processes.jump.galton_watson import GaltonWatson
from aleatory.processes.jump.gamma import GammaProcess
from aleatory.processes.jump.gen_random_walk import GeneralRandomWalk
from aleatory.processes.jump.hawkes import HawkesProcess
from aleatory.processes.jump.inverse_gaussian import InverseGaussian
from aleatory.processes.jump.poisson import PoissonProcess
from aleatory.processes.jump.poisson_mixed import MixedPoissonProcess
from aleatory.processes.jump.poisson_nonhomogeneous import InhomogeneousPoissonProcess
from aleatory.processes.jump.random_walk import RandomWalk, SimpleRandomWalk
from aleatory.processes.jump.variance_gamma import VarianceGammaProcess

from aleatory.processes.multi_dimensional.two_correlated_brownian_motions import (
    CorrelatedBMs,
)
from aleatory.processes.multi_dimensional.brownian_motion_2d import BM2D
from aleatory.processes.multi_dimensional.random_walk_2d import RandomWalk2D
