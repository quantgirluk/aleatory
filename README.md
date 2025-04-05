# *aleatory*

[![PyPI version fury.io](https://badge.fury.io/py/aleatory.svg)](https://pypi.org/project/aleatory/) [![Downloads](https://static.pepy.tech/personalized-badge/aleatory?period=total&units=international_system&left_color=black&right_color=blue&left_text=Downloads)](https://pepy.tech/project/aleatory)
![example workflow](https://github.com/quantgirluk/aleatory/actions/workflows/python-package.yml/badge.svg) [![Documentation Status](https://readthedocs.org/projects/aleatory/badge/?version=latest)](https://aleatory.readthedocs.io/en/latest/?badge=latest)

- [Git Homepage](https://github.com/quantgirluk/aleatory)
- [Pip Repository](https://pypi.org/project/aleatory/)
- [Documentation](https://aleatory.readthedocs.io/en/latest/)
- 🆕 [Gallery of Stochastic Processes](https://aleatory.readthedocs.io/en/latest/auto_examples/index.html) 🖼️

## Overview

The **_aleatory_** (/ˈeɪliətəri/) Python library provides functionality for simulating and visualising
stochastic processes. More precisely, it introduces objects representing a number of 
stochastic processes and provides methods to:

- generate realizations/trajectories from each process —over discrete time sets
- create visualisations to illustrate the processes properties and behaviour


<p align="center">
<img src="https://raw.githubusercontent.com/quantgirluk/aleatory/main/docs/source/_static/vasicek_process_drawn.png"   style="display:block;float:none;margin-left:auto;margin-right:auto;width:80%">
</p>

Currently, aleatory supports the following stochastic processes in one dimension:

- Arithmetic Brownian Motion (see [Brownian Motion](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.BrownianMotion.html#aleatory.processes.BrownianMotion))
- [Bessel process](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.BESProcess.html#aleatory.processes.BESProcess)
- [Brownian Bridge](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.BrownianBridge.html#aleatory.processes.BrownianBridge)
- [Brownian Excursion](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.BrownianExcursion.html#aleatory.processes.BrownianExcursion)
- [Brownian Meander](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.BrownianMeander.html#aleatory.processes.BrownianMeander)
- [Brownian Motion](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.BrownianMotion.html#aleatory.processes.BrownianMotion)
- [Constant Elasticity Variance (CEV) process](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.CEVProcess.html#aleatory.processes.CEVProcess)
- [Cox–Ingersoll–Ross (CIR) process](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.CIRProcess.html#aleatory.processes.CIRProcess)
- [Chan-Karolyi-Longstaff-Sanders (CKLS) process](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.CKLSProcess.html#aleatory.processes.CKLSProcess)
- [Fractional Brownian Motion process](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.fBM.html#aleatory.processes.fBM)
- [Galton-Watson process with Poisson branching](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.GaltonWatson.html#aleatory.processes.GaltonWatson)
- [Gamma process](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.GammaProcess.html#aleatory.processes.GammaProcess)
- [General Random Walk](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.GeneralRandomWalk.html#aleatory.processes.GeneralRandomWalk)
- [Geometric Brownian Motion](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.GBM.html#aleatory.processes.GBM)
- [Hawkes process](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.HawkesProcess.html#aleatory.processes.HawkesProcess)
- [Inverse Gaussian process](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.InverseGaussian.html#aleatory.processes.InverseGaussian)
- [Inhomogeneous Poisson process](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.InhomogeneousPoissonProcess.html#aleatory.processes.InhomogeneousPoissonProcess)
- [Mixed Poisson process](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.MixedPoissonProcess.html#aleatory.processes.MixedPoissonProcess)
- [Ornstein–Uhlenbeck (OU) process](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.OUProcess.html#aleatory.processes.OUProcess)
- [Poisson process](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.PoissonProcess.html#aleatory.processes.PoissonProcess)
- [Random Walk](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.RandomWalk.html#aleatory.processes.RandomWalk)
- [Squared Bessel processes](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.BESQProcess.html#aleatory.processes.BESQProcess)
- [Vasicek process](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.Vasicek.html#aleatory.processes.Vasicek)
- [Variance-Gamma process](https://aleatory.readthedocs.io/en/latest/processes/aleatory.processes.VarianceGammaProcess.html#aleatory.processes.VarianceGammaProcess)

From v1.1.1 aleatory supports the following 2-d stochastic processes:

- [Brownian Motion 2D](https://aleatory.readthedocs.io/en/latest/auto_examples/plot_brownian_2d.html#sphx-glr-auto-examples-plot-brownian-2d-py)
- [Correlated Brownian Motions](https://aleatory.readthedocs.io/en/latest/auto_examples/plot_correlated_bms.html)
- [Random Walk 2D](https://aleatory.readthedocs.io/en/latest/auto_examples/plot_random_walk_2d.html#sphx-glr-auto-examples-plot-random-walk-2d-py)

## Installation

Aleatory is available on [pypi](https://pypi.python.org/pypi) and can be
installed as follows

```
pip install aleatory
```

## Dependencies

Aleatory relies heavily on

- ``numpy``  for random number generation
- ``scipy`` and ``statsmodels`` for support for a number of one-dimensional distributions.
- ``matplotlib`` for creating visualisations

## Compatibility

Aleatory is tested on Python versions 3.8, 3.9, 3.10, and 3.11

## Quick-Start

Aleatory allows you to create fancy visualisations from different stochastic processes in an easy and concise way.

For example, the following code

```python
from aleatory.processes import BrownianMotion

brownian = BrownianMotion()
brownian.draw(n=100, N=100, colormap="cool", figsize=(12,9))

```

generates a chart like this:

<p align="center">
<img src="https://raw.githubusercontent.com/quantgirluk/aleatory/main/docs/source/_static/brownian_motion_quickstart_08.png"   style="display:block;float:none;margin-left:auto;margin-right:auto;width:80%">
</p>

For more examples visit the [Quick-Start Guide](https://aleatory.readthedocs.io/en/latest/general.html).


**If you like this project, please give it a star!** ⭐️

## Thanks for Visiting! ✨

Connect with me via:

- 🦜 [Twitter](https://twitter.com/Quant_Girl)
- 👩🏽‍💼 [Linkedin](https://www.linkedin.com/in/dialidsantiago/)
- 📸 [Instagram](https://www.instagram.com/quant_girl/)
- 👾 [Personal Website](https://quantgirl.blog)


