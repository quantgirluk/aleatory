# *aleatory*

[![PyPI version fury.io](https://badge.fury.io/py/aleatory.svg)](https://pypi.org/project/aleatory/) [![Downloads](https://static.pepy.tech/personalized-badge/aleatory?period=total&units=international_system&left_color=black&right_color=blue&left_text=Downloads)](https://pepy.tech/project/aleatory)
![example workflow](https://github.com/quantgirluk/aleatory/actions/workflows/python-package.yml/badge.svg) [![Documentation Status](https://readthedocs.org/projects/aleatory/badge/?version=latest)](https://aleatory.readthedocs.io/en/latest/?badge=latest)

- [Git Homepage](https://github.com/quantgirluk/aleatory)
- [Pip Repository](https://pypi.org/project/aleatory/)
- [Documentation](https://aleatory.readthedocs.io/en/latest/)

## Overview

The **_aleatory_** (/ˈeɪliətəri/) Python library provides functionality for simulating and visualising
stochastic processes. More precisely, it introduces objects representing a number of 
stochastic processes and provides methods to:

- generate realizations/trajectories from each process —over discrete time sets
- create visualisations to illustrate the processes properties and behaviour


<p align="center">
<img src="https://raw.githubusercontent.com/quantgirluk/aleatory/main/docs/source/_static/vasicek_process_drawn.png"   style="display:block;float:none;margin-left:auto;margin-right:auto;width:80%">
</p>

Currently, `aleatory` supports the following 13 processes:

- Arithmetic Brownian Motion
- Bessel process
- Brownian Bridge
- Brownian Excursion
- Brownian Meander
- Brownian Motion
- Constant Elasticity Variance (CEV) process
- Cox–Ingersoll–Ross (CIR) process
- Chan-Karolyi-Longstaff-Sanders (CKLS) process
- Fractional Brownian Motion
- Galton-Watson with Poisson branching
- Gamma process
- General Random Walk
- Geometric Brownian Motion
- Hawkes process
- Inverse Gaussian process
- Inhomogeneous Poisson process
- Mixed Poisson process
- Ornstein–Uhlenbeck (OU) process
- Poisson process
- Random Walk
- Squared Bessel processes
- Vasicek process
- Variance-Gamma process

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


