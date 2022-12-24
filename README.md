# *aleatory*

- Homepage: https://github.com/quantgirluk/aleatory
- Pip Repository: [aleatory](https://pypi.org/project/aleatory/)

---
## Overview

The **_aleatory_** (/ˈeɪliətəri/) Python library provides functionality for simulating and visualising
stochastic processes. More precisely, it introduces objects representing a number of continuous-time
stochastic processes $X = (X_t : t\geq 0)$ and provides methods to:

- generate realizations/trajectories from each process —over discrete time sets
- create visualisations to illustrate the processes properties and behaviour


<figure>
  <p><img src="https://raw.githubusercontent.com/quantgirluk/aleatory/main/docs/source/_static/vasicek_process_drawn.png"
    width="500" height="230">
</figure>

Currently, `aleatory` supports the following processes:

- Brownian Motion
- Geometric Brownian Motion
- Ornstein–Uhlenbeck
- Vasicek
- Cox–Ingersoll–Ross
- Constant Elasticity


## Installation


Aleatory is available on [pypi](https://pypi.python.org/pypi) and can be
installed as follows


```
pip install aleatory
```

## Dependencies


Aleatory relies heavily on

- ``numpy``  ``scipy`` for random number generation, as well as support for a number of one-dimensional distributions.

- ``matplotlib`` for creating visualisations

## Compatibility


Aleatory is tested on Python versions 3.7, 3.8, and 3.9
