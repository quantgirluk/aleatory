aleatory
========

**Aleatory** (/ˈeɪliətəri/) is a Python library for simulating and visualising stochastic processes
defined by Stochastic Differential Equations (SDEs). It introduces objects representing continuous-time
stochastic processes and provides
functionality to:

- generate realizations/trajectories of each process over discrete time sets
- create visualisations to illustrate the processes properties and behaviour

.. image:: https://raw.githubusercontent.com/quantgirluk/aleatory/main/docs/source/_static/vasicek_process_drawn.png


Currently, `aleatory` supports the following processes:

- Brownian Motion
- Geometric Brownian Motion
- Ornstein–Uhlenbeck
- Vasicek
- Cox–Ingersoll–Ross
- Constant Elasticity


Installation
------------

Stochastic is available on `pypi <https://pypi.python.org/pypi>`_ and can be
installed using ``pip``:

.. code-block:: bash

   pip install aleatory

Dependencies
------------

Aleatory relies heavily on

- ``numpy``  ``scipy`` for random number generation, as well as support for a number of one-dimensional distributions.

- ``matplotlib`` for creating visualisations

Compatibility
-------------

Aleatory is tested on Python versions 3.7, 3.8, and 3.9
