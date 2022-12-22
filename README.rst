replica
=======

**Replica** (/ˈrep.lɪ.kə/) is a Python library for simulating and visualising stochastic processes
defined by Stochastic Differential Equations (SDEs). It introduces objects representing continuous-time
stochastic processes :math:`X = \{X_t : t\geq 0\}`, and provides
functionality to:

- generate realizations/trajectories of each process over discrete time sets
- create visualisations to illustrate the processes properties and behaviour

.. image:: _static/vasicek_process_drawn.png


Currently, `replica` supports the following processes:

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

   pip install replica

Dependencies
------------

Replica relies heavily on

- ``numpy``  ``scipy`` for random number generation, as well as support for a number of one-dimensional distributions.

- ``matplotlib`` for creating visualisations

Compatibility
-------------

Stochastic is tested on Python versions 3.7, and 3.8.
