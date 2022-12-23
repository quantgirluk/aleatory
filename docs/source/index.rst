.. aleatory documentation master file, created by
   sphinx-quickstart on Thu Dec  8 09:52:55 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

aleatory
=======

**Aleatory** (/ˈeɪliətəri/) is a Python library for simulating and visualising stochastic processes
defined by Stochastic Differential Equations (SDEs). It introduces objects representing continuous-time
stochastic processes :math:`X = \{X_t : t\geq 0\}`, and provides
functionality to:

- generate realizations/trajectories of each process over discrete time sets
- create visualisations to illustrate the processes properties and behaviour

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

Aleatory is tested on Python versions 3.7, and 3.8.


Documentation
-------------


.. toctree::
   :maxdepth: 2

   general
   processes_analytical
   processes_euler_maruyama


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
