.. aleatory documentation master file, created by
   sphinx-quickstart on Thu Dec  8 09:52:55 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

aleatory
========

|build| |rtd| |pypi| |pyversions|

.. |build| image:: https://github.com/quantgirluk/aleatory/actions/workflows/python-package.yml/badge.svg
    :target: https://github.com/quantgirluk/aleatory/actions

.. |rtd| image:: https://img.shields.io/readthedocs/aleatory.svg
    :target: http://aleatory.readthedocs.io/en/latest/

.. |pypi| image:: https://img.shields.io/pypi/v/aleatory.svg
    :target: https://pypi.python.org/pypi/aleatory

.. |pyversions| image:: https://img.shields.io/pypi/pyversions/aleatory.svg
    :target: https://pypi.python.org/pypi/aleatory


**Aleatory** (/ˈeɪliətəri/) is a Python library for simulating and visualising stochastic processes. It introduces objects representing continuous-time
stochastic processes :math:`X = \{X_t : t\geq 0\}`, and provides
functionality to:

- generate realizations/trajectories of each process over discrete time sets
- create visualisations to illustrate the processes properties and behaviour

.. image:: https://raw.githubusercontent.com/quantgirluk/aleatory/main/docs/source/_static/vasicek_process_drawn.png


Currently, `aleatory` supports the following processes:

1. Brownian Motion
2. Brownian Bridge
3. Brownian Excursion
4. Brownian Meander
5. Geometric Brownian Motion
6. Ornstein–Uhlenbeck (OU) process
7. Vasicek process
8. Cox–Ingersoll–Ross (CIR) process
9. Constant Elasticity Variance (CEV) process
10. Chan-Karolyi-Longstaff-Sanders (CKLS) process
11. Bessel processes
12. Squared Bessel processes
13. Poisson process



Installation
------------

Aleatory is available on `pypi <https://pypi.python.org/pypi>`_ and can be
installed as follows

.. code-block:: bash

   pip install aleatory

Dependencies
------------

Aleatory relies heavily on

- ``numpy`` and  ``scipy`` for random number generation, as well as support for a number of one-dimensional distributions, and special functions.

- ``matplotlib`` for creating visualisations

Compatibility
-------------

Aleatory is tested on Python versions 3.8, 3.9, 3.10, and 3.11.


Documentation
-------------

.. toctree::
   :maxdepth: 2

   general
   processes_analytical
   processes_euler_maruyama
   stats


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
