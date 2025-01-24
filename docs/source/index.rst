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


.. toctree::
   :maxdepth: 1
   :titlesonly:
   :hidden:

   auto_examples/index

**Aleatory** (/ˈeɪliətəri/) is a Python library for simulating and visualising stochastic processes. It introduces
objects representing stochastic processes, and provides
functionality to:

- generate realizations/trajectories of each process over discrete time sets
- create visualisations to illustrate the processes properties and behaviour

.. image:: https://raw.githubusercontent.com/quantgirluk/aleatory/main/docs/source/_static/vasicek_process_drawn.png

Currently, `aleatory` supports the following stochastic processes:

- Arithmetic Brownian Motion (see :py:class:`Brownian Motion<aleatory.processes.BrownianMotion>`)
- :py:class:`Bessel process<aleatory.processes.BESProcess>`
- :py:class:`Brownian Bridge<aleatory.processes.BrownianBridge>`
- :py:class:`Brownian Excursion<aleatory.processes.BrownianExcursion>`
- :py:class:`Brownian Meander<aleatory.processes.BrownianMeander>`
- :py:class:`Brownian Motion<aleatory.processes.BrownianMotion>`
- :py:class:`Constant Elasticity Variance (CEV) process<aleatory.processes.CEVProcess>`
- :py:class:`Cox–Ingersoll–Ross (CIR) process<aleatory.processes.CIRProcess>`
- :py:class:`Chan-Karolyi-Longstaff-Sanders (CKLS) process<aleatory.processes.CKLSProcess>`
- :py:class:`Fractional Brownian Motion process<aleatory.processes.fBM>`
- :py:class:`Galton-Watson process with Poisson branching<aleatory.processes.GaltonWatson>`
- :py:class:`Gamma process<aleatory.processes.GammaProcess>`
- :py:class:`General Random Walk<aleatory.processes.GeneralRandomWalk>`
- :py:class:`Geometric Brownian Motion<aleatory.processes.GBM>`
- :py:class:`Hawkes process<aleatory.processes.HawkesProcess>`
- :py:class:`Inverse Gaussian process<aleatory.processes.InverseGaussian>`
- :py:class:`Inhomogeneous Poisson process<aleatory.processes.InhomogeneousPoissonProcess>`
- :py:class:`Mixed Poisson process<aleatory.processes.MixedPoissonProcess>`
- :py:class:`Ornstein–Uhlenbeck (OU) process<aleatory.processes.OUProcess>`
- :py:class:`Poisson process<aleatory.processes.PoissonProcess>`
- :py:class:`Random Walk<aleatory.processes.RandomWalk>`
- :py:class:`Squared Bessel processes<aleatory.processes.BESQProcess>`
- :py:class:`Vasicek process<aleatory.processes.Vasicek>`
- :py:class:`Variance-Gamma process<aleatory.processes.VarianceGammaProcess>`


Installation
------------

Aleatory is available on `pypi <https://pypi.python.org/pypi>`_ and can be
installed as follows

.. code-block:: bash

   pip install aleatory


Installation from GitHub
------------------------

It is possible to install the latest version of the package by cloning its GitHub repository and doing the manual
installation as follows

.. code:: bash

   git clone https://github.com/quantgirluk/aleatory.git
   pip install ./aleatory


Dependencies
------------

Aleatory relies heavily on

- ``numpy`` and  ``scipy`` for random number generation, as well as support for a number of one-dimensional distributions, and special functions.

- ``matplotlib`` for creating visualisations

Compatibility
-------------

Aleatory is tested on Python versions 3.8, 3.9, 3.10, and 3.11.

Quick-Start Guide
-----------------

.. toctree::
   :maxdepth: 2

   general


Modules
-------------

.. toctree::
   :maxdepth: 1

   processes
   stats


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
