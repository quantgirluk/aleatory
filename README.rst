Replica
=======

**Replica** (/ˈrep.lɪ.kə/) is a Python library for simulating and visualising stochastic processes
defined by Stochastic Differential Equations (SDEs).


Dependencies
~~~~~~~~~~~~

Replica relies heavily on the following Python libraries

- ``numpy`` and ``scipy`` for array calculations as well as random number generation
- ``matplotlib`` for visualisation

Processes
---------

This library offers a number of stochastic process objects available for generating realizations from their
corresponding SDEs.
In addition, for each stochastic process there is a path object which allows the creation of charts to illustrate the
trajectories of such process.

* Exact processes
    * Brownian Motion
    * Brownian Motion with Drift
    * Geometric Brownian Motion

* Euler-Maruyama processes
    * Ornstein Uhlenbeck
    * Vasicek
    *  Constant Elasticity Variance
    * Cox Ingersoll Ross
