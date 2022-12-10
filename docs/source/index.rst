.. replica documentation master file, created by
   sphinx-quickstart on Thu Dec  8 09:52:55 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

replica
=======

**Replica** (/ˈrep.lɪ.kə/) is a Python library for simulating and visualising stochastic processes
defined by Stochastic Differential Equations (SDEs).

Installation
------------

Stochastic is available on `pypi <https://pypi.python.org/pypi>`_ and can be
installed using ``pip``:

.. code-block:: bash

   pip install replica

Dependencies
------------

Replica depends heavily on

- ``numpy``  ``scipy`` for  most calculations and random number generation
- ``matplotlib`` for creating visualisations

Compatibility
-------------

Stochastic is tested on Python versions 3.7, and 3.8.


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
* :ref:`modindex`
* :ref:`search`
