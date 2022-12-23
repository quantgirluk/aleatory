Analytical Processes
--------------------


The :py:mod:`aleatory.processes` module provides classes for continuous-time stochastic processes which can be expressed in
analytical form.


* :py:class:`aleatory.processes.BrownianMotion`
* :py:class:`aleatory.processes.GBM`


.. autoclass:: aleatory.processes.BrownianMotion
    :members: T, sample, sample_at, simulate, plot, draw

.. autoclass:: aleatory.processes.GBM
    :members: T, sample, sample_at, simulate, plot, draw

