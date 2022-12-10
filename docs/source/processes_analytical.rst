Analytical Processes
--------------------


The :py:mod:`replica.processes` module provides classes for continuous-time stochastic processes which can be expressed in
analytical form.


* :py:class:`replica.processes.BrownianMotion`
* :py:class:`replica.processes.GBM`


.. autoclass:: replica.processes.BrownianMotion
    :members: T, sample, sample_at, simulate, plot, draw

.. autoclass:: replica.processes.GBM
    :members: T, sample, sample_at, simulate, plot, draw

