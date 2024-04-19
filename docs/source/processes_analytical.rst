Analytical Processes
--------------------


The :py:mod:`aleatory.processes` module provides classes for continuous-time stochastic processes which can be expressed in
analytical form.


* :py:class:`aleatory.processes.BrownianMotion`
* :py:class:`aleatory.processes.BrownianBridge`
* :py:class:`aleatory.processes.BrownianExcursion`
* :py:class:`aleatory.processes.BrownianMeander`
* :py:class:`aleatory.processes.GBM`
* :py:class:`aleatory.processes.BESProcess`
* :py:class:`aleatory.processes.BESQProcess`


.. autoclass:: aleatory.processes.BrownianMotion
    :members: T, sample, sample_at, simulate, plot, draw

.. autoclass:: aleatory.processes.BrownianBridge
    :members: T, sample, sample_at, simulate, plot, draw

.. autoclass:: aleatory.processes.BrownianExcursion
    :members: T, sample, sample_at, simulate, plot, draw

.. autoclass:: aleatory.processes.BrownianMeander
    :members: T, sample, sample_at, simulate, plot, draw

.. autoclass:: aleatory.processes.GBM
    :members: T, sample, sample_at, simulate, plot, draw

.. autoclass:: aleatory.processes.BESProcess
    :members: T, sample, simulate, plot, draw

.. autoclass:: aleatory.processes.BESQProcess
    :members: T, sample, simulate, plot, draw


