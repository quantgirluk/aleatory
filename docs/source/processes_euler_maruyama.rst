EM Processes
-------------

The :py:mod:`replica.processes` module provides classes for continuous-time stochastic processes which
cannot be expressed in analytical form. Thus, they are simulated by the Euler-Maruyama method.

* :py:class:`replica.processes.OUProcess`
* :py:class:`replica.processes.CIRProcess`
* :py:class:`replica.processes.CEVProcess`


.. autoclass:: replica.processes.OUProcess
    :members: T, sample, simulate, plot

.. autoclass:: replica.processes.CIRProcess
    :members: T, sample, simulate, plot

.. autoclass:: replica.processes.CEVProcess
    :members: T, sample, simulate, plot