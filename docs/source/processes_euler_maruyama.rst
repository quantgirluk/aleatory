EM Processes
-------------

The :py:mod:`replica.processes` module also provides functionality for continuous-time stochastic processes which
cannot be expressed in analytical form. These processes are simulated via the
`Euler-Maruyama <https://en.wikipedia.org/wiki/Euler–Maruyama_method>`_ method.

* :py:class:`replica.processes.OUProcess`
* :py:class:`replica.processes.Vasicek`
* :py:class:`replica.processes.CIRProcess`
* :py:class:`replica.processes.CEVProcess`


.. autoclass:: replica.processes.OUProcess
    :members: T, sample, simulate, plot, draw

.. autoclass:: replica.processes.Vasicek
    :members: T, sample, simulate, plot, draw

.. autoclass:: replica.processes.CIRProcess
    :members: T, sample, simulate, plot, draw

.. autoclass:: replica.processes.CEVProcess
    :members: T, sample, simulate, plot, draw