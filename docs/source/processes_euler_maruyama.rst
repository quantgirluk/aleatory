EM Processes
-------------

The :py:mod:`aleatory.processes` module also provides functionality for continuous-time stochastic processes which
cannot be expressed in analytical form. These processes are simulated via the
`Euler-Maruyama <https://en.wikipedia.org/wiki/Euler–Maruyama_method>`_ method.

* :py:class:`aleatory.processes.OUProcess`
* :py:class:`aleatory.processes.Vasicek`
* :py:class:`aleatory.processes.CIRProcess`
* :py:class:`aleatory.processes.CEVProcess`
* :py:class:`aleatory.processes.CKLSProcess`


.. autoclass:: aleatory.processes.OUProcess
    :members: T, sample, simulate, plot, draw

.. autoclass:: aleatory.processes.Vasicek
    :members: T, sample, simulate, plot, draw

.. autoclass:: aleatory.processes.CIRProcess
    :members: T, sample, simulate, plot, draw

.. autoclass:: aleatory.processes.CEVProcess
    :members: T, sample, simulate, plot, draw

.. autoclass:: aleatory.processes.CKLSProcess
    :members: T, sample, simulate, plot, draw