General Usage
=============


Stochastic Processes
--------------------

This package introduces objects representing a number of continuous-time
stochastic processes :math:`X = \{X_t : t\geq 0\}`, and provides
functionality to:

- generate realizations/trajectories of each process over discrete time sets
- create visualisations to illustrate the processes properties and behaviour

All currently supported processes can be divided in two categories according to the
method that is used to generate their paths:

- Analytical
    - Brownian Motion
    - Geometric Brownian Motion
- Euler-Maruyama
    - Ornstein–Uhlenbeck
    - Vasicek
    - Cox–Ingersoll–Ross
    - Constant Elasticity

Quick-start guide
-----------------

To start using ``replica``, import the stochastic process you want and create an
instance with the required parameters.

.. note::
    All processes instances  will be defined on an finite interval :math:`[0,T]`. Hence, the end point
    :math:`T` is a required argument to create an instance of a process. In all cases :math:`T=1` by default.

.. code-block:: python

    from replica.processes import BrownianMotion

    brownian = BrownianMotion()


The simulate() method
~~~~~~~~~~~~~~~~~~~~~
Every process class has a ``simulate`` method to  generate a number of trajectories/paths.
The ``simulate`` methods require two parameters:

- ``n`` for the number of steps in each path
- ``N`` for the number of paths

and will return a list with ``N`` paths generated from the specified process.

.. code-block:: python

    from replica.processes import BrownianMotion

    brownian = BrownianMotion()
    paths = brownian.simulate(n=100, N=10)



.. note::
    Each path contains
    ``n`` points/steps corresponding to the values of the process at evenly spaced times over the
    interval :math:`[0,T],` i.e.,

    .. math::
        \left\{X \left(\frac{i T }{n-1}\right), i=0,\cdots, n-1\right\}.



The plot() method
~~~~~~~~~~~~~~~~~
Every process class has a ``plot`` method for generating a simple chart
with showing the required simulated trajectories/paths.
Similarly to the ``simulate`` methods, the ``plot`` methods require two parameters:

- ``n`` for the number of steps in each path
- ``N`` for the number of paths

.. code-block:: python

    from replica.processes import BrownianMotion

    brownian = BrownianMotion()
    brownian.plot(n=100, N=10)


.. image:: _static/brownian_motion_quickstart_01.png


Parameters can be accessed as attributes of the instance.

The draw() method
~~~~~~~~~~~~~~~~~
Every process class has a ``draw`` method which generates a more interesting
visualisation of the simulated trajectories/paths.
The ``draw`` methods require two parameters:

- ``n`` for the number of steps in each path
- ``N`` for the number of paths

In addition, there are two optional boolean parameters

- ``marginal`` which is defaulted ``True``
- ``envelope`` which is defaulted ``False``

This allows us to produce four different charts.

.. code-block:: python

    from replica.processes import BrownianMotion

    brownian = BrownianMotion()
    brownian.draw(n=100, N=200)


.. image:: _static/brownian_motion_quickstart_02.png


.. code-block:: python

    from replica.processes import BrownianMotion

    brownian = BrownianMotion()
    brownian.draw(n=100, N=200, envelope=True)


.. image:: _static/brownian_motion_quickstart_03.png


.. code-block:: python

    from replica.processes import BrownianMotion

    brownian = BrownianMotion()
    brownian.draw(n=100, N=200, marginal=False)


.. image:: _static/brownian_motion_quickstart_04.png


.. code-block:: python

    from replica.processes import BrownianMotion
    brownian = BrownianMotion()
    brownian.draw(n=100, N=200, marginal=False, envelope=True)


.. image:: _static/brownian_motion_quickstart_05.png

Parameters can be accessed as attributes of the instance.