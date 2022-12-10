General Usage
=============


Stochastic Processes
--------------------

This package introduces objects representing a number of continuous-time
stochastic processes of the form :math:`X = \{X_t : t\geq 0\}`, and provides
functionality to:

- generate realizations/trajectories of each process over discrete time sets
- produce charts to illustrate the processes properties and behaviour

Currently supported processes can be divided in two categories according to the
method that is used to generate the trajectories/paths:

- Analytical
    - Brownian Motion
    - Geometric Brownian Motion
- Euler-Maruyama
    - Ornstein–Uhlenbeck
    - Vasicek
    - Cox–Ingersoll–Ross
    - Constant Elasticity

Usage Patterns
--------------

To use ``replica``, import the stochastic process you want and create an
instance with the required parameters.


.. code-block:: python

    from replica.processes import BrownianMotion

    brownian = BrownianMotion()


The simulate() method
~~~~~~~~~~~~~~~~~~~~~
Every process class has a ``simulate`` method for generating trajectories/paths.
The ``simulate`` methods require two parameters:

- ``n`` for the number of steps in each path
- ``N`` for the number of paths

.. code-block:: python

    from replica.processes import BrownianMotion

    brownian = BrownianMotion()
    paths = brownian.simulate(n=100, N=10)

Parameters can be accessed as attributes of the instance.

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
- ``envelope`` which is defaulted ``True``

This allows us to produce four different charts.

.. code-block:: python

    from replica.processes import BrownianMotion

    brownian = BrownianMotion()
    brownian.draw(n=100, N=200)


.. code-block:: python

    from replica.processes import BrownianMotion

    brownian = BrownianMotion()
    brownian.draw(n=100, N=200, envelope=False)

.. code-block:: python

    from replica.processes import BrownianMotion
    brownian = BrownianMotion()
    brownian.draw(n=100, N=200, marginal=False)



.. code-block:: python

    from replica.processes import BrownianMotion
    brownian = BrownianMotion()
    brownian.draw(n=100, N=200, marginal=False, envelope=False)

Parameters can be accessed as attributes of the instance.