"""
Brownian Motion 2D
==================

Simulate and visualise a path from a 2D-Brownian Motion
"""

###############################################################################

# Author: Dialid Santiago <d.santiago@outlook.com>
# License: MIT
# Description: Simulate and visualise a path from a 2D-Brownian Motion

from aleatory.processes import BM2D

process = BM2D()
fig = process.plot_sample(
    n=10000, figsize=(12, 7), dpi=150, title="Have a lovely Easter break!"
)
fig.show()


###############################################################################


# proc"ess = BM2D()
# fig = process.plot_sample(n=200, coordinates=True, figsize=(12, 7))
# fig.show()"
