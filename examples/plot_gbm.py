"""
Geometric Brownian Motion
=========================

Simulate and visualise paths
"""

###############################################################################

# Author: Dialid Santiago <d.santiago@outlook.com>
# License: MIT
# Description: Simulate and visualise a Geometric Brownian Motion

from aleatory.processes import GBM

process = GBM()
fig = process.draw(n=100, N=200, figsize=(12, 7), dpi=150)
fig.show()


###############################################################################


process = GBM()
fig = process.plot(n=100, N=10, figsize=(12, 7), dpi=250)
fig.show()
