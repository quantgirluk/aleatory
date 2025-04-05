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
from aleatory.styles import qp_style

qp_style()  # Use quant-pastel-style

process = BM2D()
fig = process.plot_sample(n=50, figsize=(12, 7), dpi=150)
fig.show()


###############################################################################


process = BM2D()
fig = process.plot_sample(n=200, coordinates=True, figsize=(12, 7))
fig.show()
