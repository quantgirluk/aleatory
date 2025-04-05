"""
Simple Random Walk 2D
=====================

Simulate and visualise a path from a 2D-Random Walk
"""

###############################################################################

# Author: Dialid Santiago <d.santiago@outlook.com>
# License: MIT
# Description: Simulate and visualise a path from a 2D-Random Walk

from aleatory.processes import RandomWalk2D
from aleatory.styles import qp_style

qp_style()  # Use quant-pastel-style

process = RandomWalk2D()
fig = process.plot_sample(n=300, figsize=(12, 7), dpi=150)
fig.show()


###############################################################################


process = RandomWalk2D()
fig = process.plot_sample(n=300, coordinates=True, figsize=(12, 7))
fig.show()
