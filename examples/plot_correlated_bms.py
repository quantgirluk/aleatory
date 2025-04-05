"""
Correlated Brownian Motions
===========================

Simulate and visualise paths
"""

###############################################################################

# Author: Dialid Santiago <d.santiago@outlook.com>
# License: MIT
# Description: Simulate and visualise a two Correlated Brownian Motions

from aleatory.processes import CorrelatedBMs
from aleatory.styles import qp_style

qp_style()  # Use quant-pastel-style

p = CorrelatedBMs(rho=-0.5)

fig = p.plot_sample(n=500, figsize=(12, 8))
fig.show()

fig = p.plot_sample(n=200, coordinates=True, figsize=(12, 10))
fig.show()
