"""
Fractional Brownian Motion
===========================

Simulate and visualise paths
"""

###############################################################################

# Author: Dialid Santiago <d.santiago@outlook.com>
# License: MIT
# Description: Simulate and visualise a Fractional Brownian Motion

from aleatory.processes import fBM
from aleatory.styles import qp_style

qp_style()  # Use quant-pastel-style

p = fBM()
fig = p.draw(n=200, N=200, figsize=(12, 7), colormap="viridis")
fig.show()

###############################################################################

p = fBM(hurst=0.25, T=2.0)
fig = p.draw(n=200, N=200, figsize=(12, 7), colormap="seismic")
fig.show()

###############################################################################

fig = p.plot(n=100, N=10, figsize=(12, 7))
fig.show()
