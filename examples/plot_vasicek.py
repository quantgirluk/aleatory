"""
Vasicek Process
===============

Simulate and visualise paths
"""

###############################################################################

# Author: Dialid Santiago <d.santiago@outlook.com>
# License: MIT
# Description: Simulate and visualise a Vasicek Process

from aleatory.processes import Vasicek
from aleatory.styles import qp_style

qp_style()  # Use quant-pastel-style

p = Vasicek()
fig = p.draw(n=200, N=200, figsize=(12, 7), colormap="PuOr")
fig.show()

###############################################################################

p = Vasicek(theta=1.0, mu=3.0, sigma=0.5, initial=1.0)
fig = p.draw(n=200, N=200, figsize=(12, 7), colormap="seismic")
fig.show()

###############################################################################

fig = p.plot(n=100, N=10, figsize=(12, 7))
fig.show()
