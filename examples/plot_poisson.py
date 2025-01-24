"""
Poisson Process
===============

Simulate and visualise paths
"""

###############################################################################

# Author: Dialid Santiago <d.santiago@outlook.com>
# License: MIT
# Description: Simulate and visualise a Poisson Process

from aleatory.processes import PoissonProcess
from aleatory.styles import qp_style

qp_style()  # Use quant-pastel-style

p = PoissonProcess()
fig = p.draw(N=200, T=10.0, colormap="pink", figsize=(12, 7))
fig.show()

###############################################################################

p = PoissonProcess(rate=2.0)
fig = p.draw(N=200, T=10.0, colormap="viridis", figsize=(12, 7), envelope=True)
fig.show()
###############################################################################

fig = p.plot(N=10, T=10.0, figsize=(10, 6))
fig.show()
