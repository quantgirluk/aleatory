"""
Hawkes Process
==============

Simulate and visualise paths
"""

###############################################################################

# Author: Dialid Santiago <d.santiago@outlook.com>
# License: MIT
# Description: Simulate and visualise a Hawkes Process

from aleatory.processes import HawkesProcess
from aleatory.styles import qp_style

qp_style()  # Use quant-pastel-style

p = HawkesProcess()
fig = p.draw(N=200, T=10.0, colormap="Purples", figsize=(12, 7))
fig.show()

###############################################################################

p = HawkesProcess(mu=0.5, alpha=1.0, beta=2.0)
fig = p.draw(N=200, T=10.0, colormap="Purples", mode="steps+points", figsize=(12, 7))
fig.show()

###############################################################################

fig = p.plot(N=10, T=10.0, figsize=(10, 6))
fig.show()
