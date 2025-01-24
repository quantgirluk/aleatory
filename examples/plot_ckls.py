"""
CKLS Process
============

Simulate and visualise paths
"""

###############################################################################

# Author: Dialid Santiago <d.santiago@outlook.com>
# License: MIT
# Description: Simulate and visualise a CKLS process

from aleatory.processes import CKLSProcess
from aleatory.styles import qp_style

qp_style()  # Use quant-pastel-style

p = CKLSProcess()
fig = p.draw(n=200, N=200, figsize=(12, 7), colormap="viridis")
fig.show()

###############################################################################

p = CKLSProcess(alpha=1.0, beta=0.5, gamma=1.0, sigma=0.2)
fig = p.draw(n=200, N=200, figsize=(12, 7), colormap="PuBuGn")
fig.show()

###############################################################################

fig = p.plot(n=100, N=10, figsize=(12, 7))
fig.show()
