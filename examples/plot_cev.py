"""
CEV Process
===========

Simulate and visualise paths
"""

###############################################################################

# Author: Dialid Santiago <d.santiago@outlook.com>
# License: MIT
# Description: Simulate and visualise a CEV process

from aleatory.processes import CEVProcess
from aleatory.styles import qp_style

qp_style()  # Use quant-pastel-style

p = CEVProcess()
fig = p.draw(n=200, N=200, figsize=(12, 7), colormap="Purples")
fig.show()

###############################################################################

p = CEVProcess(mu=1.0, gamma=1.0, sigma=0.5, initial=2.0, T=1.0)
fig = p.draw(n=200, N=200, figsize=(12, 7), colormap="copper")
fig.show()

###############################################################################

fig = p.plot(n=100, N=10, figsize=(12, 7))
fig.show()
