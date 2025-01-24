"""
CIR Process
===========

Simulate and visualise paths
"""

###############################################################################

# Author: Dialid Santiago <d.santiago@outlook.com>
# License: MIT
# Description: Simulate and visualise a CIR process

from aleatory.processes import CIRProcess
from aleatory.styles import qp_style

qp_style()  # Use quant-pastel-style

p = CIRProcess()
fig = p.draw(n=200, N=200, figsize=(12, 7), colormap="terrain")
fig.show()

###############################################################################

p = CIRProcess(theta=1.0, mu=10.0, sigma=2.0, initial=1.0, T=10.0)
fig = p.draw(n=200, N=200, figsize=(12, 7), colormap="Oranges")
fig.show()

###############################################################################

fig = p.plot(n=100, N=10, figsize=(12, 7))
fig.show()
