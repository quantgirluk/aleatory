"""
Brownian Meander
==================

Simulate and visualise paths
"""

###############################################################################

# Author: Dialid Santiago <d.santiago@outlook.com>
# License: MIT
# Description: Simulate and visualise a Brownian Meander

from aleatory.processes import BrownianMeander
from aleatory.styles import qp_style

qp_style()  # Use quant-pastel-style

p = BrownianMeander()
fig = p.draw(n=100, N=100, figsize=(12, 7), colormap="spring")
fig.show()

###############################################################################

p = BrownianMeander(fixed_end=True, end=3.0)
fig = p.draw(n=100, N=100, figsize=(12, 7), colormap="RdPu")
fig.show()

###############################################################################

fig = p.plot(n=100, N=10, figsize=(12, 7))
fig.show()
