"""
Brownian Excursion
==================

Simulate and visualise paths
"""

###############################################################################

# Author: Dialid Santiago <d.santiago@outlook.com>
# License: MIT
# Description: Simulate and visualise a Brownian Excursion

from aleatory.processes import BrownianExcursion
from aleatory.styles import qp_style

qp_style()  # Use quant-pastel-style

p = BrownianExcursion()
fig = p.draw(n=100, N=100, figsize=(10, 7), colormap="Accent")
fig.show()


###############################################################################

fig = p.plot(n=100, N=10, figsize=(12, 7))
fig.show()
