"""
Squared Bessel Process
=========================

Simulate and visualise paths
"""

###############################################################################

# Author: Dialid Santiago <d.santiago@outlook.com>
# License: MIT
# Description: Simulate and visualise a Bessel process

from aleatory.processes import BESQProcess
from aleatory.styles import qp_style

qp_style()  # Use quant-pastel-style

p = BESQProcess(dim=4.0)
fig = p.draw(n=100, N=200, figsize=(12, 7), colormap="viridis", envelope=True)
fig.show()


###############################################################################

fig = p.plot(n=100, N=10, figsize=(12, 7))
fig.show()
