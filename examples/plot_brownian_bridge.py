"""
Brownian Bridge Process
=========================

Simulate and visualise paths
"""

###############################################################################

# Author: Dialid Santiago <d.santiago@outlook.com>
# License: MIT
# Description: Simulate and visualise a Bessel process

from aleatory.processes import BrownianBridge
from aleatory.styles import qp_style

qp_style()  # Use quant-pastel-style

p = BrownianBridge()
fig = p.draw(n=100, N=200, figsize=(10, 7), colormap="PiYG", envelope=True)
fig.show()


###############################################################################

fig = p.plot(n=100, N=10, figsize=(12, 7))
fig.show()
