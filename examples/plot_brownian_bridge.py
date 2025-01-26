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
p.draw(n=200, N=100, figsize=(10, 7), colormap="Spectral", envelope=True)

###############################################################################

p.plot(n=100, N=10, figsize=(12, 7))
