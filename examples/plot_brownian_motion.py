"""
Brownian Motion
===============

Simulate and visualise paths
"""

###############################################################################

# Author: Dialid Santiago <d.santiago@outlook.com>
# License: MIT
# Description: Simulate and visualise a Brownian Motion

from aleatory.processes import BrownianMotion
from aleatory.styles import qp_style

qp_style()  # Use quant-pastel-style

process = BrownianMotion()
fig = process.draw(n=100, N=200, figsize=(12, 7), dpi=150)
fig.show()


###############################################################################


process = BrownianMotion()
fig = process.plot(n=100, N=10, figsize=(12, 7), dpi=250)
fig.show()
