"""
Arithmetic Brownian Motion
===========================

Simulate and visualise paths
"""

###############################################################################

# Author: Dialid Santiago <d.santiago@outlook.com>
# License: MIT
# Description: Simulate and visualise an Arithmetic Brownian Motion

from aleatory.processes import BrownianMotion
from aleatory.styles import qp_style

qp_style()  # Use quant-pastel-style

p = BrownianMotion(drift=1.0, scale=2.0)
fig = p.draw(n=100, N=200, figsize=(12, 7))
fig.show()


###############################################################################


p = BrownianMotion(drift=-1.0, scale=0.5)
fig = p.draw(n=100, N=200, figsize=(12, 7))
fig.show()

###############################################################################

fig = p.plot(n=100, N=10, figsize=(12, 7))
fig.show()
