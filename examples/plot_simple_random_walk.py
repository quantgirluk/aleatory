"""
Simple Random Walk
=======================

Simulate and visualise paths
"""

###############################################################################

# Author: Dialid Santiago <d.santiago@outlook.com>
# License: MIT
# Description: Simulate and visualise a Simple Random Walk

from aleatory.processes import SimpleRandomWalk, RandomWalk
from aleatory.styles import qp_style

qp_style()  # Use quant-pastel-style

p = RandomWalk()
fig = p.draw(n=100, N=200, figsize=(12, 7), colormap="cool")
fig.show()

###############################################################################

p = SimpleRandomWalk(p=0.25)
fig = p.draw(n=100, N=200, figsize=(12, 7), colormap="summer")
fig.show()

###############################################################################

fig = p.plot(n=10, N=20, figsize=(12, 7))
fig.show()
