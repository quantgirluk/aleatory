"""
Galton-Watson Process
======================

Simulate and visualise paths
"""

###############################################################################

# Author: Dialid Santiago <d.santiago@outlook.com>
# License: MIT
# Description: Simulate and visualise a Galton-Watson Process

from aleatory.processes import GaltonWatson
from aleatory.styles import qp_style

qp_style()  # Use quant-pastel-style

p = GaltonWatson(mu=1.5)
fig = p.draw(N=100, n=10, figsize=(12, 7), colormap="summer")
fig.show()

###############################################################################

fig = p.plot(N=10, n=10, figsize=(12, 7))
fig.show()

###############################################################################

fig = p.plot(N=50, n=10, figsize=(12, 7), color_survival=True)
fig.show()
