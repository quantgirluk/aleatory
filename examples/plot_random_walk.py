"""
Generalised Random Walk
=======================

Simulate and visualise paths
"""

###############################################################################

# Author: Dialid Santiago <d.santiago@outlook.com>
# License: MIT
# Description: Simulate and visualise a Generalised Random Walk

from aleatory.processes import GeneralRandomWalk
from aleatory.styles import qp_style
from scipy.stats import norm, expon

qp_style()  # Use quant-pastel-style

p = GeneralRandomWalk(step_dist=expon)
fig = p.draw(
    n=100,
    N=200,
    figsize=(12, 7),
    title="Random Walk with Exponential Steps",
    colormap="viridis",
)
fig.show()

###############################################################################

p = GeneralRandomWalk(step_dist=norm, normalised=True)
fig = p.draw(
    n=100, N=200, figsize=(12, 7), title="Random Walk with Normalised Gaussian Steps"
)
fig.show()

###############################################################################


fig = p.plot(
    n=10, N=20, figsize=(12, 7), title="Random Walk with Normalised Gaussian Steps"
)
fig.show()
