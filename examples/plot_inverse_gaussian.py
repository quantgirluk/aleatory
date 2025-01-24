"""
Inverse Gaussian Process
========================

Simulate and visualise paths
"""

###############################################################################

# Author: Dialid Santiago <d.santiago@outlook.com>
# License: MIT
# Description: Simulate and visualise a Inverse Gaussian Process

from aleatory.processes import InverseGaussian
from aleatory.styles import qp_style

qp_style()  # Use quant-pastel-style

p = InverseGaussian()
fig = p.draw(n=200, N=200, figsize=(12, 7), colormap="Spectral")
fig.show()

###############################################################################

p = InverseGaussian(scale=0.5, T=3.0)
fig = p.draw(n=200, N=200, figsize=(12, 7), colormap="Blues")
fig.show()

###############################################################################

fig = p.plot(n=200, N=10, figsize=(12, 7))
fig.show()
