"""
Variance-Gamma Process
========================

Simulate and visualise paths
"""

###############################################################################

# Author: Dialid Santiago <d.santiago@outlook.com>
# License: MIT
# Description: Simulate and visualise a Variance-Gamma Process

from aleatory.processes import VarianceGammaProcess
from aleatory.styles import qp_style

qp_style()  # Use quant-pastel-style

p = VarianceGammaProcess()
fig = p.draw(n=100, N=200, figsize=(12, 7), colormap="winter")
fig.show()

###############################################################################

p = VarianceGammaProcess(theta=-1.0, nu=4.0, sigma=2.0, T=100.0)
fig = p.draw(n=100, N=200, figsize=(12, 7), colormap="summer")
fig.show()

###############################################################################

fig = p.plot(n=200, N=10, figsize=(12, 7))
fig.show()
