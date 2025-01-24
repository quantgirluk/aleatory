"""
Gamma Process
=============

Simulate and visualise paths
"""

###############################################################################

# Author: Dialid Santiago <d.santiago@outlook.com>
# License: MIT
# Description: Simulate and visualise a Gamma Process

from aleatory.processes import GammaProcess
from aleatory.styles import qp_style

qp_style()  # Use quant-pastel-style

p = GammaProcess()
fig = p.draw(n=200, N=200, figsize=(12, 7), colormap="winter")
fig.show()

###############################################################################

p = GammaProcess(mu=2.0, nu=1.5, T=20.0)
fig = p.draw(n=200, N=200, figsize=(12, 7), colormap="YlGn")
fig.show()

###############################################################################

p = GammaProcess(mu=1.5, nu=0.5, T=10)
fig = p.plot(n=100, N=10, figsize=(12, 7))
fig.show()
