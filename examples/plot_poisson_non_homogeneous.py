"""
Non-homogeneous Poisson Process
===============================

Simulate and visualise paths
"""

###############################################################################

# Author: Dialid Santiago <d.santiago@outlook.com>
# License: MIT
# Description: Simulate and visualise a Non-homogeneous Poisson Process

from aleatory.processes import InhomogeneousPoissonProcess
from aleatory.styles import qp_style
import numpy as np

qp_style()  # Use quant-pastel-style


def myfunction(s):
    return 5 + 2.0 * np.sin(2 * np.pi * s)  # Example: periodic intensity


p = InhomogeneousPoissonProcess(intensity=myfunction)
t = f"Inhomogeneous Poisson Process $\\lambda(t)=5 + 2\\sin(2\\pi t)$"
fig = p.draw(N=100, T=5.0, figsize=(12, 7), colormap="RdPu", title=t)
fig.show()


###############################################################################


def myfunction(s):
    return s**2


p = InhomogeneousPoissonProcess(intensity=myfunction)
t = f"Inhomogeneous Poisson Process $\\lambda(t)=t^2$"
fig = p.draw(N=100, T=5.0, figsize=(12, 7), colormap="RdPu", title=t)
fig.show()

###############################################################################

fig = p.plot(N=5, T=5.0, figsize=(12, 7), title=t)
fig.show()
