"""
Vasicek Process
===============

Simulate and visualise paths
"""

###############################################################################

# Author: Dialid Santiago <d.santiago@outlook.com>
# License: MIT
# Description: Simulate and visualise a Vasicek Process

from aleatory.processes import Vasicek
from aleatory.styles import qp_style
import numpy as np

qp_style()  # Use quant-pastel-style

p = Vasicek()
fig = p.draw(n=200, N=200, figsize=(12, 7), colormap="PuOr")
fig.show()

# ###############################################################################

p = Vasicek(theta=1.0, mu=3.0, sigma=0.5, initial=1.0)
fig = p.draw(n=200, N=200, figsize=(12, 7), colormap="seismic")
fig.show()

# ###############################################################################

fig = p.plot(n=100, N=10, figsize=(12, 7))
fig.show()

# ###############################################################################


p = Vasicek(theta=0.5, mu=2.0, sigma=0.5, initial=1.0)

times = np.linspace(0, 1, 100)
fig = p.plot_kernel(times=times, title="Covariance Kernel of Vasicek Process")
fig.show()


fig = p.plot_paths_and_kernel(
    n=100, N=5, figsize=(12, 7), cmap="PuOr", title="Vasicek Process Paths and Kernel"
)
fig.show()

times = np.linspace(0, 10, 200)
fig = p.plot_mean_variance(times=times, title="Mean and Variance of Vasicek Process")
fig.show()

# fig = p.plot_kernel3d(times=times, title="3D Covariance Kernel of Vasicek Process")
# fig.show()
