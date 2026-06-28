"""

Gaussian Process Periodic
=========================

Simulate and visualise paths
"""

###############################################################################

# Author: Dialid Santiago <d.santiago@outlook.com>
# License: MIT
# Description: Simulate and visualise Periodic Gaussian Process

from aleatory.processes import GPPeriodic
from aleatory.styles import qp_style

qp_style()  # Use quant-pastel-style

process = GPPeriodic(length_scale=1.0, sigma=1.0, T=3.0)
fig = process.plot_paths_and_kernel(n=300, N=5, matrix_shape=True)
fig.show()


process = GPPeriodic(length_scale=1.0, sigma=1.0, T=5.0)
fig = process.plot_paths_and_kernel(n=300, N=5, matrix_shape=False)
fig.show()

###############################################################################


process = GPPeriodic(length_scale=0.5, sigma=1.0, T=2.0)
fig = process.draw(n=200, N=150, figsize=(12, 7), dpi=150)
fig.show()


###############################################################################


process = GPPeriodic(length_scale=1.0, sigma=1.0, T=5.0)
fig = process.plot(n=500, N=10, figsize=(12, 7), dpi=250)
fig.show()
