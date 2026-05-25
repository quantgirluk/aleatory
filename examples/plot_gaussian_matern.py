"""

Gaussian Process Matern
=======================

Simulate and visualise paths
"""

###############################################################################

# Author: Dialid Santiago <d.santiago@outlook.com>
# License: MIT
# Description: Simulate and visualise Matern Gaussian Process

from aleatory.processes import GPMatern
from aleatory.styles import qp_style

qp_style()  # Use quant-pastel-style

process = GPMatern(length_scale=0.1, sigma=1.0, T=1.0)
fig = process.plot_paths_and_kernel(n=100, N=5, matrix_shape=True)
fig.show()


###############################################################################


process = GPMatern(length_scale=0.1, sigma=1.0, T=1.0)
fig = process.draw(n=200, N=200, figsize=(12, 7), dpi=150)
fig.show()


###############################################################################


process = GPMatern(length_scale=0.1, sigma=1.0, T=1.0)
fig = process.plot(n=200, N=10, figsize=(12, 7), dpi=250)
fig.show()
