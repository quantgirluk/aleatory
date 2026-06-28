"""

Gaussian Rational Quadratic Process
===================================

Simulate and visualise paths
"""

###############################################################################

# Author: Dialid Santiago <d.santiago@outlook.com>
# License: MIT
# Description: Simulate and visualise Rational Quadratic Gaussian Process

from aleatory.processes import GPRationalQuadratic
from aleatory.processes.gaussian.gaussian_periodic import GPPeriodic
from aleatory.styles import qp_style

qp_style()  # Use quant-pastel-style

process = GPRationalQuadratic(length_scale=0.5, sigma=1.0, T=1.0)
fig = process.plot_paths_and_kernel(n=100, N=5, matrix_shape=True)
fig.show()


process = GPRationalQuadratic(length_scale=0.5, sigma=1.0, T=3.0)
fig = process.plot_paths_and_kernel(n=300, N=5, matrix_shape=False)
fig.show()


###############################################################################


process = GPRationalQuadratic(length_scale=0.5, sigma=1.0, T=4.0)
fig = process.draw(n=400, N=100, figsize=(12, 7), dpi=150)
fig.show()


###############################################################################


process = GPRationalQuadratic(length_scale=1.0, sigma=1.0, T=4.0)
fig = process.plot(n=200, N=10, figsize=(12, 7), dpi=250)
fig.show()
