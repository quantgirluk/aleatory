"""

White Noise
===========

Simulate and visualise paths
"""

###############################################################################

# Author: Dialid Santiago <d.santiago@outlook.com>
# License: MIT
# Description: Simulate and visualise White Noise

from aleatory.processes import WhiteNoise

process = WhiteNoise(sigma=1.0, T=1.0)
fig = process.plot_paths_and_kernel(n=100, N=5, matrix_shape=True)
fig.show()


###############################################################################


process = WhiteNoise(sigma=1.0, T=1.0)
fig = process.draw(n=200, N=200, figsize=(12, 7), dpi=150)
fig.show()


###############################################################################


process = WhiteNoise(sigma=1.0, T=1.0)
fig = process.plot(n=200, N=10, figsize=(12, 7), dpi=250)
fig.show()
