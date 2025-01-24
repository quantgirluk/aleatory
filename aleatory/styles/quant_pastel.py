# -*- coding: utf-8 -*-

# Author: Dialid Santiago <d.santiago@outlook.com>
# License: MIT
# Description: Apply Quant Pastel Style

###############################################################################

import matplotlib.pyplot as plt
from os.path import join, dirname, realpath

STYLE_DIR = realpath(dirname(__file__))


def qp_style():
    plt.style.use(join(STYLE_DIR, "quant-pastel-light.mplstyle"))
