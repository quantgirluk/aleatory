import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from statsmodels import api as sm


def plot_mean_variance(times, means, variances, name, process_label="X", 
                       style="seaborn-v0_8-whitegrid", **fig_kw):

    if process_label:
        labels = (f'$E[{process_label}_t]$', f'$Var[{process_label}_t]$')
    with plt.style.context(style):
        fig, (ax1, ax2,) = plt.subplots(1, 2, figsize=(9, 4), **fig_kw)
        ax1.plot(times, means, lw=1.5, color='black', label=labels[0])
        ax1.set_xlabel('t')
        ax1.legend()
        ax2.plot(times,  variances, lw=1.5, color='red', label=labels[1])
        ax2.set_xlabel('t')
        ax2.legend()
        fig.suptitle(
            f'Expectation and Variance of {name}')
        plt.show()
        
    return fig