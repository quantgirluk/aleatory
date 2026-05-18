from matplotlib import pyplot as plt


def plot_covariance_matrix(
    times,
    covariance_matrix,
    matrix_shape=True,
    style="seaborn-v0_8-whitegrid",
    colormap="coolwarm",
    title=None,
    labels={},
    **fig_kw,
):

    title = title if title else f"Covariance Matrix"
    cbar_label = labels.get("cbar", "Covariance")
    xlabel = labels.get("xlabel", "t")
    ylabel = labels.get("ylabel", "s")

    if matrix_shape:
        origin = "upper"
        my_extent = [times[0], times[-1], times[-1], times[0]]
    else:
        origin = "lower"
        my_extent = [times[0], times[-1], times[0], times[-1]]

    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=(6, 5), **fig_kw)
        im = ax.imshow(
            covariance_matrix,
            cmap=colormap,
            interpolation="none",
            origin=origin,
            extent=my_extent,
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.show()

        return fig
