from matplotlib import pyplot as plt
import numpy as np


def plot_covariance_matrix(
    times,
    covariance_matrix,
    matrix_shape=True,
    style="seaborn-v0_8-whitegrid",
    colormap="coolwarm",
    title=None,
    cbar_labels={},
    **fig_kw,
):

    title = title if title else f"Covariance Matrix"
    cbar_label = cbar_labels.get("cbar", "Covariance")
    xlabel = cbar_labels.get("xlabel", "t")
    ylabel = cbar_labels.get("ylabel", "s")

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


def plot_paths_and_kernel(
    paths,
    times,
    covariance_matrix,
    title=None,
    cmap="coolwarm",
    matrix_shape=False,
    style="seaborn-v0_8-whitegrid",
    **fig_kw,
):
    if matrix_shape:
        origin = "upper"
        extent = [times[0], times[-1], times[-1], times[0]]
    else:
        origin = "lower"
        extent = [times[0], times[-1], times[0], times[-1]]
    K = covariance_matrix

    with plt.style.context(style):
        fig, axes = plt.subplots(1, 2, **fig_kw)
        for i in range(len(paths)):
            axes[0].plot(times, paths[i])
        axes[0].set_title("Simulated Paths")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Value")

    if matrix_shape:
        origin = "upper"
        extent = [times[0], times[-1], times[-1], times[0]]
    else:
        origin = "lower"
        extent = [times[0], times[-1], times[0], times[-1]]

    axes[1].imshow(K, cmap=cmap, interpolation="none", origin=origin, extent=extent)
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=axes[1])
    cbar.set_label("K(t, s)")
    axes[1].set_title("Kernel")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("s")

    if title:
        fig.suptitle(title)
    else:
        fig.suptitle("Simulated Paths and Kernel")
    plt.tight_layout()
    plt.show()
    return fig


def plot_kernel3d(times, K, title=None, style="seaborn-v0_8-whitegrid", **fig_kw):

    cmap = fig_kw.pop("cmap", "coolwarm")  # Default colormap for 3D surface

    with plt.style.context(style):
        fig = plt.figure(figsize=(8, 6), **fig_kw)
        ax = fig.add_subplot(111, projection="3d")
        T1, T2 = np.meshgrid(times, times)
        ax.plot_surface(T1, T2, K, cmap=cmap)
        ax.set_title(title if title else f"Kernel Function")
        ax.set_xlabel("t")
        ax.set_ylabel("s")
        ax.set_zlabel("K(t, s)")
        plt.show()
    return fig
