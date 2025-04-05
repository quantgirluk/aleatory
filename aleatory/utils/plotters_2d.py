from matplotlib import pyplot as plt


def plot_paths_coordinates(
    *args,
    times,
    paths1,
    paths2,
    style="seaborn-v0_8-whitegrid",
    title=None,
    mode="steps",
    **fig_kw,
):
    with plt.style.context(style):
        fig, ax = plt.subplots(**fig_kw)
        for p1, p2 in zip(paths1, paths2):
            if mode == "points":
                ax.scatter(times, p1, s=7)
                ax.scatter(times, p2, s=7)
            elif mode == "steps":
                ax.step(times, p1, where="post", label="$X_1$")
                ax.step(times, p2, where="post", label="$X_2$")
            elif mode == "linear":
                ax.plot(times, p1, label="$X_1$", *args)
                ax.plot(times, p2, label="$X_2$", *args)
            elif mode in ["points+steps", "steps+points"]:
                ax.step(times, p1, where="post")
                ax.step(times, p2, where="post")
                color = plt.gca().lines[-1].get_color()
                ax.plot(times, p1, "o", color=color)
                ax.plot(times, p2, "o", color=color)
            else:
                raise ValueError("mode must be 'points', 'steps', or 'points+steps'.")
        ax.set_title(title)
        ax.set_xlabel("$t$")
        ax.set_ylabel("Coordinate processes")
        ax.legend(loc="best")
        plt.show()
    return fig


def plot_sample_2d(
    path,
    style="seaborn-v0_8-whitegrid",
    title=None,
    **fig_kw,
):
    x_positions, y_positions = path

    with plt.style.context(style):
        fig, ax = plt.subplots(**fig_kw)
        ax.plot(x_positions, y_positions, linewidth=1)
        ax.plot(x_positions[0], y_positions[0], "go", label="Start")  # Start point
        ax.plot(x_positions[-1], y_positions[-1], "ro", label="End")  # End point
        ax.set_title(title)
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        plt.grid(True)
        plt.legend()
        plt.axis("equal")  # Equal scaling for both axes
        plt.show()

    return fig
