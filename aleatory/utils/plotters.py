import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from statsmodels import api as sm


def plot_paths(
    times, paths, style="seaborn-v0_8-whitegrid", title=None, mode="linear", **fig_kw
):
    with plt.style.context(style):
        fig, ax = plt.subplots(**fig_kw)
        for p in paths:
            if mode == "points":
                ax.scatter(times, p, s=7)
            elif mode == "steps":
                ax.step(times, p, where="post")
            elif mode in ["points+steps", "steps+points"]:
                ax.step(times, p, where="post")
                color = plt.gca().lines[-1].get_color()
                ax.plot(times, p, "o", color=color)
            elif mode == "linear":
                ax.plot(times, p)
            else:
                raise ValueError("mode must be 'points', 'steps', or 'linear'.")
        ax.set_title(title)
        ax.set_xlabel("$t$")
        ax.set_ylabel("$X(t)$")
        plt.show()
    return fig


def draw_paths(
    times,
    paths,
    N,
    expectations,
    title=None,
    KDE=False,
    marginal=False,
    orientation="horizontal",
    marginalT=None,
    envelope=False,
    lower=None,
    upper=None,
    style="seaborn-v0_8-whitegrid",
    colormap="RdYlBu_r",
    **fig_kw,
):
    if orientation == "horizontal":
        return draw_paths_horizontal(
            times,
            paths,
            N,
            expectations,
            title=title,
            KDE=KDE,
            marginal=marginal,
            marginalT=marginalT,
            envelope=envelope,
            lower=lower,
            upper=upper,
            style=style,
            colormap=colormap,
            **fig_kw,
        )
    elif orientation == "vertical":
        return draw_paths_vertical(
            times,
            paths,
            N,
            expectations,
            title=title,
            KDE=KDE,
            marginal=marginal,
            marginalT=marginalT,
            envelope=envelope,
            lower=lower,
            upper=upper,
            style=style,
            colormap=colormap,
            **fig_kw,
        )
    else:
        raise ValueError("orientation can only take values horizontal, vertical")


def draw_paths_horizontal(
    times,
    paths,
    N,
    expectations=None,
    title=None,
    KDE=False,
    marginal=False,
    marginalT=None,
    envelope=False,
    lower=None,
    upper=None,
    style="seaborn-v0_8-whitegrid",
    colormap="RdYlBu_r",
    colorspos=None,
    mode="linear",
    estimate_quantiles=False,
    **fig_kw,
):
    cm = plt.colormaps[colormap]
    last_points = [path[-1] for path in paths]
    n_bins = int(np.sqrt(N))
    col = np.linspace(0, 1, n_bins, endpoint=True)

    with plt.style.context(style):
        if marginal:
            fig = plt.figure(**fig_kw)
            gs = GridSpec(1, 5)
            ax1 = fig.add_subplot(gs[:4])
            ax2 = fig.add_subplot(gs[4:], sharey=ax1)

            n, bins, patches = ax2.hist(
                last_points, n_bins, orientation="horizontal", density=True
            )
            for c, p in zip(col, patches):
                plt.setp(p, "facecolor", cm(c))
            my_bins = pd.cut(
                last_points, bins=bins, labels=range(len(bins) - 1), include_lowest=True
            )
            colors = [col[b] for b in my_bins]

            if KDE:
                kde = sm.nonparametric.KDEUnivariate(last_points)
                kde.fit()  # Estimate the densities
                ax2.plot(
                    kde.density,
                    kde.support,
                    "--",
                    lw=1.75,
                    alpha=0.6,
                    label="$X_T$  KDE",
                    zorder=10,
                )
                ax2.axhline(
                    y=np.mean(last_points),
                    linestyle="--",
                    lw=1.75,
                    label=r"$\overline{X_T}$",
                )
                ax2.legend()

            elif marginal and marginalT:
                marginaldist = marginalT
                # lower_q = marginaldist.ppf(0.001)
                # upper_q = marginaldist.ppf(0.999)

                if estimate_quantiles:
                    lower_val = np.min(last_points)
                    upper_val = np.max(last_points)
                else:
                    lower_val = marginaldist.ppf(0.001)
                    upper_val = marginaldist.ppf(0.999)
                x = np.linspace(lower_val, upper_val, 100)
                ax2.plot(
                    marginaldist.pdf(x), x, "-", lw=1.75, alpha=0.6, label="$X_T$ pdf"
                )
                ax2.axhline(
                    y=marginaldist.mean(), linestyle="--", lw=1.75, label="$E[X_T]$"
                )
                ax2.legend()

            plt.setp(ax2.get_yticklabels(), visible=False)
            ax2.set_title("$X_T$")

            for i in range(N):
                if mode == "linear":
                    ax1.plot(times, paths[i], "-", lw=1.0, color=cm(colors[i]))
                elif mode == "points":
                    ax1.scatter(times, paths[i], s=7, color=cm(colors[i]))
                elif mode == "steps":
                    ax1.step(times, paths[i], color=cm(colors[i]), where="post")
                elif mode in ["steps+points", "points+steps"]:
                    ax1.step(times, paths[i], color=cm(colors[i]), where="post")
                    ax1.scatter(times, paths[i], s=7, color=cm(colors[i]))
                else:
                    raise ValueError(
                        "mode must be 'linear', 'points', 'steps', 'steps+points'."
                    )

            if expectations is not None:
                ax1.plot(times, expectations, "--", lw=1.75, label="$E[X_t]$")
                ax1.legend()
            if envelope:
                ax1.fill_between(times, upper, lower, alpha=0.25, color="silver")
            plt.subplots_adjust(wspace=0.025, hspace=0.025)

        else:
            fig, ax1 = plt.subplots(**fig_kw)
            if colorspos:
                colors = [path[colorspos] / np.max(np.abs(path)) for path in paths]
            else:
                _, bins = np.histogram(last_points, n_bins)
                my_bins = pd.cut(
                    last_points,
                    bins=bins,
                    labels=range(len(bins) - 1),
                    include_lowest=True,
                )
                colors = [col[b] for b in my_bins]

            for path, color in zip(paths, colors):

                if mode == "linear":
                    ax1.plot(times, path, "-", color=cm(color), lw=0.75)
                elif mode in ["points"]:
                    ax1.scatter(times, path, s=7, color=cm(color))
                elif mode == "steps":
                    ax1.step(times, path, color=cm(color), where="post")
                elif mode in ["steps+points", "points+steps"]:
                    ax1.step(times, path, color=cm(color), where="post")
                    ax1.scatter(times, path, s=7, color=cm(color))
                else:
                    raise ValueError(
                        "mode must be 'linear', 'points', 'steps', 'steps+points'."
                    )
            if expectations is not None:
                ax1.plot(times, expectations, "--", lw=1.75, label="$E[X_t]$")
                ax1.legend()
            if envelope:
                ax1.fill_between(times, upper, lower, color="silver", alpha=0.25)

        fig.suptitle(title)
        ax1.set_title(r"Simulated Paths $X_t, t \in [t_0, T]$")  # Title
        ax1.set_xlabel("$t$")
        ax1.set_ylabel("$X(t)$")
        plt.show()

    return fig


def draw_paths_vertical(
    times,
    paths,
    N,
    expectations,
    title=None,
    KDE=False,
    marginal=False,
    marginalT=None,
    envelope=False,
    lower=None,
    upper=None,
    style="seaborn-v0_8-whitegrid",
    colormap="RdYlBu_r",
    mode="linear",
    **fig_kw,
):
    with plt.style.context(style):

        cm = plt.colormaps[colormap]
        last_points = [path[-1] for path in paths]
        n_bins = int(np.sqrt(N))
        col = np.linspace(0, 1, n_bins, endpoint=True)

        if marginal:
            fig = plt.figure(**fig_kw)
            gs = GridSpec(1, 7)
            ax1 = fig.add_subplot(gs[:4])
            ax2 = fig.add_subplot(gs[4:])

            n, bins, patches = ax2.hist(
                last_points, n_bins, orientation="vertical", density=True
            )
            for c, p in zip(col, patches):
                plt.setp(p, "facecolor", cm(c))
            my_bins = pd.cut(
                last_points, bins=bins, labels=range(len(bins) - 1), include_lowest=True
            )
            colors = [col[b] for b in my_bins]

            if KDE:
                kde = sm.nonparametric.KDEUnivariate(last_points)
                kde.fit()  # Estimate the densities
                ax2.plot(
                    kde.support,
                    kde.density,
                    "--",
                    lw=1.75,
                    alpha=0.6,
                    label="$X_T$  KDE",
                    zorder=10,
                )
                ax2.axvline(
                    x=np.mean(last_points),
                    linestyle="--",
                    lw=1.75,
                    label=r"$\overline{X_T}$",
                )
            elif marginal and marginalT:
                marginaldist = marginalT

                lower_val = np.min(last_points)
                upper_val = np.max(last_points)
                x = np.linspace(lower_val, upper_val, 100)
                # x = np.linspace(marginaldist.ppf(0.001), marginaldist.ppf(0.999), 100)
                ax2.plot(
                    x, marginaldist.pdf(x), "-", lw=1.75, alpha=0.6, label="$X_T$ pdf"
                )
                ax2.axvline(
                    x=marginaldist.mean(), linestyle="--", lw=1.75, label="$E[X_T]$"
                )

            ax2.set_title("$X_T$")
            ax2.legend()
            ax2.yaxis.tick_right()

            for i in range(N):
                if mode == "points":
                    ax1.scatter(times, paths[i], s=7, color=cm(colors[i]))
                elif mode == "steps":
                    ax1.step(times, paths[i], color=cm(colors[i]), where="post")
                elif mode in ["steps+points", "points+steps"]:
                    ax1.step(times, paths[i], color=cm(colors[i]), where="post")
                    ax1.scatter(times, paths[i], s=7, color=cm(colors[i]))
                elif mode == "linear":
                    ax1.plot(times, paths[i], "-", lw=1.0, color=cm(colors[i]))
                else:
                    raise ValueError("mode must be 'points', 'steps', or 'linear'.")

            ax1.plot(times, expectations, "--", lw=1.75, label="$E[X_t]$")
            if envelope:
                ax1.fill_between(times, upper, lower, alpha=0.25, color="grey")

        else:
            _, bins = np.histogram(last_points, n_bins)
            my_bins = pd.cut(
                last_points, bins=bins, labels=range(len(bins) - 1), include_lowest=True
            )
            colors = [col[b] for b in my_bins]

            fig, ax1 = plt.subplots(**fig_kw)
            for i in range(N):
                ax1.plot(times, paths[i], "-", color=cm(colors[i]), lw=1.0)
            ax1.plot(times, expectations, "--", lw=1.75, label="$E[X_t]$")
            if envelope:
                ax1.fill_between(times, upper, lower, color="silver", alpha=0.25)

        fig.suptitle(title)
        ax1.set_title(r"Simulated Paths $X_t, t \in [t_0, T]$")  # Title
        ax1.set_xlabel("$t$")
        ax1.set_ylabel("$X(t)$")
        ax1.legend()
        plt.show()

    return fig


def draw_paths_with_end_point(
    times,
    paths,
    expectations=None,
    title=None,
    envelope=False,
    lower=None,
    upper=None,
    style="seaborn-v0_8-whitegrid",
    colormap="RdYlBu_r",
    **fig_kw,
):
    cm = plt.colormaps[colormap]
    mid = int(len(paths[0]) / 2) + 1

    with plt.style.context(style):

        fig, ax1 = plt.subplots(**fig_kw)
        for path in paths:
            ax1.plot(
                times, path, "-", color=cm(path[mid] / np.max(np.abs(path))), lw=0.75
            )
        if expectations is not None:
            ax1.plot(times, expectations, "--", lw=1.75, label="$E[X_t]$")
            ax1.legend()

        lower_and_upper_provided = lower is not None and upper is not None
        if envelope and lower_and_upper_provided:
            ax1.fill_between(times, upper, lower, color="silver", alpha=0.25)

        fig.suptitle(title)
        ax1.set_title(r"Simulated Paths $X_t, t \in [t_0, T]$")  # Title
        ax1.set_xlabel("$t$")
        ax1.set_ylabel("$X(t)$")
        plt.show()

    return fig


def draw_poisson_like(
    T,
    paths,
    marginalT=None,
    expectations=None,
    envelope=False,
    lower=None,
    upper=None,
    style="seaborn-v0_8-whitegrid",
    colormap="RdYlBu_r",
    marginal=True,
    mode="steps",
    colorspos=None,
    title=None,
    **fig_kw,
):

    times = np.linspace(0.0, T, 200)
    N = len(paths)

    cm = plt.colormaps[colormap]
    last_points = [len(path) - 1 for path in paths]
    n_bins = int(np.sqrt(N))
    col = np.linspace(0, 1, n_bins, endpoint=True)

    with plt.style.context(style):
        if marginal:
            fig = plt.figure(**fig_kw)
            gs = GridSpec(1, 5)
            ax1 = fig.add_subplot(gs[:4])
            ax2 = fig.add_subplot(gs[4:], sharey=ax1)

            n, bins, patches = ax2.hist(
                last_points, n_bins, orientation="horizontal", density=True
            )
            for c, p in zip(col, patches):
                plt.setp(p, "facecolor", cm(c))
            my_bins = pd.cut(
                last_points,
                bins=bins,
                labels=range(len(bins) - 1),
                include_lowest=True,
            )
            colors = [col[b] for b in my_bins]

            if marginalT:
                marginaldist = marginalT
                x = np.arange(marginaldist.ppf(0.001), marginaldist.ppf(0.999) + 1)
                ax2.plot(
                    marginaldist.pmf(x),
                    x,
                    "o",
                    linestyle="",
                    color="maroon",
                    markersize=2,
                    label="$N_T$ pmf",
                )
                ax2.axhline(
                    y=marginaldist.mean(), linestyle="--", lw=1.75, label="$E[N_T]$"
                )
                ax2.legend()

            plt.setp(ax2.get_yticklabels(), visible=False)
            ax2.set_title("$N_T$")

            for i, p in enumerate(paths):
                counts = np.arange(0, len(p))
                if mode == "points":
                    ax1.scatter(p, counts, color=cm(colors[i]), s=10)
                elif mode == "steps":
                    ax1.step(
                        p, counts, color=cm(colors[i]), where="post", linewidth=1.25
                    )
                elif mode in ["points+steps", "steps+points"]:
                    ax1.step(
                        p, counts, color=cm(colors[i]), where="post", linewidth=1.25
                    )
                    ax1.plot(p, counts, "o", color=cm(colors[i]), markersize=6)
                else:
                    raise ValueError(
                        "mode can only take values 'points', 'steps' or 'points+steps'"
                    )

            if expectations:
                ax1.plot(times, expectations, "--", lw=1.75, label="$E[N_t]$")
                ax1.legend()
            if envelope:
                ax1.fill_between(times, upper, lower, alpha=0.25, color="silver")
            plt.subplots_adjust(wspace=0.2, hspace=0.5)

        else:
            fig, ax1 = plt.subplots(**fig_kw)
            if colorspos:
                colors = [path[colorspos] / np.max(np.abs(path)) for path in paths]
            else:
                _, bins = np.histogram(last_points, n_bins)
                my_bins = pd.cut(
                    last_points,
                    bins=bins,
                    labels=range(len(bins) - 1),
                    include_lowest=True,
                )
                colors = [col[b] for b in my_bins]

            for i in range(N):
                counts = np.arange(0, len(paths[i]))
                ax1.step(paths[i], counts, color=cm(colors[i]), lw=0.75, where="post")

            if expectations is not None:
                ax1.plot(times, expectations, "--", lw=1.75, label="$E[N_t]$")
                ax1.legend()

            if envelope:
                ax1.fill_between(times, upper, lower, color="lightgray", alpha=0.25)

        fig.suptitle(title)
        ax1.set_xlim(left=0.0, right=T)
        ax1.set_title(r"Simulated Paths $N_t, t \leq T$")  # Title
        ax1.set_xlabel("$t$")
        ax1.set_ylabel("$N(t)$")
        plt.show()

    return fig


def plot_paths_random_walk(
    *args,
    times,
    paths,
    style="seaborn-v0_8-whitegrid",
    title=None,
    mode="steps",
    **fig_kw,
):
    with plt.style.context(style):
        fig, ax = plt.subplots(**fig_kw)
        for p in paths:
            if mode == "points":
                ax.scatter(times, p, s=7)
            elif mode == "steps":
                ax.step(times, p, where="post")
            elif mode == "linear":
                ax.plot(times, p, *args)
            elif mode in ["points+steps", "steps+points"]:
                ax.step(times, p, where="post")
                color = plt.gca().lines[-1].get_color()
                ax.plot(times, p, "o", color=color)
            else:
                raise ValueError("mode must be 'points', 'steps', or 'points+steps'.")
        ax.set_title(title)
        ax.set_xlabel("$t$")
        ax.set_ylabel("$X(t)$")
        plt.show()
    return fig


def plot_poisson(
    jumps=None,
    T=None,
    paths=None,
    style="seaborn-v0_8-whitegrid",
    mode="steps",
    title=None,
    **fig_kw,
):
    """
    Simulates and plots paths/trajectories from the instanced stochastic process. Simple plot of times
    :param int jumps: number of jumps
    :param float T: time T
    :param list paths : a list containing the simulated paths
    :param str style: style of plot
    :param str mode: type of plot
    :param str title: title of plot
    """

    if jumps and T:
        raise ValueError("Only one must be provided either jumps or T")

    with plt.style.context(style):
        fig, ax = plt.subplots(**fig_kw)
        for p in paths:
            counts = np.arange(0, len(p))
            if mode == "points":
                ax.scatter(p, counts, s=10)
            elif mode == "steps":
                ax.step(p, counts, where="post", linewidth=1.25)
            elif mode == "linear":
                ax.plot(p, counts)
            elif mode in ["points+steps", "steps+points"]:
                ax.step(p, counts, where="post", alpha=0.5)
                color = plt.gca().lines[-1].get_color()
                ax.plot(p, counts, "o", color=color, markersize=6)

        ax.set_title(title)
        ax.set_xlabel("$t$")
        ax.set_ylabel("$N(t)$")
        if T is not None:
            ax.set_xlim(left=0.0, right=T)
        if jumps is not None:
            ax.set_ylim(top=jumps + 2)
        plt.show()

    return fig
