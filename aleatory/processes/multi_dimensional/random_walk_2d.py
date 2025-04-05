from aleatory.processes.base import StochasticProcess
from aleatory.processes.jump.random_walk import RandomWalk
from aleatory.utils.plotters_2d import plot_paths_coordinates, plot_sample_2d
from aleatory.utils.utils import (
    check_positive_integer,
    get_times,
)


class RandomWalk2D(StochasticProcess):

    def __init__(self, rng1=None, rng2=None):
        super().__init__()
        self.x_process = RandomWalk(rng=rng1)
        self.y_process = RandomWalk(rng=rng2)
        self.name = "Random Walk 2D"
        self.n = None
        self.times = None

    def sample(self, n):
        check_positive_integer(n)
        self.T = n
        self.n = n
        self.times = get_times(self.T, self.n + 1)
        x_steps = self.x_process.sample(n)
        y_steps = self.y_process.sample(n)
        path = (x_steps, y_steps)
        return path

    def plot_sample(
        self,
        n,
        coordinates=False,
        title=None,
        style="seaborn-v0_8-whitegrid",
        mode="steps",
        **fig_kw,
    ):
        if coordinates:
            fig = self.plot_sample_coordinates(
                n=n, title=title, style=style, mode=mode, **fig_kw
            )
        else:
            fig = self.plot_sample_2d(n=n, title=title, style=style, **fig_kw)

        return fig

    def plot_sample_coordinates(
        self, n, title=None, style="seaborn-v0_8-whitegrid", **fig_kw
    ):
        chart_title = title if title is not None else self.name
        X, Y = self.sample(n)
        times = self.times
        fig = plot_paths_coordinates(
            times=times,
            paths1=[X],
            paths2=[Y],
            style=style,
            title=chart_title,
            **fig_kw,
        )
        return fig

    def plot_sample_2d(self, n, title=None, style="seaborn-v0_8-whitegrid", **fig_kw):
        chart_title = title if title is not None else self.name
        sample = self.sample(n)
        fig = plot_sample_2d(sample, title=chart_title, style=style, **fig_kw)

        return fig


# if __name__ == "__main__":
#     p = RandomWalk2d()
#     f = p.plot_sample_coordinates(n=30)
#     g = p.plot_sample(n=200)
