import numpy as np

from aleatory.processes.base_analytical import SPAnalytical
from aleatory.utils.utils import check_positive_integer, get_times


class SPEulerMaruyama(SPAnalytical):
    def __int__(self, f=None, g=None, initial=0.0, T=1.0, rng=None):
        super().__init__(T=T, rng=rng, initial=initial)
        self.f = f
        self.g = g
        self.n = None
        self.dt = None
        self.times = None
        self.name = None

    def _sample_em_process(self, n, log=False):
        check_positive_integer(n)
        self.n = n
        self.dt = 1.0 * self.T / self.n
        self.times = get_times(self.T, self.n)
        dws = self.rng.normal(scale=np.sqrt(self.dt), size=self.n)

        if log:
            origin = np.log(self.initial)
        else:
            origin = self.initial

        path = [origin]
        previous = origin

        for (t, dw) in zip(self.times[1:], dws[1:]):
            previous += self.f(previous, t) * self.dt + self.g(previous, t) * dw
            path.append(previous)

        if log:
            return np.exp(path)

        return path

    def sample(self, n):
        return self._sample_em_process(n)
