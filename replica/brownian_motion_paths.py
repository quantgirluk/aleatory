from brownian_motion import BrownianMotion
import matplotlib.pyplot as plt


class BrownianPaths:

    def __init__(self, N, times, drift=0.0, scale=1.0):
        self.N = N
        self.times = times
        self.drift = drift
        self.scale = scale
        brownian = BrownianMotion(drift=self.drift, scale=self.scale)
        self.paths = [brownian.sample_at(times) for k in range(int(N))]

    def _draw_paths(self):
        for p in self.paths:
            plt.plot(self.times, p)
        plt.show()
        return 1

    def draw(self):
        self._draw_paths()
        return 1
