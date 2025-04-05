from aleatory.processes.multi_dimensional.two_correlated_brownian_motions import (
    CorrelatedBMs,
)


class BM2D(CorrelatedBMs):
    def __init__(self, T=1):
        super().__init__(rho=0.0, T=T)
        self.name = "Brownian Motion 2D"


# if __name__ == "__main__":
#     p = BM2D()
#
#     f = p.plot_sample(n=100)
#     g = p.plot_sample_coordinates(n=100)
#     h = p.plot(n=100, N=3)
