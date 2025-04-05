from aleatory.processes.multi_dimensional.two_correlated_brownian_motions import (
    CorrelatedBMs,
)


class BM2D(CorrelatedBMs):
    def __init__(self, T=1):
        super().__init__(rho=0.0, T=T)
        self.name = "Brownian Motion 2D"
