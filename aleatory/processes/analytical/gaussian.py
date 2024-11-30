from aleatory.processes.analytical.increments import IndependentIncrements


class GaussianIncrements(IndependentIncrements):
    """
    Gaussian Independent Increments
    """

    def __init__(self, T=1.0, rng=None):
        super().__init__(T=T, rng=rng)
        self.name = "Gaussian Noise"

