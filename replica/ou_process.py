from base_EMProcess import BaseEulerMaruyamaProcess


class OUProcess(BaseEulerMaruyamaProcess):

    def __init__(self, theta=1.0, mu=1.0, sigma=1.0, initial=0.0, T=1.0, rng=None):
        super().__init__(T=T, rng=rng)
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.initial = initial

        def f(x, _):
            return self.theta * (self.mu - x)

        def g(x, _):
            return self.sigma * x

        self.f = f
        self.g = g
