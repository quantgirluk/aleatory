from base_EMProcess import BaseEulerMaruyamaProcess


class OUProcess(BaseEulerMaruyamaProcess):

    def __init__(self, theta=1.0, mu=1.0, sigma=1.0, initial=0.0, T=1.0, n=1, rng=None):
        super().__init__(T=T, rng=rng)
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.initial = initial
        self.n = n
        self.dt = 1.0 * self.T / self.n
        self.times = None
        self.name = "Ornstein–Uhlenbeck process"

        def f(x, _):
            return self.theta * (self.mu - x)

        def g(x, _):
            return self.sigma

        self.f = f
        self.g = g

    def __str__(self):
        return "Orstein-Uhlenbeck process with parameters {speed}, {mean}, and {volatility} on [0, {T}].".format(
            T=str(self.T), speed=str(self.theta), mean=str(self.mu), volatility=str(self.sigma))
