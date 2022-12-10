from replica.processes.euler_maruyama.vasicek import Vasicek


class OUProcess(Vasicek):

    def __init__(self, theta=1.0, sigma=1.0, initial=0.0, T=1.0, rng=None):
        super().__init__(theta=theta, mu=0.0, sigma=sigma, initial=initial, T=T, rng=rng)
        self.n = 1.0
        self.dt = 1.0 * self.T / self.n
        self.times = None
        self.name = "Ornstein–Uhlenbeck process"

    def __str__(self):
        return "Ornstein–Uhlenbeckprocess with parameters {speed}, and {volatility} on [0, {T}].".format(
            T=str(self.T), speed=str(self.theta), volatility=str(self.sigma))