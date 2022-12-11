from replica.processes import BrownianMotion, GBM, Vasicek, OUProcess, CIRProcess, CEVProcess


process = Vasicek(theta=2.0, mu=0.04, sigma=0.2, initial=0.08)
process.plot(n=200, N=3)
process.draw(n=100, N=100, envelope=False)