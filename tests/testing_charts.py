from replica.processes import BrownianMotion, GBM, Vasicek, OUProcess, CIRProcess, CEVProcess


# process = Vasicek(theta=2.0, mu=0.04, sigma=0.2, initial=0.08)
# process.plot(n=200, N=3)
# process.draw(n=100, N=100, envelope=False)
#
# from replica.processes import BrownianMotion

process = BrownianMotion()
figure = process.plot(n=100, N=10)
name = process.name.replace(" ", "_").lower()
figure.savefig(name + '_quickstart_01.png', dpi=300)

# figure = process.draw(n=100, N=200)
# figure.savefig(name + '_quickstart_02.png', dpi=300)
#
# figure = process.draw(n=100, N=200, envelope=True)
# figure.savefig(name + '_quickstart_03.png', dpi=300)
#
# figure = process.draw(n=100, N=200, marginal=False)
# figure.savefig(name + '_quickstart_04.png', dpi=300)
#
# figure = process.draw(n=100, N=200, marginal=False, envelope=True)
# figure.savefig(name + '_quickstart_05.png', dpi=300)
