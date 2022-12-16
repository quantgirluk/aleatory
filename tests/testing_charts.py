from replica.processes import BrownianMotion, GBM, Vasicek, OUProcess, CIRProcess, CEVProcess


# process = Vasicek(theta=2.0, mu=0.04, sigma=0.2, initial=0.08)
# process.plot(n=200, N=3)
# process.draw(n=100, N=100, envelope=False)
#
# from replica.processes import BrownianMotion

# process = BrownianMotion()
process = OUProcess(theta=1.5, sigma=0.9, initial=4.0)
# process = Vasicek(theta=2.0, mu=0.04, sigma=0.2, initial=0.08)
# process = Vasicek(theta=2.0, mu=0.5, sigma=4.0, initial=5.0)
# process= CEVProcess(gamma=0.5, mu=1.50, sigma=0.6, initial=1.0)
# process.plot(n=100, N=10)
# process.plot(n=100, N=10, style='ggplot')
# process.plot(n=100, N=10, style='bmh')
# process.plot(n=100, N=10, style='dark_background')
# process.plot(n=100, N=10, figsize=(10, 6))


# process.draw(n=100, N=100)
# process.draw(n=100, N=100, figsize=(12, 6), dpi=200)
# process.draw(n=100, N=100, figsize=(10, 5), dpi=200)
process.draw(n=100, N=100, figsize=(12, 6), dpi=200)
process.draw(n=100, N=100, figsize=(12, 6), colormap="viridis", dpi=200)
process.draw(n=100, N=100, figsize=(12, 6), colormap="Spectral", dpi=200)
process.draw(n=100, N=100, style='Solarize_Light2', figsize=(12, 6), dpi=200)
process.draw(n=100, N=100, style='Solarize_Light2', colormap="coolwarm", figsize=(12, 6), dpi=200)
process.draw(n=100, N=100, style='ggplot', figsize=(12, 6), dpi=200)
process.draw(n=100, N=100, style='ggplot', colormap="Spectral",figsize=(12, 6), dpi=200)
# process.draw(n=100, N=100, style='bmh', figsize=(12, 6), dpi=200)
# process.draw(n=100, N=100, style='dark_background', figsize=(12, 6), dpi=200)
# process.plot(n=100, N=10, figsize=(5, 3))



# figure = process.plot(n=100, N=10)
name = process.name.replace(" ", "_").lower()
# figure.savefig(name + '_quickstart_01.png', dpi=300)

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
