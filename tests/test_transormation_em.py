from replica.processes import BrownianMotion, GBM, Vasicek, OUProcess, CIRProcess, CEVProcess



# process = CIRProcess(theta=1.0, mu=2.0, sigma=0.25)
process= CEVProcess(gamma=0.5, mu=0.50, sigma=0.1, initial=1.0)


pitaya = 'https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle'
process.draw(n=100, N=100, dpi=200)
process.draw(n=100, N=100, style=pitaya, colormap="PiYG", dpi=200)
process.draw(n=100, N=100,  dpi=200)
process.draw(n=100, N=100,  colormap="Spectral", dpi=200)
process.draw(n=100, N=100, style='Solarize_Light2', dpi=200)
process.plot(n=100, N=100,  style=pitaya)
process.draw(n=100, N=100, style='Solarize_Light2', colormap="coolwarm", dpi=200)
process.draw(n=100, N=100, style='ggplot', dpi=200)
process.draw(n=100, N=100, style='ggplot', colormap="Spectral", dpi=200)
process.draw(n=100, N=100, style='ggplot', colormap="BrBG", dpi=200)
process.draw(n=100, N=100, style='bmh', dpi=200)
process.draw(n=100, N=100, style='dark_background', dpi=200)
process.plot(n=100, N=10, figsize=(5, 3))



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
