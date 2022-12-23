from aleatory.processes import CIRProcess, CEVProcess

pitaya = 'https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle'

cir = CIRProcess(theta=1.0, mu=2.0, sigma=0.25)
cev = CEVProcess(gamma=1.5, mu=0.50, sigma=0.1, initial=1.0)

for process in [cir, cev]:
    process.draw(n=100, N=100, dpi=200)
    process.draw(n=100, N=100, style=pitaya, colormap="PiYG", dpi=200)
    process.draw(n=100, N=100, dpi=200)
    process.draw(n=100, N=100, colormap="Spectral", dpi=200)
    process.draw(n=100, N=100, style='Solarize_Light2', dpi=200)
    process.plot(n=100, N=100, style=pitaya)
    process.draw(n=100, N=100, style='Solarize_Light2', colormap="coolwarm", dpi=200)
    process.draw(n=100, N=100, style='ggplot', dpi=200)
    process.draw(n=100, N=100, style='ggplot', colormap="Spectral", dpi=200)
    process.draw(n=100, N=100, style='ggplot', colormap="BrBG", dpi=200)
    process.draw(n=100, N=100, style='bmh', dpi=200)
    process.draw(n=100, N=100, style='dark_background', dpi=200)
    process.plot(n=100, N=10)
