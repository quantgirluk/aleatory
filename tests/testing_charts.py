from replica.processes import BrownianMotion, GBM, Vasicek, OUProcess, CIRProcess, CEVProcess

# process = Vasicek(theta=2.0, mu=0.04, sigma=0.2, initial=0.08)
# process.plot(n=200, N=3)
# process.draw(n=100, N=100, envelope=False)
#
# from replica.processes import BrownianMotion

# process = BrownianMotion(drift=2.0, scale=1.5)
# process = OUProcess(theta=1.5, sigma=0.9, initial=4.0)
# process = Vasicek(theta=2.0, mu=0.04, sigma=0.2, initial=0.08)
# process = Vasicek(theta=2.0, mu=0.5, sigma=4.0, initial=5.0)
# process= CEVProcess(gamma=1.0, mu=1.50, sigma=0.8, initial=1.0)

# process.plot(n=100, N=10)
# process.plot(n=100, N=10, style='ggplot')
# process.plot(n=100, N=10, style='bmh')
# process.plot(n=100, N=10, style='dark_background')
# process.plot(n=100, N=10, figsize=(10, 6))


# process.draw(n=100, N=100)
# process.draw(n=100, N=100, figsize=(12, 6), dpi=200)
# process.draw(n=100, N=100, figsize=(10, 5), dpi=200)
# process.draw(n=100, N=100)
# process.draw(n=100, N=100, figsize=(12, 6), colormap="viridis", dpi=200)
# process.draw(n=100, N=100,  colormap="Spectral", dpi=200)
# process.draw(n=100, N=100, style='Solarize_Light2', dpi=200)
# process.draw(n=100, N=100, style='Solarize_Light2', colormap="coolwarm", dpi=200)
# process.draw(n=100, N=100, style='ggplot', dpi=200)
# process.draw(n=100, N=100, style='ggplot', colormap="Spectral", dpi=200)
# process.draw(n=100, N=100, style='ggplot', colormap="BrBG", dpi=200)

# pitaya = 'https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle'
# process.draw(n=100, N=100, style=pitaya, colormap="spring", dpi=200)
# process.draw(n=100, N=100, style=pitaya, colormap="PiYG", dpi=200)
# process.draw(n=100, N=100,  dpi=200)
# process.draw(n=100, N=100,  colormap="Spectral", dpi=200)
# process.draw(n=100, N=100, style='Solarize_Light2', dpi=200)
# process.plot(n=100, N=100,  style=pitaya)
# process.draw(n=100, N=100, style='Solarize_Light2', colormap="coolwarm", dpi=200)
# process.draw(n=100, N=100, style='ggplot', dpi=200)
# process.draw(n=100, N=100, style='ggplot', colormap="Spectral", dpi=200)
# process.draw(n=100, N=100, style='ggplot', colormap="BrBG", dpi=200)
# process.draw(n=100, N=100, style='bmh', dpi=200)
# process.draw(n=100, N=100, style='dark_background', dpi=200)
# process.plot(n=100, N=10, figsize=(5, 3))

SAVE = False
FIGURES = True
QS = True
CIR_convergenge = True


if CIR_convergenge:
    process = CIRProcess()
    process.draw(n=100, N=200,)
    process = CIRProcess(T=10)
    process.draw(n=100, N=200)




if FIGURES:

    bm = BrownianMotion()
    bmd = BrownianMotion(drift=-1.0, scale=0.5)
    gbm = GBM()
    vasicek = Vasicek()
    ouprocess = OUProcess()
    cirprocess = CIRProcess()
    cev = CEVProcess()

    for process in [bm,
                    # bmd, gbm, vasicek, ouprocess,cirprocess,
                    cev]:
        figure = process.plot(n=100, N=5)
        name = process.name.replace(" ", "_").lower()
        save_paths = '../docs/source/_static/'
        if SAVE:
            figure.savefig(save_paths + name + '_simple_plot.png', dpi=300)
        figure = process.draw(n=100, N=200, envelope=False)
        if SAVE:
            figure.savefig(save_paths + name + '_drawn.png', dpi=300)
        process.draw(n=100, N=200, envelope=True)
        # process.draw(n=100, N=200, marginal=False)
        # process.draw(n=100, N=200, marginal=False, envelope=True)

if QS:

    process = BrownianMotion()
    name = process.name.replace(" ", "_").lower()
    figure = process.plot(n=100, N=10)

    if SAVE:
        figure.savefig(save_paths + name + '_quickstart_01.png', dpi=300)

    figure = process.draw(n=100, N=200)
    if SAVE:
        figure.savefig(save_paths + name + '_quickstart_02.png', dpi=300)

    figure = process.draw(n=100, N=200, envelope=True)
    if SAVE:
        figure.savefig(save_paths + name + '_quickstart_03.png', dpi=300)

    figure = process.draw(n=100, N=200, marginal=False)
    if SAVE:
        figure.savefig(save_paths + name + '_quickstart_04.png', dpi=300)

    figure = process.draw(n=100, N=200, marginal=False, envelope=True)
    if SAVE:
        figure.savefig(save_paths + name + '_quickstart_05.png', dpi=300)

    figure = process.plot(n=100, N=200, style='ggplot')
    if SAVE:
        figure.savefig(save_paths + name + '_quickstart_06.png', dpi=300)

    figure = process.draw(n=100, N=200, style='Solarize_Light2')
    if SAVE:
        figure.savefig(save_paths + name + '_quickstart_07.png', dpi=300)

    figure = process.draw(n=100, N=200, colormap="cool", )
    if SAVE:
        figure.savefig(save_paths + name + '_quickstart_08.png', dpi=300)
