from aleatory.processes import BrownianMotion, GBM, Vasicek, OUProcess, CIRProcess, CEVProcess
import numpy as np

SAVE = False
SAVE_PATH = '../docs/source/_static/'


def test_cir_convergence():
    process = CIRProcess()
    process.draw(n=100, N=200, )
    process = CIRProcess(T=10)
    process.draw(n=100, N=200)


def test_rng():
    seed = 123
    bm = BrownianMotion(drift=2.0, scale=0.5, rng=np.random.default_rng(seed=seed))
    bm.plot(n=100, N=10)


def test_figures():
    bm = BrownianMotion()
    bmd = BrownianMotion(drift=-1.0, scale=0.5)
    gbm = GBM()
    vasicek = Vasicek()
    ouprocess = OUProcess()
    cirprocess = CIRProcess()
    cev = CEVProcess()

    for process in [bm, bmd, gbm, vasicek, ouprocess, cirprocess, cev]:
        figure = process.plot(n=100, N=5)
        name = process.name.replace(" ", "_").lower()

        if SAVE:
            figure.savefig(SAVE_PATH + name + '_simple_plot.png', dpi=300)
        figure = process.draw(n=100, N=200, envelope=False)
        if SAVE:
            figure.savefig(SAVE_PATH + name + '_drawn.png', dpi=300)
        process.draw(n=100, N=200, envelope=True)
        process.draw(n=100, N=200, marginal=False)
        process.draw(n=100, N=200, marginal=False, envelope=True)


def test_quick_start():
    process = BrownianMotion()
    name = process.name.replace(" ", "_").lower()
    figure = process.plot(n=100, N=10)

    if SAVE:
        figure.savefig(SAVE_PATH + name + '_quickstart_01.png', dpi=300)

    figure = process.draw(n=100, N=200)
    if SAVE:
        figure.savefig(SAVE_PATH + name + '_quickstart_02.png', dpi=300)

    figure = process.draw(n=100, N=200, envelope=True)
    if SAVE:
        figure.savefig(SAVE_PATH + name + '_quickstart_03.png', dpi=300)

    figure = process.draw(n=100, N=200, marginal=False)
    if SAVE:
        figure.savefig(SAVE_PATH + name + '_quickstart_04.png', dpi=300)

    figure = process.draw(n=100, N=200, marginal=False, envelope=True)
    if SAVE:
        figure.savefig(SAVE_PATH + name + '_quickstart_05.png', dpi=300)

    figure = process.plot(n=100, N=200, style='ggplot')
    if SAVE:
        figure.savefig(SAVE_PATH + name + '_quickstart_06.png', dpi=300)

    figure = process.draw(n=100, N=200, style='Solarize_Light2')
    if SAVE:
        figure.savefig(SAVE_PATH + name + '_quickstart_07.png', dpi=300)

    figure = process.draw(n=100, N=200, colormap="cool", )
    if SAVE:
        figure.savefig(SAVE_PATH + name + '_quickstart_08.png', dpi=300)


test_cir_convergence()
test_rng()
test_figures()
test_quick_start()