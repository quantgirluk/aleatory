import matplotlib.figure
import numpy as np

from aleatory.processes import (BrownianMotion, GBM, Vasicek, OUProcess, CIRProcess, CEVProcess, BESProcess,
                                BESQProcess, BrownianBridge, BrownianExcursion, BrownianMeander,
                                CKLSProcess, RandomWalk, PoissonProcess)

SAVE = False
SAVE_PATH = '../docs/source/_static/'


def test_sample(n=100):
    process = BrownianMotion()
    sample = process.sample(n)
    times = process.times
    grid_times = np.linspace(0, process.T, n)
    assert len(times) == len(grid_times)
    assert len(times) == len(sample)
    for (t1, t2) in zip(times, grid_times):
        assert t1 == t2


def test_cir_convergence():
    process = CIRProcess()
    process.draw(n=100, N=200)
    process = CIRProcess(T=10)
    process.draw(n=100, N=200)


def test_rng():
    seed = 123
    bm1 = BrownianMotion(drift=2.0, scale=0.5, rng=np.random.default_rng(seed=seed))
    sim1 = bm1.simulate(n=100, N=1)

    bm2 = BrownianMotion(drift=2.0, scale=0.5, rng=np.random.default_rng(seed=seed))
    sim2 = bm2.simulate(n=100, N=1)

    for v1, v2 in zip(sim1[0], sim2[0]):
        assert v1 == v2


def test_figures_examples():
    bm = BrownianMotion()
    bm_arithmetic = BrownianMotion(initial=1.0, drift=1.0, scale=2.5)
    bridge = BrownianBridge(initial=1.0, end=2.0)
    excursion = BrownianExcursion()
    meander = BrownianMeander()
    meander_end = BrownianMeander(fixed_end=True, end=2.0)
    meander_end_automatic = BrownianMeander(fixed_end=True)
    bmd = BrownianMotion(drift=-1.0, scale=0.5)
    gbm = GBM()
    vasicek = Vasicek()
    ouprocess = OUProcess()
    cirprocess = CIRProcess()
    cev = CEVProcess()
    bes = BESProcess(dim=10)
    bes_float = BESProcess(dim=4.5)
    besq = BESQProcess(dim=10)
    besq_float = BESQProcess(dim=3.5)
    ckls = CKLSProcess()

    # processes = [cirprocess]
    # processes = [bridge, excursion, meander, meander_end, meander_end_automatic]
    # processes = [cirprocess, cev]
    # processes = [bm, bmd, bridge, excursion, meander, gbm, vasicek, ouprocess, cirprocess, cev, bes, besq, bes_float, besq_float]
    # processes = [bm, besq_float, bes_float]
    # processes = [gbm, vasicek, ouprocess, cirprocess, cev, bes, besq]
    # processes = [bes_float, besq_float]
    # processes = [bm, bmd, bridge, excursion, meander, meander_end]
    # [bm, bmd, gbm,  cirprocess, cev, bes, besq, bes_float, besq_float]
    processes = [ckls]
    style = "https://raw.githubusercontent.com/quantgirluk/matplotlib-stylesheets/main/quant-pastel-light.mplstyle"
    # with plt.style.context(style):
    for process in processes:
        process.plot(n=100, N=5, figsize=(9.5, 6), dpi=200)
        # process.plot(n=100, N=5, title='My favourite figure', figsize=(9.5, 6), dpi=200)  # figure_with_title
        #
        name = process.name.replace(" ", "_").lower()
        if SAVE:
            figure = process.plot(n=100, N=5, figsize=(9.5, 6), dpi=200)
            figure.savefig(SAVE_PATH + name + '_simple_plot.png')
            figure = process.draw(n=100, N=200, figsize=(10, 6), dpi=200, colormap="spring")
            figure.savefig(SAVE_PATH + name + '_drawn.png')

        process.draw(n=100, N=200, figsize=(10, 6), dpi=200, colormap="spring", envelope=False, marginal=True)
        # process.draw(n=100, N=200, figsize=(10, 6), dpi=200, colormap="spring", envelope=True, marginal=True)
        # process.draw(n=100, N=200, envelope=False, dpi=200, orientation='horizontal')
        # process.draw(n=100, N=250, envelope=False, orientation='vertical', figsize=(12, 6))
        # process.draw(n=100, N=250, envelope=True, orientation='vertical', figsize=(12, 6), style=style)
        # process.draw(n=100, N=250, envelope=True, orientation='horizontal', figsize=(12, 6), style=style)
        # process.draw(n=100, N=1, envelope=False, figsize=(12, 6),)
        # process.draw(n=100, N=2, envelope=False, figsize=(12, 6), dpi=200)
        # process.draw(n=100, N=3, envelope=False, figsize=(12, 6))
        # process.draw(n=100, N=3, envelope=False, orientation='vertical', figsize=(12, 6))
        # process.draw(n=100, N=200, marginal=False, figsize=(9.5, 6), dpi=200)
        # process.draw(n=100, N=200, marginal=False, envelope=True, figsize=(9.5, 6), dpi=200)
        #
        # process.draw(n=100, N=200, figsize=(12, 6), dpi=200, title='My favourite figure')
        # process.draw(n=100, N=200, envelope=True, figsize=(12, 6), dpi=200, title='My favourite figure')
        # process.draw(n=100, N=200, marginal=False, figsize=(9.5, 6), dpi=200, title='My favourite figure')
        # process.draw(n=100, N=200, marginal=False, envelope=True, figsize=(9.5, 6), dpi=200,
        #              title='My favourite figure')


def test_charts_orientation():
    for process in [BrownianMotion(), Vasicek()]:
        process.draw(n=100, N=200, envelope=False, orientation='horizontal', figsize=(12, 6))
        process.draw(n=100, N=200, envelope=False, orientation='vertical', figsize=(12, 6))


def test_charts_modes():
    for sty in ["steps", "points", "linear"]:
        for process in [Vasicek()]:
            process.plot(n=100, N=200, figsize=(12, 6), mode=sty)
            process.draw(n=100, N=200, envelope=False, figsize=(12, 6), mode=sty)


def test_poisson():
    process = PoissonProcess()
    process.plot(N=10, T=100)
    process.draw(N=100, T=100, envelope=False, figsize=(12, 8))


def test_charts_type():
    for process in [BrownianMotion(), Vasicek()]:
        fig = process.draw(n=100, N=200, envelope=False, figsize=(12, 6))
        assert (isinstance(fig, matplotlib.figure.Figure))

        fig = process.draw(n=100, N=200, envelope=False, orientation='vertical', figsize=(12, 6))
        assert (isinstance(fig, matplotlib.figure.Figure))


def test_quick_start():
    process = BrownianMotion()
    name = process.name.replace(" ", "_").lower()

    figure = process.plot(n=100, N=10, figsize=(9.5, 6), dpi=100)
    if SAVE:
        figure.savefig(SAVE_PATH + name + '_quickstart_01.png')

    figure = process.draw(n=100, N=200, figsize=(12, 6), dpi=100)
    if SAVE:
        figure.savefig(SAVE_PATH + name + '_quickstart_02.png')

    figure = process.draw(n=100, N=200, envelope=True, figsize=(12, 6), dpi=100)
    if SAVE:
        figure.savefig(SAVE_PATH + name + '_quickstart_03.png')

    figure = process.draw(n=100, N=200, marginal=False, figsize=(9.5, 6), dpi=100)
    if SAVE:
        figure.savefig(SAVE_PATH + name + '_quickstart_04.png', figsize=(12, 6), dpi=100)

    figure = process.draw(n=100, N=200, marginal=False, envelope=True, figsize=(9.5, 6), dpi=100)
    if SAVE:
        figure.savefig(SAVE_PATH + name + '_quickstart_05.png', figsize=(12, 6), dpi=100)

    figure = process.plot(n=100, N=200, style='ggplot', figsize=(12, 6), dpi=100)
    if SAVE:
        figure.savefig(SAVE_PATH + name + '_quickstart_06.png')

    figure = process.draw(n=100, N=200, style='Solarize_Light2', figsize=(12, 6), dpi=100)
    if SAVE:
        figure.savefig(SAVE_PATH + name + '_quickstart_07.png')

    figure = process.draw(n=100, N=200, colormap="cool", figsize=(12, 6), dpi=100)
    if SAVE:
        figure.savefig(SAVE_PATH + name + '_quickstart_08.png', dpi=300)
