from aleatory.processes import CKLSProcess, CEVProcess
import unittest


def test_ckls_process():
    name = "Chan-Karolyi-Longstaff-Sanders (CKLS) Process"
    process = CKLSProcess(alpha=0.5, beta=1.5, sigma=0.5, gamma=0.3, initial=1.0, T=1.0)
    # fig = process.plot(n=100, N=10, figsize=(9, 6), dpi=200, title=name)
    # fig.savefig("ckls1.png")
    fig = process.draw(n=100, N=300, figsize=(11, 6), dpi=200, title=name, envelope=True)
    # fig.savefig("ckls2.png")
    # process = CKLSProcess(alpha=0.5, beta=-2.0, sigma=0.5, gamma=0.0, initial=1.0, T=1.0)
    # fig = process.draw(n=100, N=300, colormap="PuOr", figsize=(11, 6), dpi=200, title=name)
    # fig.savefig("ckls3.png")
    # process = CKLSProcess(alpha=0.0, beta=-2.0, sigma=0.5, gamma=1.0, initial=3.0, T=1.0)
    # fig = process.draw(n=100, N=300, colormap="PiYG", figsize=(11, 6), dpi=200, title=name)
    # fig.savefig("ckls4.png")
    # process = CKLSProcess(alpha=0.0, beta=0.0, sigma=0.5, gamma=0.0, initial=2.0, T=1.0)
    # fig = process.draw(n=100, N=300, colormap="Spectral", figsize=(11, 6), dpi=200, title=name)
    # fig.savefig("ckls5.png")
    # process.draw(n=100, N=200, envelope=False, figsize=(12, 6))
    # process.draw(n=100, N=200, envelope=False, orientation='horizontal', figsize=(12, 6))
    # process.draw(n=100, N=200, envelope=False, orientation='vertical', figsize=(12, 6), )

class test_ckls:
    vis = False

    @unittest.skipIf(not vis, "No Visualisation Required")
    def test_ckls_process(self):
        process = CKLSProcess(alpha=0.5, beta=0.5, sigma=0.1, gamma=1.0, initial=1.0, T=1.0)
        process.plot(n=100, N=5, figsize=(9.5, 6), dpi=200)
        process.draw(n=100, N=200)
        process.draw(n=100, N=200, figsize=(10, 6), dpi=200, colormap="spring")
        process.draw(n=100, N=200, envelope=False, figsize=(12, 6))
        process.draw(n=100, N=200, envelope=False, orientation='horizontal', figsize=(12, 6))
        process.draw(n=100, N=200, envelope=False, orientation='vertical', figsize=(12, 6), )

    @unittest.skipIf(not vis, "No Visualisation Required")
    def test_ckls_particular_cases(self):
        print("Test")
        test = CKLSProcess(alpha=0.7, beta=0.5, sigma=0.1, gamma=1.0, initial=1.0, T=1.0)
        print(test.__str__())
        test_bm = CKLSProcess(alpha=0.5, beta=0.0, sigma=0.1, gamma=0.0, initial=1.0, T=1.0)
        print(test_bm.__str__())
        test_vas = CKLSProcess(alpha=0.5, beta=-2.0, sigma=0.1, gamma=0.0, initial=1.0, T=1.0)
        print(test_vas.__str__())
        test_cir = CKLSProcess(alpha=0.5, beta=-2.0, sigma=0.1, gamma=0.5, initial=1.0, T=1.0)
        print(test_cir.__str__())
        test_gbm = CKLSProcess(alpha=0.0, beta=-2.0, sigma=0.1, gamma=1.0, initial=1.0, T=1.0)
        print(test_gbm.__str__())
        test_cev = CKLSProcess(alpha=0.0, beta=-2.0, sigma=0.1, gamma=1.5, initial=1.0, T=1.0)
        print(test_cev.__str__())

    @unittest.skipIf(not vis, "No Visualisation Required")
    def test_cev(self):
        cev = CEVProcess(mu=1.0, sigma=0.1, gamma=0.0, initial=1.0, T=1.0)
        cir = CEVProcess(mu=-1.0, sigma=0.1, gamma=0.5, initial=1.0, T=1.0)
        for process in [cev, cir]:
            print(process.__str__())
            process.draw(n=100, N=100, envelope=True)
