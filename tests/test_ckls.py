from aleatory.processes import CKLSProcess, CEVProcess


def test_ckls_process():
    process = CKLSProcess(alpha=0.5, beta=0.5, sigma=0.1, gamma=1.0, initial=1.0, T=1.0)
    process.plot(n=100, N=5, figsize=(9.5, 6), dpi=200)
    process.draw(n=100, N=200)
    process.draw(n=100, N=200, figsize=(10, 6), dpi=200, colormap="spring")
    process.draw(n=100, N=200, envelope=False, figsize=(12, 6))
    process.draw(n=100, N=200, envelope=False, orientation='horizontal', figsize=(12, 6))
    process.draw(n=100, N=200, envelope=False, orientation='vertical', figsize=(12, 6), )


def test_ckls_particular_cases():
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


def test_cev():
    cev = CEVProcess(mu=1.0, sigma=0.1, gamma=0.0, initial=1.0, T=1.0)
    cir = CEVProcess(mu=-1.0, sigma=0.1, gamma=0.5, initial=1.0, T=1.0)
    for process in [cev, cir]:
        print(process.__str__())
        process.draw(n=100, N=100, envelope=True)
