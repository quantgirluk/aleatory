from aleatory.processes import CIRProcess, CEVProcess


def test_transformation_em():
    cir = CIRProcess(theta=1.0, mu=2.0, sigma=0.25)
    cev = CEVProcess(gamma=1.5, mu=0.50, sigma=0.1, initial=1.0)

    cir.draw(n=100, N=200)
    cev.draw(n=100, N=200)
