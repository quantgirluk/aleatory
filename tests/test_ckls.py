from aleatory.processes import CKLSProcess

def test_ckls_process():
    process = CKLSProcess()
    process.plot(n=100, N=5, figsize=(9.5, 6), dpi=200)
    process.draw(n=100, N=200)
    process.draw(n=100, N=200, figsize=(10, 6), dpi=200, colormap="spring")
    process.draw(n=100, N=200, envelope=False, figsize=(12, 6))
    process.draw(n=100, N=200, envelope=False, orientation='horizontal', figsize=(12, 6))
    process.draw(n=100, N=200, envelope=False, orientation='vertical', figsize=(12, 6), )

