from aleatory.processes import BESProcess
import time

def test_multi():

    process = BESProcess(dim=3)
    print('Start')
    t0 = time.time()
    sim1 = process.simulate(n=100, N=5000)
    t1 = time.time()
    print(t1-t0)
    sim2 = process.simulate2(n=100, N=5000)
    t2 = time.time()
    print(t2 - t1)
    sim3 = process.simulate3(n=100, N=5000)
    t3 = time.time()
    print(t3 - t2)
