import numpy as np

from gaussian import Gaussian
from brownian_motion import BrownianMotion
import matplotlib.pyplot as plt

# my_process = Gaussian()
# n =200
#
# print(my_process.sample_at(times=[1,2,3, 4.5]))
# plt.plot(my_process.times(n=n), my_process.sample(n=n))
# plt.show()

# n = 200
# for k in range(2):
#     my_brownian = BrownianMotion(drift=-1.0, scale=2.0, T=5.0)
#     sample = my_brownian.sample(n=n)
#     print(type(sample))
#     times = my_brownian.times
#     plt.plot(times, sample)
# plt.show()

# my_times = np.linspace(0, 5, 100, endpoint=True)
# for k in range(10):
#     my_brownian = BrownianMotion(drift=-1.0, scale=2.0)
#     sample = my_brownian.sample_at(my_times)
#     plt.plot(my_times, sample)
# plt.show()

from brownian_motion_paths import BrownianPaths

my_times = np.linspace(0, 5, 100, endpoint=True)
paths = BrownianPaths(N=20, times=my_times).draw()

