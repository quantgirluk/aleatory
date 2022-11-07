from gaussian import Gaussian
from brownian_motion import BrownianMotion
import matplotlib.pyplot as plt

my_process = Gaussian()
n =200

print(my_process.sample_at(times=[1,2,3, 4.5]))
plt.plot(my_process.times(n=n), my_process.sample(n=n))
plt.show()


n = 200
my_brownian = BrownianMotion()
sample = my_brownian.sample(n=n)
times = my_brownian._times
# print(times)
# print(sample)

plt.plot(times, sample)
plt.show()