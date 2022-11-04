from gaussian import Gaussian
import matplotlib.pyplot as plt

my_process = Gaussian()
n =200

print(my_process.sample_at(times=[1,2,3, 4.5]))


plt.plot(my_process.times(n=n), my_process.sample(n=n))
plt.show()

