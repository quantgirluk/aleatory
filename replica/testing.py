from gaussian import Gaussian
import matplotlib.pyplot as plt

my_process = Gaussian()
n =200
plt.plot(my_process.times(n=n), my_process.sample(n=n))
plt.show()