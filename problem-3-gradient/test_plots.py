# test_plots
import numpy as np
import matplotlib.pyplot as plt

x_values = np.linspace(-1, 1, 101)
print(x_values)

fast_sigmoid = x_values / (1 + np.absolute(x_values))
plt.figure()
plt.plot(x_values, fast_sigmoid)
plt.show()

piecewise_linear = np.zeros(len(x_values))
piecewise_linear[np.where(x_values < -0.5)] = 1
piecewise_linear[np.where((x_values >= -0.5) & (x_values < 0))] = 2
# piecewise_linear[0 <= x_values < 0.5] = -2
plt.figure()
plt.plot(x_values, piecewise_linear)
plt.show()