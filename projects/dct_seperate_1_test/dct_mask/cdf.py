import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 8), dpi=80)
x = np.arange(0,1,0.01)
y = 8*(x-x**2)
y2 = 16*(x-x**2)**2
plt.plot(x, y, color='r')
plt.plot(x, y2, color='b')
plt.show()


