import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(-2, 2, 500)
y = x**2
print(x)
print(y)

plt.plot(x, y)
plt.title("Square function")
plt.xlabel("x")
plt.ylabel("y = x**2")
plt.grid(True)
plt.show()