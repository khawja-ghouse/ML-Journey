import matplotlib.pyplot as plt

# 1D array value of verticle axis
# plt.plot([1, 2, 4, 9, 5, 3])
# plt.show()

# 2D aray both Verticle and Horizontal axis
# plt.plot([-3, -2, 5, 0], [1, 6, 4, 3])
# plt.show()

# With Controlled axis
plt.plot([-3, -2, 5, 0], [1, 6, 4, 3])
plt.axis([-1, 6, 0, 7])
plt.show()

