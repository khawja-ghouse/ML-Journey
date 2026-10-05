"""
STEP-BY-STEP: Straight Line Model + Cost Function + Choosing w, b
------------------------------------------------------------------
We go through this one small step at a time. Only the current step's
code lives in this file - ask for the next step when you're ready.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

plt.style.use(os.path.join(os.path.dirname(__file__), 'deeplearning.mplstyle'))


# ======================================================================
# STEP 1: THE DATA
# ======================================================================
# Known facts: for each house size (x_train), we know the price it sold
# for (y_train). Nothing is "learned" yet - this is just the raw history.
x_train = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])   # size in 1000 sqft
y_train = np.array([300.0, 320.0, 365.0, 390.0, 440.0, 475.0])  # price in 1000s $
m = x_train.shape[0]  # number of training examples

print("=" * 70)
print("STEP 1: THE DATA")
print("=" * 70)
for i in range(m):
    print(f"  house {i}: size = {x_train[i]} (1000 sqft)  ->  price = {y_train[i]} (1000$)")

plt.figure()
plt.scatter(x_train, y_train, marker='x', c='r', label="Actual price")
plt.title("Step 1: Raw training data")
plt.xlabel("Size (1000 sqft)")
plt.ylabel("Price (1000s of dollars)")
plt.legend()
plt.show()


# ======================================================================
# STEP 2: THE MODEL -> A STRAIGHT LINE
# ======================================================================
# f_wb(x) = w * x + b
#   w (weight) -> slope of the line  (how steep)
#   b (bias)   -> where the line crosses the y-axis (starting price)
def predict(x, w, b):
    """Return f_wb(x) = w*x + b for every value in x."""
    f_wb = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        f_wb[i] = w * x[i] + b
    return f_wb


print("\n" + "=" * 70)
print("STEP 2: THE MODEL  f_wb(x) = w*x + b")
print("=" * 70)

# First guess, picked by eye - almost always wrong, that's fine.
w_guess = 150
b_guess = 100
f_wb_guess = predict(x_train, w_guess, b_guess)

print(f"  first guess -> w = {w_guess}, b = {b_guess}")
for i in range(m):
    print(f"  x = {x_train[i]}  ->  predicted = {f_wb_guess[i]:0.1f}   actual = {y_train[i]}")

plt.figure()
plt.plot(x_train, f_wb_guess, c='b', label=f"Our line: w={w_guess}, b={b_guess}")
plt.scatter(x_train, y_train, marker='x', c='r', label="Actual price")
plt.title("Step 2: A first guess at the line")
plt.xlabel("Size (1000 sqft)")
plt.ylabel("Price (1000s of dollars)")
plt.legend()
plt.show()


# ======================================================================
# STEP 3: THE COST FUNCTION -> HOW WRONG IS THE LINE?
# ======================================================================
#            1    m-1
# J(w,b) = ----- * sum ( f_wb(x[i]) - y[i] )^2
#           2*m   i=0
#
# - (f_wb(x[i]) - y[i]) is the error for one point (prediction - actual)
# - we square it so +ve and -ve errors don't cancel out
# - we add up the squared errors for every point
# - we divide by 2*m to get an "average" error (the 2 just keeps the
#   math nice later on, it does not change which w,b is best)
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        error = f_wb - y[i]
        cost_sum += error ** 2
    return cost_sum / (2 * m)


print("\n" + "=" * 70)
print("STEP 3: THE COST FUNCTION J(w, b)")
print("=" * 70)
cost_guess = compute_cost(x_train, y_train, w_guess, b_guess)
print(f"  cost of our first guess (w={w_guess}, b={b_guess}) = {cost_guess:0.2f}")
print("  smaller cost = better fitting line. Now let's search for a better w.")


# ======================================================================
# STEP 4: TRY MANY VALUES OF w (keep b fixed) AND COMPARE COST
# ======================================================================
b_fixed = 100
w_values = np.arange(50, 251, 10)  # try w = 50, 60, 70, ... 250
cost_values = np.zeros(w_values.shape[0])

print("\n" + "=" * 70)
print(f"STEP 4: SWEEP w FROM {w_values[0]} TO {w_values[-1]} (b fixed = {b_fixed})")
print("=" * 70)
for i, w in enumerate(w_values):
    cost_values[i] = compute_cost(x_train, y_train, w, b_fixed)
    print(f"  w = {w:>4}  ->  cost J(w,b) = {cost_values[i]:>10.2f}")

best_index = np.argmin(cost_values)
best_w = w_values[best_index]
print(f"\n  lowest cost found at w = {best_w}  (cost = {cost_values[best_index]:0.2f})")

plt.figure()
plt.plot(w_values, cost_values, c='b')
plt.scatter([best_w], [cost_values[best_index]], c='r', zorder=5,
            label=f"best w = {best_w}")
plt.title(f"Step 4: Cost J(w) vs w   (b fixed = {b_fixed})")
plt.xlabel("w")
plt.ylabel("Cost J(w, b)")
plt.legend()
plt.show()


# ======================================================================
# STEP 5: SHOW THE WINNING LINE AGAINST THE DATA
# ======================================================================
print("\n" + "=" * 70)
print("STEP 5: BEST LINE FOUND BY THE SWEEP")
print("=" * 70)
f_wb_best = predict(x_train, best_w, b_fixed)
print(f"  best line so far -> w = {best_w}, b = {b_fixed}")

plt.figure()
plt.plot(x_train, f_wb_best, c='b', label=f"Best line: w={best_w}, b={b_fixed}")
plt.scatter(x_train, y_train, marker='x', c='r', label="Actual price")
plt.title("Step 5: Line chosen by minimizing cost over w")
plt.xlabel("Size (1000 sqft)")
plt.ylabel("Price (1000s of dollars)")
plt.legend()
plt.show()


# ======================================================================
# STEP 6: b MATTERS TOO - COST DEPENDS ON BOTH w AND b
# ======================================================================
# Up to now we kept b fixed and only searched w. In reality the cost
# changes if we change b as well, so the true "lowest point" lives in
# a 2D surface J(w, b). A grid search makes that visible.
print("\n" + "=" * 70)
print("STEP 6: SWEEP BOTH w AND b TOGETHER")
print("=" * 70)
b_values = np.arange(50, 201, 10)
W, B = np.meshgrid(w_values, b_values)
J = np.zeros(W.shape)
for row in range(W.shape[0]):
    for col in range(W.shape[1]):
        J[row, col] = compute_cost(x_train, y_train, W[row, col], B[row, col])

min_pos = np.unravel_index(np.argmin(J), J.shape)
best_w_2d = W[min_pos]
best_b_2d = B[min_pos]
print(f"  best (w, b) found over the grid -> w = {best_w_2d}, b = {best_b_2d}")
print(f"  cost at that point = {J[min_pos]:0.2f}")
print("  (next lab: gradient descent finds this automatically, without")
print("   having to try every single value by hand)")

plt.figure()
cp = plt.contour(W, B, J, levels=20)
plt.clabel(cp, inline=True, fontsize=8)
plt.scatter([best_w_2d], [best_b_2d], c='r', marker='x', s=100,
            label=f"best (w={best_w_2d}, b={best_b_2d})")
plt.title("Step 6: Cost J(w, b) contour - lower rings = lower cost")
plt.xlabel("w")
plt.ylabel("b")
plt.legend()
plt.show()
