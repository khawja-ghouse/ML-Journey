"""
STEP-BY-STEP: Multiple Variable Linear Regression
------------------------------------------------------------------
We go through this one small step at a time. Only the current step's
code lives in this file - ask for the next step when you're ready.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)  # reduced display precision on numpy arrays
_style_path = os.path.join(os.path.dirname(__file__), '..', 'C1_W1_Lab03_Cost_function', 'deeplearning.mplstyle')
plt.style.use(_style_path)


# ======================================================================
# STEP 1: THE DATA -> NOW WITH MULTIPLE FEATURES
# ======================================================================
# Before, x_train was one list of house sizes (ONE feature per house).
# Now each house has FOUR features: size, bedrooms, floors, age.
# So X_train is a MATRIX - one row per house, one column per feature.
X_train = np.array([[2104, 5, 1, 45],
                     [1416, 3, 2, 40],
                     [852,  2, 1, 35]])
y_train = np.array([460, 232, 178])

m, n = X_train.shape  # m = number of houses (examples), n = number of features

print("=" * 70)
print("STEP 1: THE DATA (multiple features)")
print("=" * 70)
print(f"X_train shape: {X_train.shape}  ->  m = {m} examples, n = {n} features")
print(f"y_train shape: {y_train.shape}")
print()

feature_names = ["size(sqft)", "bedrooms", "floors", "age(yrs)"]
for i in range(m):
    row = "  ".join(f"{name}={X_train[i, j]:<5}" for j, name in enumerate(feature_names))
    print(f"  house {i}: {row}  ->  price = {y_train[i]} (1000$)")

print()
print(f"one full example (a row):      X_train[0] = {X_train[0]}")
print(f"one single feature (a column): X_train[:, 0] = {X_train[:, 0]}  (every house's size)")


# ======================================================================
# STEP 2: THE MODEL -> f(x) = w.x + b, NOW WITH A VECTOR OF WEIGHTS
# ======================================================================
# Before: f(x) = w*x + b, where w was ONE number (one slope).
# Now: f(x) = w0*x0 + w1*x1 + w2*x2 + w3*x3 + b, one weight PER FEATURE.
# In vector notation that sum is just a DOT PRODUCT:  f(x) = w . x + b
#
# w and b below are not random - they're close to the "best fit" values
# for this tiny dataset, just so the example predictions look reasonable.
b_init = 785.1811367994083
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])


def predict_single_loop(x, w, b):
    """f(x) = w0*x0 + w1*x1 + ... + wn-1*xn-1 + b, done one feature at a time."""
    n = x.shape[0]
    p = 0
    for i in range(n):
        p_i = x[i] * w[i]
        p = p + p_i
    p = p + b
    return p


def predict(x, w, b):
    """Same formula, but np.dot does the multiply-and-add-them-all-up in one call."""
    return np.dot(x, w) + b


print("\n" + "=" * 70)
print("STEP 2: THE MODEL  f(x) = w.x + b")
print("=" * 70)
x_vec = X_train[0, :]  # house 0's four features, as a vector
print(f"x_vec (house 0's features): {x_vec}")

pred_loop = predict_single_loop(x_vec, w_init, b_init)
pred_dot = predict(x_vec, w_init, b_init)
print(f"prediction (loop version):   {pred_loop:0.2f}")
print(f"prediction (np.dot version): {pred_dot:0.2f}")
print(f"actual price for house 0:    {y_train[0]}")
print("(both versions agree - np.dot is just a faster way to write the same sum)")

print("\n" + "-" * 70)
print("SAME w, DIFFERENT ROWS OF X -> each house gets its own term-by-term math")
print("-" * 70)
print(f"w (shared by every house): {w_init}")
print(f"b (shared by every house): {b_init:0.2f}")
print()
for i in range(m):
    x_i = X_train[i, :]
    terms = " + ".join(f"({w_init[j]:0.2f}*{x_i[j]})" for j in range(n))
    print(f"house {i}: x = {x_i}")
    print(f"  f(x) = {terms} + {b_init:0.2f}")
    print(f"       = {predict(x_i, w_init, b_init):0.2f}   (actual price: {y_train[i]})")
    print()


# ======================================================================
# STEP 3: THE COST FUNCTION -> SAME FORMULA, VECTOR PREDICTIONS INSIDE
# ======================================================================
#            1    m-1
# J(w,b) = ----- * sum ( f_w,b(X[i]) - y[i] )^2
#           2*m   i=0
#
# This is IDENTICAL to the single-feature cost function from the earlier
# lab. The only change: f_wb is now np.dot(X[i], w) + b instead of
# w*x[i] + b, because X[i] and w are vectors now, not single numbers.
def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b       # prediction for house i (a scalar)
        cost = cost + (f_wb_i - y[i]) ** 2  # squared error for house i
    cost = cost / (2 * m)
    return cost


print("\n" + "=" * 70)
print("STEP 3: THE COST FUNCTION J(w, b)")
print("=" * 70)
cost = compute_cost(X_train, y_train, w_init, b_init)
print(f"w = {w_init}")
print(f"b = {b_init:0.2f}")
print(f"cost at this (w, b) = {cost:0.2e}")
print()
print("This w, b pair was chosen to be near-optimal for this data, which")
print("is why the cost is tiny (close to 0) rather than some huge number.")


# ======================================================================
# STEP 4: THE GRADIENT -> ONE PARTIAL DERIVATIVE PER WEIGHT, PLUS dJ/db
# ======================================================================
# Before: ONE dj_dw (a single number), because there was only one w.
# Now: dj_dw is a VECTOR with one entry per feature - dj_dw[j] tells us
# the slope of the cost w.r.t. w[j] specifically, holding everything
# else fixed. dj_db stays a single number (there's still only one b).
def compute_gradient(X, y, w, b):
    m, n = X.shape           # m examples, n features
    dj_dw = np.zeros((n,))   # one slot per feature's weight
    dj_db = 0.0

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]   # prediction error for house i
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]   # feature j's contribution
        dj_db = dj_db + err

    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_db, dj_dw


print("\n" + "=" * 70)
print("STEP 4: THE GRADIENT (one slope per weight)")
print("=" * 70)
dj_db, dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
print(f"dj_db at this (w, b): {dj_db:0.2e}")
print(f"dj_dw at this (w, b): {dj_dw}")
print()
print("Each dj_dw[j] says which way to nudge w[j]:")
feature_names = ["size(sqft)", "bedrooms", "floors", "age(yrs)"]
for j in range(n):
    direction = "decrease" if dj_dw[j] > 0 else "increase"
    print(f"  w[{j}] ({feature_names[j]:<11}): dj_dw = {dj_dw[j]:>12.2e}  -> should {direction}")


# ======================================================================
# STEP 5: GRADIENT DESCENT -> REPEAT THE DOWNHILL STEP, FOR EVERY WEIGHT
# ======================================================================
# Same loop structure as the single-feature lab's gradient_descent.
# The only change: w is now a vector, so "w = w - alpha * dj_dw" updates
# ALL 4 weights at once (NumPy applies it element-by-element automatically).
import copy
import math


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_in)  # avoid modifying the caller's original array
    b = b_in

    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b)

        # move opposite to the gradient - same update rule as before,
        # just w and dj_dw are vectors now instead of single numbers
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:  # prevent resource exhaustion on huge num_iters
            J_history.append(cost_function(X, y, w, b))

        if i % math.ceil(num_iters / 10) == 0:
            print(f"  iteration {i:4}: cost {J_history[-1]:8.2f}")

    return w, b, J_history


print("\n" + "=" * 70)
print("STEP 5: RUN GRADIENT DESCENT (starting from scratch this time)")
print("=" * 70)
initial_w = np.zeros_like(w_init)
initial_b = 0.0
iterations = 1000
alpha = 5.0e-7

w_final, b_final, J_hist = gradient_descent(
    X_train, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)

print(f"\n  b, w found by gradient descent: {b_final:0.2f}, {w_final}")
print()
for i in range(m):
    pred = np.dot(X_train[i], w_final) + b_final
    print(f"  house {i}: prediction = {pred:0.2f}, actual = {y_train[i]}")
print()
print("Predictions are in the right ballpark but not great - cost is still")
print("falling slowly after 1000 iterations. The next lab (feature scaling)")
print("explains why multi-feature data makes gradient descent converge")
print("much more slowly, and how to fix it.")

# Same two-panel trick from the single-feature lab: the full curve makes
# the early rapid drop visible, the "tail" zooms into the slow part after.
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration")
ax2.set_title("Cost vs. iteration (tail)")
ax1.set_xlabel("iteration step")
ax2.set_xlabel("iteration step")
ax1.set_ylabel("Cost")
ax2.set_ylabel("Cost")
plt.show()
