"""
STEP-BY-STEP: Gradient Descent for Linear Regression
------------------------------------------------------------------
We go through this one small step at a time. Only the current step's
code lives in this file - ask for the next step when you're ready.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

_style_path = os.path.join(os.path.dirname(__file__), '..', 'C1_W1_Lab03_Cost_function', 'deeplearning.mplstyle')
plt.style.use(_style_path)


# ======================================================================
# STEP 1: RECAP - WHY DO WE NEED GRADIENT DESCENT?
# ======================================================================
# Same data and cost function as the previous lab.
x_train = np.array([1.0, 2.0])        # size in 1000 sqft
y_train = np.array([300.0, 500.0])    # price in 1000s $
m = x_train.shape[0]


def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i]) ** 2
    return cost / (2 * m)


print("=" * 70)
print("STEP 1: WHY DO WE NEED GRADIENT DESCENT?")
print("=" * 70)
print("Last lab we found a good (w, b) by sweeping a whole grid of values")
print("and picking whichever combo had the lowest cost. That worked, but:")
print("  - we had to guess a w range AND a b range ahead of time")
print("  - the grid only had a handful of values, so the answer is approximate")
print("  - with more features (not just house size), the grid explodes in size")
print()

w_test, b_test = 100, 100
print(f"example: cost at a random guess (w={w_test}, b={b_test}) = "
      f"{compute_cost(x_train, y_train, w_test, b_test):0.2f}")
print()
print("Gradient descent finds the lowest point WITHOUT checking a grid.")
print("Instead, at any (w, b) it asks 'which direction makes cost go down?'")
print("and takes a small step that way, repeating until it stops improving.")


# ======================================================================
# STEP 2: THE GRADIENT -> WHICH WAY IS "DOWNHILL"?
# ======================================================================
# dJ/dw and dJ/db are the partial derivatives of the cost - the slope
# of the cost "bowl" at a particular (w, b).
#   - positive dJ/dw -> cost goes UP as w increases   -> decrease w
#   - negative dJ/dw -> cost goes DOWN as w increases  -> increase w
#   - dJ/dw == 0     -> flat, this is the bottom of the bowl (the minimum)
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db


print("\n" + "=" * 70)
print("STEP 2: THE GRADIENT (slope of the cost bowl)")
print("=" * 70)
b_fixed = 100
test_ws = [100, 200, 300]
for w in test_ws:
    dj_dw, dj_db = compute_gradient(x_train, y_train, w, b_fixed)
    if dj_dw > 0:
        direction = "slope is positive -> should DECREASE w"
    elif dj_dw < 0:
        direction = "slope is negative -> should INCREASE w"
    else:
        direction = "slope is zero -> already at the minimum"
    print(f"  at w = {w:>4} (b={b_fixed}) -> dJ/dw = {dj_dw:>10.2f}   {direction}")

plt.figure()
w_range = np.arange(50, 351, 2)
cost_range = np.array([compute_cost(x_train, y_train, w, b_fixed) for w in w_range])
plt.plot(w_range, cost_range, c='b', label="J(w)  (b fixed = 100)")
for w in test_ws:
    cost_here = compute_cost(x_train, y_train, w, b_fixed)
    dj_dw, _ = compute_gradient(x_train, y_train, w, b_fixed)
    plt.scatter([w], [cost_here], c='r', zorder=5)
    # arrow points in the direction that REDUCES cost (opposite sign of dj_dw)
    plt.annotate("", xy=(w - np.sign(dj_dw) * 25, cost_here), xytext=(w, cost_here),
                 arrowprops=dict(arrowstyle="->", color='black', lw=2))
plt.title("Step 2: Cost curve J(w) - arrows show the downhill direction")
plt.xlabel("w")
plt.ylabel("Cost J(w, b)")
plt.legend()
plt.show()


# ======================================================================
# STEP 3: GRADIENT DESCENT -> REPEAT THE DOWNHILL STEP UNTIL IT STOPS HELPING
# ======================================================================
# Step 2 gave us ONE direction at ONE point. Gradient descent just
# repeats that over and over, each time moving a little further downhill:
#     w = w - alpha * dJ/dw
#     b = b - alpha * dJ/db
# alpha ("learning rate") controls how big each step is.
import math


def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # move opposite to the gradient - that's the "downhill" direction
        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        if i < 100000:  # prevent resource exhaustion on huge num_iters
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])

        if i % math.ceil(num_iters / 10) == 0:
            print(f"  iteration {i:4}: cost {J_history[-1]:0.2e}  "
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  "
                  f"w: {w: 0.3e}, b: {b: 0.3e}")

    return w, b, J_history, p_history


print("\n" + "=" * 70)
print("STEP 3: RUN GRADIENT DESCENT")
print("=" * 70)
w_init = 0
b_init = 0
alpha = 1.0e-2
iterations = 10000

w_final, b_final, J_hist, p_hist = gradient_descent(
    x_train, y_train, w_init, b_init, alpha, iterations, compute_cost, compute_gradient)

print(f"\n  (w, b) found by gradient descent: ({w_final:0.4f}, {b_final:0.4f})")
print("  compare to the grid-search answer from the last lab: (w=200, b=100)")

# A single linear-scale plot hides the shape: cost crashes from ~79,300 to
# under 1 within the first ~200 iterations, so the remaining 9,800 look
# like a flat line at zero even though they're still (slowly) improving.
# Two zoomed views - one on the fast start, one on the slow end - fix that.
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration (start)")
ax2.set_title("Cost vs. iteration (end)")
ax1.set_xlabel("iteration step")
ax2.set_xlabel("iteration step")
ax1.set_ylabel("Cost J(w, b)")
ax2.set_ylabel("Cost J(w, b)")
plt.show()
