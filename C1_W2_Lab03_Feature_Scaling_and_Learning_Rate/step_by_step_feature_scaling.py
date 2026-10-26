"""
STEP-BY-STEP: Feature Scaling and Learning Rate (Multiple Variables)
------------------------------------------------------------------
We go through this one small step at a time. Only the current step's
code lives in this file - ask for the next step when you're ready.
"""

import os
import copy
import math
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)
_style_path = os.path.join(os.path.dirname(__file__), '..', 'C1_W1_Lab03_Cost_function', 'deeplearning.mplstyle')
plt.style.use(_style_path)
_data_path = os.path.join(os.path.dirname(__file__), 'data', 'houses.txt')


# ======================================================================
# STEP 1: THE DATA -> A BIGGER, REAL-WORLD DATASET
# ======================================================================
# Same 4 features as the previous lab (size, bedrooms, floors, age), but
# now 100 houses instead of 3. More examples means the scale mismatch
# between features (hundreds/thousands for size vs. single digits for
# bedrooms/floors) really starts to hurt gradient descent.
data = np.loadtxt(_data_path, delimiter=',')
X_train = data[:, :4]
y_train = data[:, 4]
m, n = X_train.shape

feature_names = ["size(sqft)", "bedrooms", "floors", "age(yrs)"]

print("=" * 70)
print("STEP 1: THE DATA (100 houses, 4 features)")
print("=" * 70)
print(f"X_train shape: {X_train.shape}  ->  m = {m} examples, n = {n} features")
print()
print("Range of each feature (this is the whole problem in one table):")
for j in range(n):
    col = X_train[:, j]
    print(f"  {feature_names[j]:<12}: min={col.min():>8.1f}  max={col.max():>8.1f}  "
          f"peak-to-peak={np.ptp(col):>8.1f}")
print()
print("'size(sqft)' spans thousands while 'floors' only spans 1-2.")
print("Gradient descent uses ONE shared alpha for every w_j - that's about")
print("to cause trouble.")

fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:, i], y_train)
    ax[i].set_xlabel(feature_names[i])
ax[0].set_ylabel("Price (1000's)")
fig.suptitle("Step 1: each feature vs. price")
plt.show()


# ======================================================================
# STEP 2: RECAP -> MODEL, COST, GRADIENT (identical to the previous lab)
# ======================================================================
# Nothing new here. Same f(x) = w.x + b, same squared-error cost, same
# per-feature gradient. The only thing that changed is the data going in.
def predict(x, w, b):
    return np.dot(x, w) + b


def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost = cost + (f_wb_i - y[i]) ** 2
    cost = cost / (2 * m)
    return cost


def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.0
    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_db, dj_dw


print("\n" + "=" * 70)
print("STEP 2: RECAP - MODEL, COST, GRADIENT")
print("=" * 70)
zero_w = np.zeros(n)
print(f"cost at w=0, b=0 on this bigger dataset: {compute_cost(X_train, y_train, zero_w, 0.0):0.2e}")
print("(same formulas as before - just confirming they still work on 100 rows)")

_, dj_dw_zero = compute_gradient(X_train, y_train, zero_w, 0.0)
print(f"dj_dw at w=0, b=0: {dj_dw_zero}")
for j in range(n):
    print(f"  dj_dw[{j}] ({feature_names[j]:<12}): {dj_dw_zero[j]: 0.3e}")
print("Notice dj_dw[0] (size) is already orders of magnitude bigger than")
print("the others - that mismatch is what the rest of this lab is about.")


# ======================================================================
# STEP 3: GRADIENT DESCENT -> NOW WITH A COST/PARAMETER HISTORY FOR PLOTS
# ======================================================================
# Same update rule as before. We additionally record w[0] at every step so
# we can later plot "cost vs w[0]" the way the lecture does, holding the
# other 3 weights fixed at near-optimal reference values.
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    hist = {"iter": [], "cost": [], "w0": [], "w": []}
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b)
        w = w - alpha * dj_dw   # updates w[0], w[1], w[2], w[3] simultaneously (elementwise)
        b = b - alpha * dj_db

        hist["iter"].append(i)
        hist["cost"].append(cost_function(X, y, w, b))
        hist["w0"].append(w[0])
        hist["w"].append(w.copy())

        print(f"  iteration {i:4}: cost {hist['cost'][-1]:0.5e}  "
              f"w = {w}  b = {b: 0.2e}")

    return w, b, hist


print("\n" + "=" * 70)
print("STEP 3: TRY GRADIENT DESCENT (alpha = 1e-7, 10 iterations)")
print("=" * 70)
w_final, b_final, hist = gradient_descent(
    X_train, y_train, np.zeros(n), 0.0, compute_cost, compute_gradient, alpha=1e-7, num_iters=10)
print(f"\nfinal w = {w_final}")
print(f"final b = {b_final:0.2f}")
print("Cost is dropping, but slowly - next we'll see what happens with")
print("different choices of alpha.")


# ======================================================================
# STEP 4: DIAGNOSTIC PLOT -> COST VS. ITERATION, COST VS. w[0]
# ======================================================================
# This is the check: plot cost after every iteration. If it's steadily
# falling, alpha is fine (maybe just slow). If it's jumping up and down
# or climbing, alpha is too big and will diverge. The "cost vs w[0]"
# panel holds w[1..3], b fixed at near-optimal reference values and
# sweeps w[0] alone, so you can see WHERE on the cost curve each
# iteration landed (undershooting, overshooting, or smoothly descending).
W_REF = np.array([0.2619, -32.62, -67.25, -1.47])  # near-optimal reference weights, from a long converged run
B_REF = 220.0


def plot_cost_and_w0(hist, title):
    ws = np.array(hist["w0"])
    rng = max(abs(ws.min()), abs(ws.max()), 1e-6)
    wr = np.linspace(-rng * 1.5, rng * 1.5, 40)
    cst = [compute_cost(X_train, y_train, np.array([wr[i], *W_REF[1:]]), B_REF) for i in range(len(wr))]

    fig, ax = plt.subplots(1, 2, figsize=(12, 3))
    ax[0].plot(hist["iter"], hist["cost"])
    ax[0].set_title("Cost vs. iteration")
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("Cost")
    ax[1].plot(wr, cst)
    ax[1].plot(ws, hist["cost"])
    ax[1].set_title("Cost vs. w[0]")
    ax[1].set_xlabel("w[0]")
    ax[1].set_ylabel("Cost")
    fig.suptitle(title)
    plt.show()


print("\n" + "=" * 70)
print("STEP 4: PLOT COST VS. ITERATION FOR THE STEP 3 RUN")
print("=" * 70)
print("Left panel: cost over iterations - falling steadily means alpha is")
print("safe for now. Right panel: where w[0] actually sits on the cost")
print("curve - still far from the minimum means progress is just slow.")
plot_cost_and_w0(hist, "Step 4: alpha = 1e-7")


# ======================================================================
# STEP 5: USE THE DIAGNOSTIC PLOT TO COMPARE LEARNING RATES
# ======================================================================
# Same gradient_descent + plot_cost_and_w0, just a bigger alpha. Watch
# the left panel: cost should climb (diverge), not fall.
print("\n" + "=" * 70)
print("STEP 5: alpha = 9.9e-7 (too large)")
print("=" * 70)
_, _, hist_high = gradient_descent(
    X_train, y_train, np.zeros(n), 0.0, compute_cost, compute_gradient, alpha=9.9e-7, num_iters=10)
print("\nCost is INCREASING - w[0] overshoots the minimum every step and")
print("the step gets bigger each time instead of smaller. This diverges.")
plot_cost_and_w0(hist_high, "Step 5: alpha = 9.9e-7 (diverges)")

print("\n" + "=" * 70)
print("STEP 5b: alpha = 9e-7 (converges, but oscillates)")
print("=" * 70)
_, _, hist_mid = gradient_descent(
    X_train, y_train, np.zeros(n), 0.0, compute_cost, compute_gradient, alpha=9e-7, num_iters=10)
print("\nCost decreases every iteration now, but w[0] is still jumping back")
print("and forth across the minimum (dj_dw[0] keeps flipping sign).")
plot_cost_and_w0(hist_mid, "Step 5b: alpha = 9e-7 (converges, oscillates)")


# ======================================================================
# STEP 6: PROOF -> ALL 4 WEIGHTS MOVE TOGETHER, EVERY ITERATION
# ======================================================================
# plot_cost_and_w0 only ever drew w[0] because that's all a 2D plot can
# show. This step plots EVERY weight's own trajectory, side by side, so
# it's obvious w[1], w[2], w[3] were being updated the whole time too -
# they just move far less per step than w[0] does.
def plot_all_weights(hist, title):
    w_hist = np.array(hist["w"])  # shape (num_iters, n)
    fig, ax = plt.subplots(1, w_hist.shape[1], figsize=(12, 3), sharex=True)
    for j in range(w_hist.shape[1]):
        ax[j].plot(hist["iter"], w_hist[:, j])
        ax[j].set_title(f"w[{j}] ({feature_names[j]})")
        ax[j].set_xlabel("iteration")
    fig.suptitle(title)
    plt.show()


print("\n" + "=" * 70)
print("STEP 6: PLOT ALL 4 WEIGHTS' TRAJECTORIES (alpha = 9e-7 run)")
print("=" * 70)
print("Every panel is updating on every iteration - w[0] swings wildly")
print("(zig-zagging through the minimum) while w[1..3] barely creep,")
print("because their gradients are so much smaller than w[0]'s.")
plot_all_weights(hist_mid, "Step 6: all weights vs. iteration (alpha = 9e-7)")


# ======================================================================
# STEP 7: WHY? -> ONE SHARED alpha, BUT WILDLY DIFFERENT FEATURE SCALES
# ======================================================================
# dj_dw[j] = (1/m) * sum((f_wb(x_i) - y_i) * x_i[j])
# The SAME error term gets multiplied by x_i[j]. Since 'size(sqft)' is
# ~1000x bigger than 'floors', dj_dw[0] is naturally ~1000x bigger than
# dj_dw[2] too - so the SAME alpha moves w[0] much further per step than
# it moves w[2], w[3]. Pick alpha small enough for w[0] to behave, and
# the other weights crawl. Pick it big enough for the others, and w[0]
# diverges. Feature scaling removes this mismatch entirely.
print("\n" + "=" * 70)
print("STEP 7: WHY DOES THIS HAPPEN?")
print("=" * 70)
print("Every w_j shares the same alpha, but dj_dw[j] is scaled by x[:,j].")
print("Compare the gradient magnitudes at the starting point w=0, b=0:")
_, dj_dw_start = compute_gradient(X_train, y_train, np.zeros(n), 0.0)
for j in range(n):
    print(f"  dj_dw[{j}] ({feature_names[j]:<12}): {dj_dw_start[j]: 0.3e}")
print("size(sqft)'s gradient dwarfs the others - one alpha can't fit both:")
print("  - big enough for w[1..3] to move at a useful pace -> w[0] diverges")
print("  - small enough for w[0] to behave -> w[1..3] barely move")
print("The fix isn't a better alpha - it's putting every feature on the")
print("same scale first, so one alpha is fair to all of them.")


# ======================================================================
# STEP 8: THE FIX -> Z-SCORE NORMALIZATION
# ======================================================================
# x_norm[j] = (x[j] - mean_j) / std_j
# After this, every feature has mean 0 and std 1 - same scale, same
# "weight" in the gradient, regardless of the original units.
def zscore_normalize_features(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


print("\n" + "=" * 70)
print("STEP 8: Z-SCORE NORMALIZATION")
print("=" * 70)
X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)
print(f"X_mu    = {X_mu}")
print(f"X_sigma = {X_sigma}")
print()
print(f"peak-to-peak BEFORE normalizing: {np.ptp(X_train, axis=0)}")
print(f"peak-to-peak AFTER  normalizing: {np.ptp(X_norm, axis=0)}")
print("Every feature is now squeezed into roughly the same +/- few range.")

fig, ax = plt.subplots(1, 4, figsize=(12, 3))
for i in range(len(ax)):
    ax[i].hist(X_train[:, i], bins=20)
    ax[i].set_xlabel(feature_names[i])
ax[0].set_ylabel("count")
fig.suptitle("Step 8: feature distributions BEFORE normalization")
plt.show()

fig, ax = plt.subplots(1, 4, figsize=(12, 3))
for i in range(len(ax)):
    ax[i].hist(X_norm[:, i], bins=20)
    ax[i].set_xlabel(feature_names[i])
ax[0].set_ylabel("count")
fig.suptitle("Step 8: feature distributions AFTER normalization")
plt.show()


# ======================================================================
# STEP 9: GRADIENT DESCENT ON NORMALIZED DATA -> MUCH BIGGER alpha WORKS
# ======================================================================
# Because every feature is on the same scale now, one alpha treats every
# w_j fairly. alpha = 0.1 would have diverged instantly on raw data - on
# normalized data it converges fast and smoothly.
print("\n" + "=" * 70)
print("STEP 9: GRADIENT DESCENT ON NORMALIZED FEATURES (alpha = 0.1)")
print("=" * 70)
w_norm, b_norm, hist_norm = gradient_descent(
    X_norm, y_train, np.zeros(n), 0.0, compute_cost, compute_gradient, alpha=1.0e-1, num_iters=100)
print(f"\nfinal w = {w_norm}")
print(f"final b = {b_norm:0.2f}")
print(f"final cost = {hist_norm['cost'][-1]:0.2f}  (compare to {hist['cost'][-1]:0.2f} "
      f"after 10 tiny unnormalized steps in Step 3)")

plt.figure()
plt.plot(hist_norm["iter"], hist_norm["cost"])
plt.title("Step 9: cost vs. iteration (normalized features, alpha=0.1)")
plt.xlabel("iteration")
plt.ylabel("Cost")
plt.show()

plot_all_weights(hist_norm, "Step 9: all weights vs. iteration (normalized, alpha=0.1)")

# predictions use the NORMALIZED features - plotted against the original
# (human-readable) feature values.
yp = X_norm @ w_norm + b_norm
fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:, i], y_train, label="target")
    ax[i].scatter(X_train[:, i], yp, color="orange", label="predict")
    ax[i].set_xlabel(feature_names[i])
ax[0].set_ylabel("Price")
ax[0].legend()
fig.suptitle("Step 9: target vs. prediction (z-score normalized model)")
plt.show()


# ======================================================================
# STEP 10: PREDICTING A NEW HOUSE -> NORMALIZE THE INPUT THE SAME WAY
# ======================================================================
# The model only understands NORMALIZED features. Any new x must be
# normalized with the SAME mu/sigma that came from the training set -
# never recompute mu/sigma from the new point alone.
print("\n" + "=" * 70)
print("STEP 10: PREDICT A NEW HOUSE (1200 sqft, 3 bed, 1 floor, 40 yrs)")
print("=" * 70)
x_house = np.array([1200, 3, 1, 40])
x_house_norm = (x_house - X_mu) / X_sigma
print(f"x_house (raw):        {x_house}")
print(f"x_house (normalized): {x_house_norm}")

x_house_predict = np.dot(x_house_norm, w_norm) + b_norm
print(f"\npredicted price = ${x_house_predict * 1000:0.0f}")
print()
print("Takeaways:")
print("  - one shared alpha + mismatched feature scales -> some weights")
print("    overshoot while others crawl, no single alpha works well")
print("  - z-score normalization puts every feature on the same footing,")
print("    letting gradient descent use a much larger, much faster alpha")
print("  - always normalize NEW inputs with the TRAINING set's mu/sigma")
