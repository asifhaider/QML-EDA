"""
models/mlp.py — the "pooled-8" MLP classical baseline.

WHY THIS ARCHITECTURE (not a naive flatten-and-MLP)
-----------------------------------------------------
The challenge rule is: same input features as the QML model, trainable
parameters <= QML's count. The QML headline model (M3, L=1) has 38 trainable
parameters and consumes a 3x3x4 patch (36 numbers: 4 features -- Physical-4 or
PCA-4 -- per pixel, 9 pixels).

A naive "flatten the 36 inputs into a small MLP" does NOT fit this budget with
any real nonlinearity:
  - even ONE hidden unit costs 36 weights + 1 bias = 37 params, leaving ~1
    param for everything after it -> hidden width is forced to 1.
  - a single hidden unit is a mathematical dead end: the network reduces to
    sigmoid(a * f(w.x) + b) for a monotonic activation f. A composition of
    monotonic functions of ONE scalar (w.x) is itself monotonic in that
    scalar, so its pixel RANKING is identical to plain logistic regression on
    w.x. AP and ROC-AUC are rank-based -> identical. F1 (after threshold
    sweep) is invariant to monotone rescaling of scores for the same reason
    (see the QML repo's own train/inference.py docstring, which makes exactly
    this point). So a literal 36->1->1 MLP is NOT a different model class
    from logistic regression at this budget -- it would be a wasted baseline.
  - 2 hidden units would already cost >=72 params in layer 1 alone -- over
    budget before even reaching the output layer.

Resolution: pool the 3x3x4 patch DOWN to 8 numbers with a FIXED (non-trainable)
transform before the MLP:
    x_pool = [ center_pixel (4,), mean_of_8_neighbours (4,) ]
This is not new information smuggled in -- it is the same 4 features (physical
or PCA) the QML circuit sees, aggregated the same way the EDA itself already
used and validated (mean-pooling over a spatial window, see
QML-Binary-Segmentation/README.md Part 2 "spatial sweep"). With an 8-dim input,
a hidden layer of width >= 2 becomes affordable within the 38-param budget,
which removes the monotonic-collapse degeneracy: the network can combine two
genuinely different linear projections of x through a nonlinearity before
recombining them, which a single scalar composition cannot express.

PARAMETER BUDGET (hidden=3, the default)
-----------------------------------------
  layer 1 (8 -> 3):  W1 (8,3) = 24 weights + b1 (3,) = 3 biases  -> 27
  layer 2 (3 -> 1):  W2 (3,)  =  3 weights + b2 ()   = 1 bias    ->  4
  total                                                          -> 31
31 <= 38 (M3's count), with headroom deliberately left unused in favour of a
standard architecture (both layers biased) rather than squeezing to exactly 38
by dropping a bias — see README.md for the alternative (hidden=4, no hidden
bias, 37 params) if the full budget should be used later.

OUTPUT CONTRACT
----------------
This model predicts ONE probability per patch: the label of the CENTRE pixel.
That matches the QML ladder's "center_mean" readout mode (see
QML-Binary-Segmentation/models/qml.py's ModelSpec.readout) and lets us reuse
the QML repo's `predict_city_center` / `predict_coordinates` /
`evaluate_predictions` inference machinery UNCHANGED for full-city prediction
and metrics — no overlap-averaging is needed because each patch already yields
exactly one (row, col) prediction (its own centre), unlike the dense 3x3->3x3
models.

    forward(params, X, S) -> P
      X : (B,3,3,4)  same features as QML (physical [0,1] or pca [-1,1])
      S : (B,3,3) or (B,3,3,2)  QML "change strength" side-channel — accepted
          and IGNORED, kept only so this model is a drop-in for the QML repo's
          forward(params, X, S) calling convention.
      P : (B,)   P(centre pixel is 'change')

No autodiff framework is used (no PennyLane, no autograd/JAX/torch) — the
network is a 2-layer MLP, small enough that the exact backprop gradient is a
few lines (below), and this keeps the classical baseline framework-independent
of the QML side's runtime. The gradient is verified against a numerical
(finite-difference) gradient in the smoke test at the bottom of this file.
"""
import numpy as np

INPUT_DIM = 8       # [center(4), neighbour_mean(4)]
HIDDEN = 3          # default hidden width -> 31 params (see module docstring)


def param_count(hidden=HIDDEN):
    """9*hidden (layer1: 8*hidden weights + hidden biases) + hidden (layer2
    weights) + 1 (layer2 bias) = 10*hidden + 1."""
    return 10 * hidden + 1


def init_params(seed=0, hidden=HIDDEN):
    """Flat trainable vector: [W1.ravel()(8*hidden), b1(hidden), W2(hidden), b2(1)].
    W1 uses Glorot-uniform-like scaling (suits the tanh hidden activation);
    W2, b1, b2 start at/near zero so the network starts close to predicting a
    constant (calibrated during training by b2), a standard, stable MLP init."""
    rng = np.random.RandomState(seed)
    limit = np.sqrt(6.0 / (INPUT_DIM + hidden))          # Glorot uniform bound
    W1 = rng.uniform(-limit, limit, size=(INPUT_DIM, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.uniform(-limit, limit, size=(hidden,)) * 0.1  # small: output layer
    b2 = np.zeros(1)
    v = np.concatenate([W1.ravel(), b1, W2, b2])
    assert v.size == param_count(hidden)
    return v


def _unpack(params, hidden=HIDDEN):
    i = 0
    W1 = params[i:i + INPUT_DIM * hidden].reshape(INPUT_DIM, hidden); i += INPUT_DIM * hidden
    b1 = params[i:i + hidden]; i += hidden
    W2 = params[i:i + hidden]; i += hidden
    b2 = params[i]
    return W1, b1, W2, b2


def pool_patch(X):
    """(...,3,3,4) -> (...,8): [center, mean_of_8_neighbours].
    Fixed, non-trainable — the same 4 features, spatially aggregated."""
    X = np.asarray(X)
    center = X[..., 1, 1, :]                                   # (...,4)
    total = X.sum(axis=(-3, -2))                                # (...,4) sum of all 9
    neighbour_mean = (total - center) / 8.0                     # (...,4) mean of the 8 others
    return np.concatenate([center, neighbour_mean], axis=-1)    # (...,8)


def forward(params, X, S=None, hidden=HIDDEN):
    """X (B,3,3,4) -> P (B,). S accepted and ignored (see module docstring)."""
    W1, b1, W2, b2 = _unpack(params, hidden)
    x = pool_patch(X)                       # (B,8)
    z1 = x @ W1 + b1                        # (B,hidden)
    h = np.tanh(z1)                         # (B,hidden)
    score = h @ W2 + b2                     # (B,)
    p = 1.0 / (1.0 + np.exp(-score))        # (B,)
    return p


def bce_loss_and_grad(params, X, Y, hidden=HIDDEN, w_pos=1.0, eps=1e-7):
    """Mean weighted-BCE loss over the batch + exact analytic gradient wrt
    `params` (same flat layout as init_params). w_pos=1.0 -> plain BCE.

    Forward:
      x = pool(X) (B,8);  z1 = x@W1+b1;  h = tanh(z1);  score = h@W2+b2
      p = sigmoid(score)
      L = -mean( w_pos*y*log(p+eps) + (1-y)*log(1-p+eps) )

    Backward (standard 2-layer-MLP + sigmoid-BCE chain rule):
      dL/dscore_i = ( w_pos*y_i*(p_i-1)/(?) ... ) -- see derivation below;
      for PLAIN BCE (w_pos=1) this simplifies to the classic dL/dscore = p - y
      (per-sample, before the 1/B mean). With w_pos != 1 the sigmoid+BCE
      shortcut no longer applies exactly at p==y edge cases but the eps-regularised
      derivative below is used directly and is exact for the eps-regularised loss.
    """
    W1, b1, W2, b2 = _unpack(params, hidden)
    B = X.shape[0]
    x = pool_patch(X)                       # (B,8)
    z1 = x @ W1 + b1                        # (B,hidden)
    h = np.tanh(z1)                         # (B,hidden)
    score = h @ W2 + b2                     # (B,)
    p = 1.0 / (1.0 + np.exp(-score))        # (B,)

    loss = -np.mean(w_pos * Y * np.log(p + eps) + (1 - Y) * np.log(1 - p + eps))

    # dL/dp for the eps-regularised loss (exact, matches the forward above)
    dL_dp = (-(w_pos * Y) / (p + eps) + (1 - Y) / (1 - p + eps)) / B    # (B,)
    dp_dscore = p * (1 - p)                                             # (B,)
    dscore = dL_dp * dp_dscore                                          # (B,)

    dW2 = h.T @ dscore                                                  # (hidden,)
    db2 = dscore.sum()                                                  # scalar

    dh = np.outer(dscore, W2)                                           # (B,hidden)
    dz1 = dh * (1 - h ** 2)                                             # tanh'  (B,hidden)

    dW1 = x.T @ dz1                                                     # (8,hidden)
    db1 = dz1.sum(axis=0)                                               # (hidden,)

    grad = np.concatenate([dW1.ravel(), db1, dW2, np.array([db2])])
    assert grad.shape == params.shape
    return float(loss), grad


# --------------------------------------------------------------------------- #
# smoke test: pooling correctness, param count, output range, AND a numeric
# (finite-difference) gradient check against the analytic backprop above.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.RandomState(0)

    # 1) param count matches the documented budget, and is <= M3's 38.
    n = param_count(HIDDEN)
    print(f"[1] param_count(hidden={HIDDEN}) = {n}  "
          f"{'OK (<= 38)' if n <= 38 else 'FAIL (> 38, exceeds QML budget)'}")

    # 2) pooling: constant patch -> center == neighbour_mean == that constant.
    const_patch = np.full((1, 3, 3, 4), 0.37)
    pooled = pool_patch(const_patch)[0]
    ok = np.allclose(pooled, 0.37)
    print(f"[2] pool_patch constant-patch -> {pooled[:2]}...  "
          f"{'OK' if ok else 'FAIL'}")

    # 2b) pooling: center differs from neighbours -> pooled correctly separates them.
    patch = np.zeros((1, 3, 3, 4))
    patch[0, 1, 1, :] = 1.0             # center = 1
    patch[0, 0, 0, :] = 0.5             # one neighbour = 0.5, rest = 0
    pooled2 = pool_patch(patch)[0]
    expect_center = np.ones(4)
    expect_neigh = np.full(4, 0.5 / 8)  # (0.5*1 + 0*7)/8
    ok2 = np.allclose(pooled2[:4], expect_center) and np.allclose(pooled2[4:], expect_neigh)
    print(f"[2b] pool_patch center-vs-neighbour separation  {'OK' if ok2 else 'FAIL'}")

    # 3) forward: output shape and range.
    params = init_params(seed=1, hidden=HIDDEN)
    X = rng.uniform(-1, 1, size=(16, 3, 3, 4))
    S = rng.uniform(0, 1, size=(16, 3, 3))         # must be accepted and ignored
    P = forward(params, X, S, hidden=HIDDEN)
    ok3 = P.shape == (16,) and np.all((P >= 0) & (P <= 1)) and np.all(np.isfinite(P))
    print(f"[3] forward shape={P.shape} range=[{P.min():.4f},{P.max():.4f}]  "
          f"{'OK' if ok3 else 'FAIL'}")

    # 3b) S is truly ignored (changing it must not change P).
    S_other = rng.uniform(0, 1, size=(16, 3, 3, 2))    # even a different shape
    P_other = forward(params, X, S_other, hidden=HIDDEN)
    ok3b = np.allclose(P, P_other)
    print(f"[3b] S ignored (per-stage shape too)  {'OK' if ok3b else 'FAIL'}")

    # 4) numeric gradient check: finite differences vs analytic backprop.
    Xb = rng.uniform(-1, 1, size=(8, 3, 3, 4))
    Yb = rng.randint(0, 2, size=(8,)).astype(float)
    loss0, grad = bce_loss_and_grad(params, Xb, Yb, hidden=HIDDEN)
    num_grad = np.zeros_like(params)
    h_fd = 1e-6
    for i in range(len(params)):
        pp = params.copy(); pp[i] += h_fd
        lp, _ = bce_loss_and_grad(pp, Xb, Yb, hidden=HIDDEN)
        pm = params.copy(); pm[i] -= h_fd
        lm, _ = bce_loss_and_grad(pm, Xb, Yb, hidden=HIDDEN)
        num_grad[i] = (lp - lm) / (2 * h_fd)
    max_abs_err = np.max(np.abs(grad - num_grad))
    rel_err = max_abs_err / (np.max(np.abs(num_grad)) + 1e-12)
    ok4 = max_abs_err < 1e-4
    print(f"[4] analytic vs numeric grad -> max|Δ|={max_abs_err:.2e}  "
          f"rel={rel_err:.2e}  {'OK' if ok4 else 'FAIL'}")

    # 5) weighted BCE (w_pos != 1) gradient also checked.
    loss_w, grad_w = bce_loss_and_grad(params, Xb, Yb, hidden=HIDDEN, w_pos=3.6)
    num_grad_w = np.zeros_like(params)
    for i in range(len(params)):
        pp = params.copy(); pp[i] += h_fd
        lp, _ = bce_loss_and_grad(pp, Xb, Yb, hidden=HIDDEN, w_pos=3.6)
        pm = params.copy(); pm[i] -= h_fd
        lm, _ = bce_loss_and_grad(pm, Xb, Yb, hidden=HIDDEN, w_pos=3.6)
        num_grad_w[i] = (lp - lm) / (2 * h_fd)
    max_abs_err_w = np.max(np.abs(grad_w - num_grad_w))
    ok5 = max_abs_err_w < 1e-4
    print(f"[5] weighted-BCE (w_pos=3.6) grad check -> max|Δ|={max_abs_err_w:.2e}  "
          f"{'OK' if ok5 else 'FAIL'}")

    all_ok = ok and ok2 and ok3 and ok3b and ok4 and ok5
    print(f"\nSMOKE TEST: {'PASS' if all_ok else 'FAIL'}")
