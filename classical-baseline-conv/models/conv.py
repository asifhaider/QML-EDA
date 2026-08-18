"""
models/conv.py — the 3x3 same-pad convolution classical baseline.

WHAT THIS MODEL IS, AND WHY THIS ONE SPECIFICALLY
----------------------------------------------------
This is not a design choice made from scratch — it is the LITERAL classical
twin the QML side's own documentation designates for its main model, M3:

  "M3 (38 params) vs classical 3x3 same-pad conv, 4->1 channels: 37 params.
   Same input, same receptive field."
  -- QML-Binary-Segmentation/docs/model_ladder.md

So the job here is to implement that model precisely and honestly, not to
invent a new architecture. It differs from the pooled-8 MLP
(classical-baseline-mlp/) along two independent axes, worth being explicit
about because "conv" sounds like the more sophisticated model and isn't,
exactly:

  1. SPATIAL DETAIL is preserved, not pooled away. The MLP's first step
     collapses the 8 non-centre pixels into a single averaged number per
     band, discarding which neighbour changed. A conv kernel instead keeps a
     separate weight per (row, col, band) triple -- it can learn that a
     change to the LEFT matters differently from a change ABOVE, which the
     pooled-8 MLP structurally cannot express.
  2. LINEARITY: despite (1), a single conv layer with no hidden layer is
     mathematically JUST a linear filter -- each output pixel is
     sigmoid(sum(weight * pixel) + bias), i.e. logistic regression over the
     36 raw inputs, with the same 37-number weight vector reused at all 9
     output positions (weight SHARING / translation invariance). It has no
     internal nonlinearity, unlike the MLP's tanh hidden layer. So this
     model is "sees more spatial detail, but is strictly linear" -- the
     opposite trade-off from the MLP's "sees an averaged summary, but is
     nonlinear."

PARAMETER COUNT: 37
-----------------------
    kernel : (3,3,4) = 3*3*4 = 36 weights   (kh, kw, in_channels; single
             out_channel, so no separate output-channel dimension)
    bias   :  1
    total  : 37
`4*3*3 + 1 = 37 <= 38` (M3's count) -- documented and verified identical to
the QML side's own arithmetic for this exact model.

OUTPUT CONTRACT: dense 3x3 -> 3x3, matching M3's actual readout
-------------------------------------------------------------------
Unlike the MLP (which only naturally produces one number per patch, matching
the QML ladder's secondary "center_mean" readout), a conv layer naturally
produces a full 3x3 map from a 3x3x4 input -- this is M3's ACTUAL headline
readout mode ("per_pixel": patch -> dense 3x3 output, combined across
overlapping patches by averaging). So this model reuses `predict_city` (the
dense/overlap-averaging inference path) and the ORIGINAL `predict_coordinates`
behaviour (which already auto-detects a (B,3,3)-shaped output and takes the
centre pixel for the cheap-val proxy) UNCHANGED from
QML-Binary-Segmentation/train/inference.py -- no new inference code was
needed, unlike the MLP, which required predict_city_center specifically.

    forward(params, X, S) -> P
      X : (B,3,3,4)  same features as QML (physical [0,1] or pca [-1,1])
      S : (B,3,3) or (B,3,3,2)  QML "change strength" side-channel -- accepted
          and IGNORED, same reason as the MLP: this model has no use for it.
      P : (B,3,3)   per-pixel change probability, one per patch position

WHY "SAME" PADDING IS APPLIED PER-PATCH, NOT ACROSS THE WHOLE IMAGE
------------------------------------------------------------------------
A convolution's whole appeal is that it can, in principle, be run ONCE over
an entire city image, using every pixel's TRUE neighbours -- no padding
needed except at the true image border. That would be a strictly BETTER
inference procedure than what's implemented here. It is deliberately NOT
what this model does, for a fairness reason: the QML circuit is fundamentally
restricted to ever seeing one isolated 3x3x4 patch at a time (9 qubits), and
recovers full-image coverage only by sliding that patch with stride 1 and
averaging overlapping predictions (`predict_city`). Giving the classical
model true whole-image context instead would violate "same input features /
same receptive field" -- it would let the conv see strictly MORE at
inference time than the quantum model ever can. So this model is trained and
evaluated under the exact same artificial restriction: every application of
the kernel operates on an isolated, ZERO-PADDED 3x3x4 patch (see `_patches`
below), and full-image predictions come ONLY from the same overlap-averaging
`predict_city` procedure M3 itself uses. This is a conscious trade-off
against a strictly stronger model, in favour of comparison validity -- worth
knowing before ever "improving" this by convolving over the whole image
directly.

PADDING DEFINITION (worked out explicitly, since off-by-one errors here are
the classic conv bug)
-------------------------------------------------------------------------------
Input X: (3,3,4). Zero-pad by 1 on the two spatial dims -> Xp: (5,5,4).
For output position (i,j) in {0,1,2}x{0,1,2}, its receptive field in Xp is
rows [i, i+3) and cols [j, j+3) -- i.e. output (0,0) reads Xp[0:3, 0:3],
which is centred on the padded coordinate (1,1) = original X's (0,0), with
one ring of zeros on its top/left (the true out-of-patch neighbours, which
this model never gets to see -- see the fairness note above). Output (1,1)
(the patch's true centre) reads Xp[1:4, 1:4] = the FULL, unpadded original
3x3x4 patch -- the only one of the 9 outputs with zero missing context.

No autodiff framework is used, matching classical-baseline-mlp's choice (see
that repo's README §3): the exact backprop gradient for this linear+sigmoid
model is a few lines, verified against finite differences in the smoke test.
"""
import numpy as np

KH, KW, CIN = 3, 3, 4
N_WEIGHTS = KH * KW * CIN     # 36
# (di, dj) offsets, matching the flattening order used everywhere below --
# fixed once here so forward/backward/init all agree on weight layout.
_OFFSETS = [(di, dj) for di in range(KH) for dj in range(KW)]


def param_count():
    return N_WEIGHTS + 1     # 37


def init_params(seed=0):
    """Flat vector: [W.ravel() (36, order = di,dj,c -- see _OFFSETS/_patches), b (1)].
    Glorot-uniform-like scale for the weights (appropriate for a
    sigmoid-output linear layer); bias starts at 0 (no prior class-balance
    assumption baked in -- the sampler's ~78:22 mixture is handled by
    training, not initialization, matching the MLP's approach)."""
    rng = np.random.RandomState(seed)
    limit = np.sqrt(6.0 / (N_WEIGHTS + 1))
    W = rng.uniform(-limit, limit, size=N_WEIGHTS)
    b = np.zeros(1)
    v = np.concatenate([W, b])
    assert v.size == param_count()
    return v


def _pad(X):
    """(...,3,3,4) -> (...,5,5,4), zero-padded by 1 on the two spatial dims."""
    pw = [(0, 0)] * (X.ndim - 3) + [(1, 1), (1, 1), (0, 0)]
    return np.pad(X, pw, mode="constant", constant_values=0.0)


def _patches(Xp):
    """(B,5,5,4) padded input -> (B,9,36): for each of the 9 output positions
    (flattened row-major, k = i*3+j), the flattened 3x3x4 receptive field
    that produces it, in the SAME (di,dj,c) order as the weight vector."""
    B = Xp.shape[0]
    cols = [Xp[:, i:i + 3, j:j + 3, :].reshape(B, N_WEIGHTS) for (i, j) in _OFFSETS]
    return np.stack(cols, axis=1)          # (B,9,36)


def forward(params, X, S=None):
    """X (B,3,3,4) -> P (B,3,3). S accepted and ignored (see module docstring)."""
    W, b = params[:N_WEIGHTS], params[N_WEIGHTS]
    B = X.shape[0]
    patches = _patches(_pad(np.asarray(X)))            # (B,9,36)
    score = patches @ W + b                             # (B,9)
    p = 1.0 / (1.0 + np.exp(-score))                    # (B,9)
    return p.reshape(B, 3, 3)


def bce_loss_and_grad(params, X, Y, w_pos=1.0, eps=1e-7):
    """Mean weighted-BCE over ALL 9 patch pixels AND the batch (matches M3's
    `L = (1/9) sum_i BCE(y_i, p_i)` convention -- Y here is the FULL (B,3,3)
    patch label grid, not just the centre, unlike the MLP). Returns
    (loss, grad) with grad in the same flat [W(36), b(1)] layout as
    init_params.

    Backward: this is a linear layer (shared weights across 9 positions) +
    sigmoid + BCE, so per output position k, dL/dscore_k = (p_k - y_k) (the
    standard sigmoid-BCE shortcut), scaled by 1/(9*B) for the mean. Because
    the SAME weight vector produced all 9 outputs (weight sharing), its
    gradient is the SUM over all 9 positions of (that position's dscore *
    that position's receptive-field patch) -- the standard
    "convolution weight gradient = cross-correlation of the input patches
    with the output gradient" identity, implemented directly via `patches`
    (already computed in the forward pass, so it's reused here rather than
    recomputed)."""
    W, b = params[:N_WEIGHTS], params[N_WEIGHTS]
    B = X.shape[0]
    patches = _patches(_pad(np.asarray(X)))             # (B,9,36)
    score = patches @ W + b                              # (B,9)
    p = 1.0 / (1.0 + np.exp(-score))                     # (B,9)
    y = np.asarray(Y).reshape(B, 9)

    loss = -np.mean(w_pos * y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

    dL_dp = (-(w_pos * y) / (p + eps) + (1 - y) / (1 - p + eps)) / (B * 9)   # (B,9)
    dp_dscore = p * (1 - p)                                                  # (B,9)
    dscore = dL_dp * dp_dscore                                               # (B,9)

    dW = np.einsum("bk,bkf->f", dscore, patches)          # (36,) -- summed over B and 9 positions
    db = dscore.sum()                                     # scalar

    grad = np.concatenate([dW, np.array([db])])
    assert grad.shape == params.shape
    return float(loss), grad


# --------------------------------------------------------------------------- #
# smoke test: padding/indexing correctness, param count, output range,
# S-ignored, AND a numeric (finite-difference) gradient check.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.RandomState(0)

    # 1) param count.
    n = param_count()
    print(f"[1] param_count() = {n}  {'OK (<= 38)' if n <= 38 else 'FAIL'}"
          f"  (QML docs: 4*3*3+1=37 -> {'matches' if n == 37 else 'MISMATCH'})")

    # 2) padding/indexing: centre output (1,1) must read the FULL, unpadded
    # patch with no zeros involved -- verified by using a kernel = 1 at
    # (di,dj,c)=(1,1,c) [i.e. picks out exactly the centre input pixel] and
    # checking output (1,1) equals a simple, independently-computed function.
    W = np.zeros(N_WEIGHTS); b = np.zeros(1)
    # index of (di=1,dj=1,c=0) in the flattened (di,dj,c) order:
    idx_center_c0 = _OFFSETS.index((1, 1)) * CIN + 0
    W[idx_center_c0] = 1.0
    params = np.concatenate([W, b])
    X = rng.uniform(-1, 1, size=(4, 3, 3, 4))
    P = forward(params, X)
    # with this kernel, score(i,j) = Xp[i+1, j+1, 0] = X's channel-0 value at
    # the pixel one step INTO the patch from (i,j) in padded coords, which
    # for (i,j)=(1,1) is exactly X[1,1,0] (the true centre, channel 0) since
    # sigmoid(x) for x=that raw value: check pre-sigmoid recovers X[1,1,0].
    recovered = np.log(P[:, 1, 1] / (1 - P[:, 1, 1]))       # invert sigmoid -> score
    ok2 = np.allclose(recovered, X[:, 1, 1, 0], atol=1e-6)
    print(f"[2] centre-output padding/indexing check -> max|Δ|="
          f"{np.abs(recovered - X[:,1,1,0]).max():.2e}  {'OK' if ok2 else 'FAIL'}")

    # 2b) a CORNER output position must be affected by zero-padding: same
    # kernel, output (0,0) should equal Xp[1,1,0] = X[0,0,0] (one step in
    # from the padded corner) -- NOT X[1,1,0]. Confirms padding is being
    # applied (not silently skipped) and the offset direction is right.
    recovered00 = np.log(P[:, 0, 0] / (1 - P[:, 0, 0]))
    ok2b = np.allclose(recovered00, X[:, 0, 0, 0], atol=1e-6)
    print(f"[2b] corner-output padding/indexing check -> max|Δ|="
          f"{np.abs(recovered00 - X[:,0,0,0]).max():.2e}  {'OK' if ok2b else 'FAIL'}")

    # 3) forward: output shape/range with a real random kernel.
    params = init_params(seed=1)
    S = rng.uniform(0, 1, size=(4, 3, 3))
    P = forward(params, X, S)
    ok3 = P.shape == (4, 3, 3) and np.all((P >= 0) & (P <= 1)) and np.all(np.isfinite(P))
    print(f"[3] forward shape={P.shape} range=[{P.min():.4f},{P.max():.4f}]  "
          f"{'OK' if ok3 else 'FAIL'}")

    # 3b) S truly ignored.
    S_other = rng.uniform(0, 1, size=(4, 3, 3, 2))
    ok3b = np.allclose(P, forward(params, X, S_other))
    print(f"[3b] S ignored (per-stage shape too)  {'OK' if ok3b else 'FAIL'}")

    # 4) numeric gradient check: finite differences vs analytic backprop,
    # over the FULL flat parameter vector, with a dense (B,3,3) target.
    Xb = rng.uniform(-1, 1, size=(8, 3, 3, 4))
    Yb = rng.randint(0, 2, size=(8, 3, 3)).astype(float)
    loss0, grad = bce_loss_and_grad(params, Xb, Yb)
    num_grad = np.zeros_like(params)
    h_fd = 1e-6
    for i in range(len(params)):
        pp = params.copy(); pp[i] += h_fd
        lp, _ = bce_loss_and_grad(pp, Xb, Yb)
        pm = params.copy(); pm[i] -= h_fd
        lm, _ = bce_loss_and_grad(pm, Xb, Yb)
        num_grad[i] = (lp - lm) / (2 * h_fd)
    max_abs_err = np.max(np.abs(grad - num_grad))
    rel_err = max_abs_err / (np.max(np.abs(num_grad)) + 1e-12)
    ok4 = max_abs_err < 1e-4
    print(f"[4] analytic vs numeric grad -> max|Δ|={max_abs_err:.2e}  "
          f"rel={rel_err:.2e}  {'OK' if ok4 else 'FAIL'}")

    # 5) weighted BCE gradient check too.
    loss_w, grad_w = bce_loss_and_grad(params, Xb, Yb, w_pos=3.6)
    num_grad_w = np.zeros_like(params)
    for i in range(len(params)):
        pp = params.copy(); pp[i] += h_fd
        lp, _ = bce_loss_and_grad(pp, Xb, Yb, w_pos=3.6)
        pm = params.copy(); pm[i] -= h_fd
        lm, _ = bce_loss_and_grad(pm, Xb, Yb, w_pos=3.6)
        num_grad_w[i] = (lp - lm) / (2 * h_fd)
    max_abs_err_w = np.max(np.abs(grad_w - num_grad_w))
    ok5 = max_abs_err_w < 1e-4
    print(f"[5] weighted-BCE (w_pos=3.6) grad check -> max|Δ|={max_abs_err_w:.2e}  "
          f"{'OK' if ok5 else 'FAIL'}")

    # 6) weight SHARING sanity: the SAME weight vector produces all 9 output
    # positions, so perturbing a weight whose receptive field never falls in
    # the zero-padded region must move EVERY position. Only the CENTRE
    # kernel weight (offset (1,1)) and the BIAS have this property -- every
    # other offset's receptive field lands in the padding zone for at least
    # one output position, where it has EXACTLY zero effect (0 * weight = 0,
    # not just a small gradient) -- see part 6b for that (expected, not a
    # bug) asymmetry made explicit.
    params2 = init_params(seed=2)
    P_a = forward(params2, Xb)

    idx_center_c0 = _OFFSETS.index((1, 1)) * CIN + 0
    params2b = params2.copy(); params2b[idx_center_c0] += 0.3
    moved_center = np.abs(P_a - forward(params2b, Xb)) > 1e-9
    ok6 = moved_center.all()
    print(f"[6] weight sharing (centre weight, no padding blind-spot) -> "
          f"{moved_center.sum()}/{moved_center.size} of all (sample,position) "
          f"outputs moved  {'OK' if ok6 else 'FAIL'}")

    params2c = params2.copy(); params2c[N_WEIGHTS] += 0.3     # bias
    moved_bias = np.abs(P_a - forward(params2c, Xb)) > 1e-9
    ok6b = moved_bias.all()
    print(f"[6b] weight sharing (bias, no padding blind-spot)       -> "
          f"{moved_bias.sum()}/{moved_bias.size} moved  {'OK' if ok6b else 'FAIL'}")

    # 6c) documents the expected asymmetry (not asserted as pass/fail, just
    # shown): a non-centre weight, e.g. offset (0,1) [reads one row ABOVE
    # each output position], structurally CANNOT affect the output row that
    # has no row above it within the padded patch (the top row, i=0), since
    # that receptive-field read is always the zero-padding value there.
    idx_top_c1 = _OFFSETS.index((0, 1)) * CIN + 1
    params2d = params2.copy(); params2d[idx_top_c1] += 0.3
    moved_edge = (np.abs(P_a - forward(params2d, Xb)) > 1e-9).reshape(8, 3, 3)
    print(f"[6c] (informational) offset-(0,1) weight moves rows i=1,2 but "
          f"NOT i=0 (always-padded there): per-row moved-count, summed over "
          f"8 samples x 3 columns = {moved_edge.sum(axis=(0,2)).tolist()} "
          f"(expect [0, 24, 24])")

    all_ok = ok2 and ok2b and ok3 and ok3b and ok4 and ok5 and ok6 and ok6b
    print(f"\nSMOKE TEST: {'PASS' if all_ok else 'FAIL'}")
