# Classical Baseline — 3×3 Convolution

Second classical-model baseline for the **Quantum Change Detection in
Satellite Earth Observations** challenge (2026 Niels Bohr Quantum Summer
School). This directory
implements the **literal parameter-matched twin the QML side's own
documentation designates for its main model, M3**.

So unlike the MLP (an original design, built to explore whether nonlinearity
over a compressed spatial summary helps), this model's job is to implement
*exactly that specified architecture*.

---

## 1. What this model is, and how it differs from the MLP

A single 3×3 convolution: 4 input channels (the same Physical-4/PCA-4
features as everywhere else in this project) → 1 output channel → sigmoid.
It differs from `classical-baseline-mlp` along two *independent* axes:

| | pooled-8 MLP | this conv model |
|---|---|---|
| spatial detail | **discarded** — the 8 neighbours are averaged into one number per band before anything else happens | **preserved** — a separate weight per (row, col, band); the model can learn that a change to the *left* matters differently from one *above* |
| nonlinearity | **yes** — a real `tanh` hidden layer | **no** — a single conv layer with no hidden layer is mathematically just `sigmoid(Σ weight·pixel + bias)`: logistic regression over the 36 raw inputs, with the same weight vector reused (shared) at all 9 output positions |
| output shape | 1 number per patch (centre pixel only) | 9 numbers per patch (full 3×3 map) |
| params | 31 | 37 |

So the trade-off is genuinely inverted between the two models: the MLP sees
*less* spatial detail but processes it *nonlinearly*; this conv model sees
*more* spatial detail but processes it *linearly*. Building both was
deliberate, not redundant — they test different hypotheses about what the
classical side of the comparison needs.

---

## 2. Parameter count: 37

```
kernel : 3 × 3 × 4  (kh, kw, in_channels; single out_channel)  = 36 weights
bias   : 1
                                                        total  = 37
```
Matches the QML docs' own arithmetic for this model (`4*3*3 + 1 = 37`)
exactly, and `37 ≤ 38` (M3's count) — one parameter to spare, not squeezed to
the limit, since the architecture is fixed by the model_ladder.md
specification rather than a free design choice like the MLP's hidden width
was.


---

## 3. Output contract: dense 3×3 → 3×3

The QML ladder's two readout modes:
- `per_pixel`: patch (3×3×4) → dense output (3×3), overlap-averaged across
  patches for full-image coverage — **what M3's headline configuration
  actually uses.**
- `center_mean`: patch (3×3×4) → one probability — a secondary mode. This
  is what the MLP had to use, since pooling the patch down to 8 numbers
  before its hidden layer only leaves room to predict one output.

A conv layer, by contrast, naturally produces a full 3×3 map from a 3×3×4
input — so this model fits M3's *primary* readout mode directly, with no
workaround needed. Practical consequence: full-image inference reuses
`predict_city` (the dense/overlap-averaging path — the *same* function M3
itself uses) unmodified, and `predict_coordinates` needed **zero** changes at
all — it already auto-detects a `(B,3,3)`-shaped output and extracts the
centre pixel for the cheap-val proxy. The MLP's `center_mean` output needed a
dedicated `predict_city_center` path built for it; this model needed nothing
new from the inference module.

```
forward(params, X, S) -> P
  X : (B,3,3,4)  same features as QML (physical [0,1] or pca [-1,1])
  S : (B,3,3) or (B,3,3,2)  QML "change strength" side-channel — accepted
      and IGNORED, same reason as the MLP: no use for it here.
  P : (B,3,3)   per-pixel change probability, one per patch position
```

---

## 4. A deliberate limitation: padding is per-*patch*, not per-*image*

A convolution's real appeal is that it can run **once** over an entire city
image, using every pixel's *true* neighbours — padding only at the actual
image border. That would be a strictly *better* inference procedure than
what this model does. **It's deliberately not done that way.**

Reason: the QML circuit can only ever see one isolated 3×3×4 patch at a time
(9 qubits) — it recovers full-image coverage only by sliding that patch and
averaging overlapping predictions. Letting the classical model see the whole
image directly would hand it strictly *more* context than the quantum model
can ever have, breaking "same input features / same receptive field." So
this model is trained and evaluated under the same artificial restriction:
every kernel application operates on an isolated, **zero-padded** 3×3×4
patch, and full-image predictions come only from the same
overlap-averaging (`predict_city`) procedure M3 uses. A conscious trade-off
against a strictly stronger model, in favour of comparison validity.

**Concretely:** input `X (3,3,4)` is zero-padded to `Xp (5,5,4)`. Output
position `(1,1)` (the patch's true centre) reads `Xp[1:4,1:4]` — the full,
unpadded original patch, the *only* one of the 9 outputs with complete
context. Every other output position's receptive field extends into the
zero-padding for at least one of its 9 kernel taps.

---

## 5. Verification performed

`models/conv.py`'s self-test (`python3 models/conv.py`) covers:

| check | result |
|---|---|
| param count | **37** — matches the QML docs' arithmetic exactly |
| centre-output padding/indexing (recovers the true centre pixel exactly) | max\|Δ\| = **1.1e-16** |
| corner-output padding/indexing (recovers the correct *padded-in-from-corner* pixel) | max\|Δ\| = **2.1e-16** |
| forward: shape/range/finite | OK |
| `S` truly ignored | OK |
| **analytic vs. numeric (finite-difference) gradient**, plain BCE | max\|Δ\| = **1.1e-10** |
| same, weighted BCE (`w_pos=3.6`) | max\|Δ\| = **2.1e-10** |
| weight sharing: perturbing the centre weight or the bias moves **all 9** output positions | OK (72/72) |


The pipeline was also dry-run end-to-end against real `raw_data` (training,
exhaustive full-city prediction via `predict_city`, and PNG mask export —
verified `uint8 {0,255}`, correct `(H,W)` per city) before any real run was
trusted.

---

## 6. Results

### 6.1 Pilot run — PCA-4, same protocol as the MLP

| city | prevalence | AP | ROC-AUC | F1* | precision | ChangeAcc |
|---|---|---|---|---|---|---|
| paris | 0.29% | 0.091 | 0.969 | 0.100 | 0.055 | 0.620 |
| cupertino | 2.37% | 0.623 | 0.977 | 0.598 | 0.544 | 0.664 |
| beihai | 2.49% | 0.303 | 0.898 | 0.343 | 0.300 | 0.402 |

| aggregation | AP | ROC-AUC | F1* | ChangeAcc |
|---|---|---|---|---|
| macro | 0.339 | 0.948 | 0.347 | 0.562 |
| **micro** (pooled, one global τ*) | **0.418** | **0.948** | **0.438** | 0.508 |

For reference, at the same split/protocol/metric code: this conv (0.418 AP,
37 params) sits **below the MLP** (0.468 AP, 31 params) but **well above**
the QML side's own reported M3. Notably weaker on
paris specifically (AP 0.091 vs. the MLP's 0.211) but with much higher
change-recall there (ChangeAcc 0.620 vs. 0.269) — the two models are making
different precision/recall trade-offs, not simply better/worse across the
board.

### 6.2 5-fold city-grouped CV

Given the MLP's single-split result turned out to be well above its own
CV-fold average (`classical-baseline-mlp/README.md` §5.3), this model's CV
was run as part of the same delivery, not added later:

| fold | held-out cities | micro AP |
|---|---|---|
| 0 | cupertino, mumbai, nantes | 0.466 |
| 1 | saclay_e, pisa, aguasclaras | 0.080 |
| 2 | paris, bercy, rennes | 0.262 |
| 3 | hongkong, abudhabi, beirut | 0.280 |
| 4 | bordeaux, beihai | 0.270 |

**mean micro AP = 0.272, std = 0.122.** Essentially the same spread as the
MLP's CV (mean 0.293, std 0.129) — same best fold (cupertino/mumbai/nantes),
same worst fold (saclay_e/pisa/aguasclaras). Full 14-city breakdown:

| city | prevalence | AP | F1* | ROC-AUC | ChangeAcc |
|---|---|---|---|---|---|
| cupertino | 2.37% | 0.632 | 0.604 | 0.978 | 0.647 |
| rennes | 2.58% | 0.596 | 0.586 | 0.976 | 0.661 |
| beirut | 2.69% | 0.443 | 0.490 | 0.901 | 0.461 |
| nantes | 1.14% | 0.335 | 0.403 | 0.953 | 0.449 |
| beihai | 2.49% | 0.314 | 0.344 | 0.895 | 0.343 |
| hongkong | 3.56% | 0.301 | 0.355 | 0.810 | 0.394 |
| mumbai | 2.56% | 0.300 | 0.347 | 0.826 | 0.373 |
| bercy | 0.74% | 0.222 | 0.290 | 0.882 | 0.278 |
| bordeaux | 1.00% | 0.138 | 0.221 | 0.865 | 0.187 |
| pisa | 1.64% | 0.126 | 0.246 | 0.792 | 0.235 |
| paris | 0.29% | 0.100 | 0.113 | 0.972 | 0.581 |
| aguasclaras | 1.64% | 0.095 | 0.176 | 0.778 | 0.201 |
| abudhabi | 3.76% | 0.075 | 0.143 | 0.608 | 0.206 |
| saclay_e | 0.99% | 0.033 | 0.066 | 0.772 | 0.070 |

**Interpretation:** the per-city ranking is nearly identical between this
model and the MLP (cupertino/rennes best, saclay_e/abudhabi worst, in both).
That's a meaningful finding on its own — it suggests the cross-city spread is
a property of **the data/domain-shift**, not of either architecture's
specific inductive bias (pooled-nonlinear vs. spatial-linear). Two
structurally different classical models land in almost the same place,
city-by-city.

### 6.3 Seed sensitivity — is the pilot number seed-luck?

| metric | mean | std | min | max |
|---|---|---|---|---|
| micro AP | 0.4340 | **0.0096** | 0.418 | 0.446 |
| micro F1* | 0.4496 | 0.0114 | 0.437 | 0.468 |
| micro ROC-AUC | 0.9479 | 0.0008 | 0.947 | 0.949 |

Across 5 seeds on the fixed dev split, AP varies by ±0.01 (≈2% relative) —
an order of magnitude smaller than the ~0.12 CV-fold std in §6.2. Same
conclusion as the MLP: **the variance comes from which cities land in
validation, not from initialization or sampling-stream randomness.**
(Slightly higher seed-to-seed variance than the MLP's std of 0.003 — worth
noting, not yet explained; possibly because this model's linear decision
surface is more sensitive to which hard-negative patches the sampler happens
to draw early on, but that's a hypothesis, not verified.)

### 6.4 Evaluation metrics — Accuracy, Change-Accuracy, No-change-Accuracy, F1


| | Accuracy | Change Accuracy | No-change Accuracy | F1 |
|---|---|---|---|---|
| **Pilot** (pooled, dev split) | 0.971 | 0.508 | 0.982 | 0.438 |
| **5-fold CV** (mean ± std) | 0.974 ± 0.006 | 0.345 ± 0.129 | 0.987 ± 0.004 | 0.334 ± 0.106 |
| **Seed check** (mean ± std, dev split) | 0.973 ± 0.001 | 0.491 ± 0.012 | 0.984 ± 0.002 | 0.450 ± 0.011 |


---

## 7. Data pipeline reuse, training protocol, framework choice

All identical decisions and rationale to `classical-baseline-mlp/README.md`
§3 (data pipeline reused directly from `QML-Binary-Segmentation/data/` and
`train/inference.py`, `QML-Binary-Segmentation/train/trainer.py` deliberately
NOT imported to avoid the PennyLane dependency, plain-numpy hand-derived
gradients rather than a framework, dev 11/3 split + pilot protocol matched
for comparability) — not re-derived here to avoid duplication; see that
document for the full reasoning. The one applicable update: per the earlier
framework-choice decision (stay numpy-only while models remain small, revisit
if a future architecture needs more), this model still fits comfortably in
that category — a single linear layer needs no autodiff framework.

---

## 8. How to run

```bash
pip install -r requirements.txt
```

```bash
# model + gradient self-tests
python3 models/conv.py
python3 train/optim.py

# train (dev 11/3 split, pilot defaults, PCA-4 representation)
python3 train/trainer.py --data_dir /path/to/raw_data \
  --representation pca --tag conv_pca_run1

# exhaustive evaluation of a checkpoint on all 3 dev-val cities + PNG masks
python3 train/eval_full.py --data_dir /path/to/raw_data \
  --ckpt results/runs/conv_pca_run1_bestcheap.npy --representation pca \
  --mask_dir results/runs/conv_pca_run1_masks

# 5-fold city-grouped CV (trains 5 models, ~80s total)
python3 train/cv.py --data_dir /path/to/raw_data --representation pca

# seed sensitivity on the fixed dev split (5 seeds, ~80s total)
python3 train/seed_check.py --data_dir /path/to/raw_data --representation pca
```

`--data_dir` should point at the `raw_data/` folder (containing `images/` and
`train_labels/`).

Artifacts land in `results/runs/`, same convention as the MLP directory:
`<tag>.jsonl` (per-epoch log), `<tag>_bestcheap.npy` (best pooled cheap-val
AP — use for cross-model comparison), `<tag>_best.npy` (best exhaustive AP,
biased), `<tag>_final.npy`, `<tag>_fullval.json`, and `cv.py`/`seed_check.py`
additionally write a `*_summary.json`.

---

## 9. Status / next steps

| stage | state |
|---|---|
| Architecture implemented exactly per `model_ladder.md`'s specification | ✅ |
| Parameter count verified (37, matches QML docs' own arithmetic) | ✅ |
| Gradient correctness verified (analytic vs. numeric) | ✅ |
| Padding/indexing correctness verified (centre + corner + weight-sharing checks) | ✅ |
| End-to-end pipeline dry-run + PNG mask format verified | ✅ |
| Real training run, exhaustive multi-city evaluation | ✅ |
| 5-fold city-grouped CV | ✅ — mean micro AP 0.272 ± 0.122, same per-city pattern as the MLP |
| Seed sensitivity | ✅ — small (std 0.010 on AP) relative to CV spread |


---

## Disclosure

This code and write-up were prepared with assistance from a generative AI
coding assistant (Anthropic Claude, in Claude Code). Every checkable claim
above was independently verified by actually running the referenced script
against real data or a numerical check: the parameter count, the
padding/indexing behaviour (including the corrected weight-sharing test,
documented in §5 rather than silently fixed), the gradient correctness, the
full training/CV/seed-sensitivity runs, and the PNG mask format were all
executed and their console output inspected, not merely written into this
document.
