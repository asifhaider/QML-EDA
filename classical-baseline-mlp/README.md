# Classical Baseline — Pooled-8 MLP

Classical-model half of the **Quantum Change Detection in Satellite Earth
Observations** challenge (2026 Niels Bohr Quantum Summer School). This directory implements the parameter-matched classical
comparison the challenge requires: *"a classical model receiving the same
input features and not containing more trainable parameters than the QML
model."*

This first model is a small **MLP** (multi-layer perceptron). 

---

## 1. 38 parameters, 36 raw inputs

The QML side's main model (**M3**, depth 1) has **38 trainable parameters**
and consumes a **3×3×4 patch** (9 pixels × 4 features/pixel — either the
`Physical-4` bands `{B04,B05,B12,B08}` or a `PCA-4` compression of the
13-band base. The challenge rule (`N_classical ≤ N_QML`, same input
features) means the classical model must:
- receive the same 3×3×4 = 36 numbers,
- use **≤ 38** trainable parameters.

### 1.1 Why a naive "flatten → small MLP" doesn't work

The obvious first idea — flatten the patch to 36 numbers and feed a tiny
MLP — fails immediately on the arithmetic: a fully-connected layer from 36
inputs to even **one** hidden unit already costs `36 weights + 1 bias = 37`
parameters, leaving essentially nothing for anything downstream. So the
hidden layer would be forced to width 1.

That's not just small, it's **degenerate**. A network with one hidden unit is
```
p = sigmoid(a · f(w·x) + b)
```
for some monotonic activation `f` (tanh/ReLU/sigmoid). A composition of
monotonic functions of a *single* scalar (`w·x`) is itself monotonic in that
scalar. That means:
- its **ranking of pixels by predicted probability is identical** to plain
  logistic regression on `w·x` alone,
- so **AP and ROC-AUC — both rank-based metrics — come out the same**,
- and **F1** (ranking + one threshold) is also invariant to this kind of
  monotone rescaling — `QML-Binary-Segmentation/train/inference.py`'s
  docstring makes exactly this point about AP/F1 for a different reason
  (calibration invariance), and it applies here too.

In other words: **a literal 36→1→1 MLP is not a different model from
logistic regression at this budget** — it would just be extra code that
produces the same answer, dressed up as "nonlinear." Worth catching before
building it, not after.

The next-cheapest fix — 2 hidden units — doesn't help either: `36 × 2 = 72`
weights in the first layer alone, already ~2× over budget.

### 1.2 The fix: pool the patch to 8 numbers *before* the MLP

Instead of feeding all 36 raw numbers into the MLP, the patch is first
reduced with a **fixed, non-trainable** transform:

```
x_pool = [ center_pixel (4,),  mean_of_the_8_neighbours (4,) ]     ->  8 numbers
```

This is not new information smuggled into the input — it's the *same* 4
features (whichever representation is used) that the QML circuit sees,
spatially aggregated. Mean-pooling over a window is exactly the operation
validated by the QML side's own EDA. So this isn't an arbitrary
classical-side trick; it leans on the same idea the feature analysis already
established.

With an 8-dimensional input, a hidden layer of width ≥ 2 becomes affordable,
which breaks the monotonic-collapse problem: the network can now combine
*two different linear projections* of `x_pool` through a nonlinearity before
recombining them — something a single scalar composition provably cannot do.

### 1.3 Parameter budget (hidden width = 3, the default)

```
layer 1  (8 -> 3):   W1 (8,3) = 24 weights  +  b1 (3,) = 3 biases   = 27
layer 2  (3 -> 1):   W2 (3,)  =  3 weights  +  b2 ()   = 1 bias     =  4
                                                          total     = 31
```
`31 ≤ 38` ✓ — with 7 params of headroom left deliberately unused, in favor of
a standard, fully-biased architecture over squeezing to exactly 38 by
dropping a bias somewhere. (If it later seems worth using the full budget:
hidden width 4 with the hidden-layer bias dropped gives exactly `9·4 + 4 + 1
= 37` params — noted here as the next option, not implemented yet.)

`hidden` is a config parameter (`models/mlp.py::HIDDEN`, default 3;
`param_count(hidden)` computes the exact count for any width), so this is
easy to revisit.

---

## 2. Output contract: predicts the *centre* pixel, not the whole 3×3

The QML ladder has two readout modes:
- `per_pixel`: patch (3×3×4) → dense output (3×3) — what M3's headline
  config uses, combined across overlapping patches by averaging.
- `center_mean`: patch (3×3×4) → **one** probability, for the centre pixel.

Because this MLP pools the patch down to a summary before its hidden layer,
it can only naturally produce **one** number per patch — i.e. it is a
`center_mean`-style model. Practically this means:
- **no overlap-averaging is needed for full-image inference** — stride the
  3×3 window across the city, one prediction per position, done. (This also
  makes classical inference dramatically cheaper than the QML side's
  overlap-averaged dense evaluation — full-city evaluation here takes ~0.2–1.5
  s per city, since this is a small numpy MLP rather than a simulated quantum
  circuit.)
- it reuses `predict_city_center`, `predict_coordinates`, and
  `evaluate_predictions` from `QML-Binary-Segmentation/train/inference.py`
  *unmodified* — those functions only assume a `forward(params, X, S) -> P`
  contract, they don't care what's inside `forward`.
- the comparison against M3's dense output is still fair: both ultimately
  produce a full H×W probability map, evaluated with the identical
  `evaluate_predictions` metric code — they just get there by different
  routes (overlap-averaging several patches vs. one patch per pixel).

---

## 3. Data pipeline: reused, not reimplemented

This directory imports `splits.py`, `preprocess.py`, `pools.py`, `sampler.py`
(from `QML-Binary-Segmentation/data/`) and `inference.py` (from
`QML-Binary-Segmentation/train/`) **directly**, via `sys.path`, rather than
re-deriving equivalent logic here.

**Why:** the challenge's "same input features" requirement is easy to
silently violate — a slightly different percentile clip, a subtly different
median-correction, a different train/val city split — and such a mismatch
would invalidate the whole comparison without being obviously wrong-looking.
Importing the *exact same functions* makes that class of bug impossible by
construction, at the cost of a soft dependency between the two directories
(this one won't run standalone without its sibling directory present).

**What's reused:** city splits (`get_dev_split`), the robust band
normalization + per-pair median correction + Physical-4/PCA-4 transforms
(`build_fold`, `transform_physical4/pca4`), the positive/hard-negative/
ordinary-negative pool logic (`build_center_pools`), the city-balanced 1:1:2
stochastic sampler (`SpatialPatchSampler`), and all inference/metric code.

**What's NOT reused, on purpose:** `QML-Binary-Segmentation/train/trainer.py`
is *not* imported, even though it has a `build_representation` helper this
directory also needs (duplicated here, ~10 lines) — importing it would pull
in `import pennylane as qml` as a side effect, and this directory has no
other reason to depend on a quantum simulation framework. Confirmed:
`requirements.txt` here lists only `numpy`, `scikit-learn`, `pillow` — no
PennyLane. (Considered switching everything to PyTorch for extensibility;
decided to stay numpy-only while models remain this small.

**Training protocol matched to the QML side's pilot defaults** (dev 11/3
split, `lr=0.02`, `batch=32`, `steps_per_epoch=160`, `epochs=20`, plain BCE,
same 1:1:2 sampler) — for comparability, not because these were independently
tuned. §5 below reports what happened when that assumption was actually
tested.

---

## 4. Verification performed

`models/mlp.py`, `train/optim.py` are self-checking (`python3 <file>.py`):

| check | result |
|---|---|
| param count (hidden=3) | **31**, confirmed ≤ 38 |
| pooling: constant patch → center = neighbour-mean = that constant | **OK** |
| pooling: center ≠ neighbours → correctly separated | **OK** |
| forward: output shape `(B,)`, range `[0,1]`, all finite | **OK** |
| forward: `S` truly ignored (changing it, even its shape, doesn't change `P`) | **OK** |
| **analytic backprop gradient vs. numerical (finite-difference) gradient**, plain BCE | max\|Δ\| = **1.1e-10** |
| same check, weighted BCE (`w_pos=3.6`) | max\|Δ\| = **3.4e-10** |
| Adam optimizer smoke test (converges a toy quadratic to the true minimum) | max\|err\| = **1.2e-11** |

The full pipeline was also dry-run **end-to-end against the real `raw_data`**
(not just synthetic data) for both representations:
- `pca` and `physical`, tiny configs, **completed with no errors**, produced
  finite, sane (non-degenerate) AP/F1 numbers.
- The **exhaustive full-city path** (`predict_city_center` over an entire
  city, e.g. paris 408×390) was exercised and produced correct-shaped,
  correctly-valued metrics.
- `eval_full.py` was run end-to-end including **PNG mask export**: verified
  the output masks are `uint8`, values exactly `{0, 255}`, and shaped
  identically to each city's true `(H, W)` — matching the challenge's
  required deliverable format.

Scratch/smoke-test artifacts were deleted after verification; `results/runs/`
holds only real runs (§5).

---

## 5. Results so far (dev split: train on 11 cities, val = paris / cupertino / beihai)

### 5.1 Pilot run — PCA-4 representation, default protocol

Full 20-epoch run, evaluated exhaustively on all 3 held-out cities (not just
the cheap single-city proxy used during training):

| city | prevalence | AP | ROC-AUC | F1* | precision | ChangeAcc |
|---|---|---|---|---|---|---|
| paris | 0.29% | 0.211 | 0.978 | 0.263 | 0.257 | 0.269 |
| cupertino | 2.37% | 0.627 | 0.977 | 0.596 | 0.556 | 0.643 |
| beihai | 2.49% | 0.339 | 0.915 | 0.378 | 0.356 | 0.404 |

| aggregation | AP | ROC-AUC | F1* | ChangeAcc |
|---|---|---|---|---|
| macro (per-city mean) | 0.392 | 0.957 | 0.412 | 0.439 |
| **micro** (pooled, one global τ*) | **0.468** | **0.954** | **0.476** | 0.494 |

Wall time: **~10 s** train + **~10 s** exhaustive eval (a plain-numpy 31-param
model vs. a simulated 9-qubit circuit — several orders of magnitude cheaper
per forward pass, which is why exploratory runs below cost seconds, not
hours).

At the same split / protocol / metric code, the QML side's own reported M3
numbers (`QML-Binary-Segmentation/docs/results_capacity_sweep.md`) are:

| model | params | micro AP | micro F1* | micro ROC-AUC |
|---|---|---|---|---|
| this MLP | 31 | **0.468** | **0.476** | **0.954** |
| QML M3, L=1 | 38 | 0.111 | 0.189 | 0.865 |
| QML M3, L=2 | 74 | 0.147 | 0.302 | 0.899 |

At fewer-or-comparable parameters and an identical protocol, this MLP
currently scores well above every M3 configuration on every headline metric,
**on this specific 3-city split**. §5.3 below tests whether that holds up
across different splits — it partially does, and partially doesn't; read
that section before treating this table as a general conclusion.

### 5.2 Is 20 epochs enough? (tested, not assumed)

The pilot protocol's `epochs=20` was copied from the QML side's own pilot
defaults, not independently tuned — so it was tested directly:

| run | config | best checkpoint at | exhaustive micro AP |
|---|---|---|---|
| pilot | lr=0.02, 20 epochs | epoch 2 | 0.468 |
| 5× longer | lr=0.02, **100 epochs** | epoch 2 (unchanged — deterministic, same seed) | 0.468 |
| lower lr | lr=0.005, 40 epochs | epoch 4 | 0.475 |
| lower lr | lr=0.002, 40 epochs | epoch 14 | 0.467 |

Findings:
- Running **5× longer changed nothing**: train BCE never dropped below ~0.41
  at any point across 100 epochs — it plateaus by epoch ~4–5, then
  noisily oscillates (0.41–0.44) for the remaining 95 epochs with no trend.
- Lowering the learning rate 4×–10× smoothed the curve but **did not raise
  the ceiling** — exhaustive micro AP across all three learning rates lands
  in a tight band, **0.467–0.475**.
- The best checkpoint was always found early (epoch 2, 4, or 14) — never
  late — in every run tested.

**Conclusion: 20 epochs is sufficient.** This model reaches its plateau
within the first ~15 epochs regardless of learning rate; training longer or
tuning the learning rate does not move the ceiling. That ceiling (micro AP ≈
0.47) looks like a genuine fit limit of this architecture/feature
combination on this data, not an artifact of under-training. This does *not*
rule out that a different pooling design, a wider hidden layer within
budget, or the planned conv baseline could push past it — that's an
architecture question, separate from training duration. (Whether it's
seed-dependent is checked next, in §5.4 — it isn't.)

The trainer already guards against reporting a stale/degraded late-epoch
checkpoint: it tracks the best pooled-cheap-AP checkpoint every epoch
(`*_bestcheap.npy`), so the number in §5.1 reflects the actual best point
found, not an arbitrary final epoch.

### 5.3 5-fold city-grouped CV — does the §5.1 result hold up?

`train/cv.py` trains 5 independent models, each holding out a different
~2–3 city group (`splits.py::get_grouped_folds(n_splits=5)`), so all 14
labelled cities are evaluated as a held-out city exactly once. Same protocol
as §5.1 (PCA-4, 20 epochs — already shown sufficient in §5.2).

| fold | held-out cities | micro AP |
|---|---|---|
| 0 | cupertino, mumbai, nantes | 0.468 |
| 1 | saclay_e, pisa, aguasclaras | 0.080 |
| 2 | paris, bercy, rennes | 0.371 |
| 3 | hongkong, abudhabi, beirut | 0.256 |
| 4 | bordeaux, beihai | 0.290 |

**mean micro AP = 0.293, std = 0.129** across folds — a ~6× spread between
the best (0.468) and worst (0.080) fold. Full per-city breakdown (each city
evaluated once, as its fold's held-out city):

| city | prevalence | AP | F1* | ROC-AUC | ChangeAcc |
|---|---|---|---|---|---|
| cupertino | 2.37% | 0.639 | 0.605 | 0.978 | 0.634 |
| rennes | 2.58% | 0.588 | 0.582 | 0.975 | 0.632 |
| beirut | 2.69% | 0.394 | 0.446 | 0.905 | 0.470 |
| beihai | 2.49% | 0.350 | 0.398 | 0.915 | 0.399 |
| mumbai | 2.56% | 0.331 | 0.376 | 0.817 | 0.384 |
| nantes | 1.14% | 0.328 | 0.393 | 0.956 | 0.434 |
| hongkong | 3.56% | 0.277 | 0.351 | 0.819 | 0.366 |
| paris | 0.29% | 0.207 | 0.270 | 0.978 | 0.378 |
| bercy | 0.74% | 0.181 | 0.250 | 0.889 | 0.233 |
| bordeaux | 1.00% | 0.147 | 0.214 | 0.890 | 0.190 |
| pisa | 1.64% | 0.138 | 0.217 | 0.775 | 0.192 |
| aguasclaras | 1.64% | 0.083 | 0.180 | 0.780 | 0.226 |
| abudhabi | 3.76% | 0.080 | 0.148 | 0.631 | 0.209 |
| saclay_e | 0.99% | 0.028 | 0.062 | 0.758 | 0.075 |

**What this changes about §5.1's comparison:** the 0.468 headline number was
computed on one specific 3-city split (paris/cupertino/beihai), and that
split is real — it wasn't cherry-picked, it's the QML side's own dev split,
reused for exact comparability. But it isn't representative of the model's
*average* case: it happens to include cupertino, the single best-performing
city (AP 0.64), which pulls the pooled/micro number up. The honest summary
is: **on the dev split specifically, at matched protocol, this MLP still
outperforms the reported QML M3 numbers** (0.468 vs 0.111–0.147 AP) — that
comparison is unaffected by this CV, since it was never claimed to be
anything but a same-split comparison. But **this MLP's generalization is
highly city-dependent** (AP swings from 0.03 to 0.64 by city), so the dev
split's 0.468 should not be read as "the model's AP is ~0.47" in general —
the CV mean (0.293) is a more honest single-number summary of typical
performance, and even individual cities span nearly a 10× range in AP by
this model's ranking of them (saclay_e 0.028 vs cupertino 0.639), independent
of prevalence (e.g. abudhabi has the *highest* prevalence, 3.76%, and one of
the *worst* APs, 0.080). What drives that per-city spread hasn't been
investigated yet — it's a genuine open question, not explained by class
imbalance alone.

### 5.4 Seed sensitivity — is the ceiling seed-dependent?

`train/seed_check.py` reruns the §5.1 dev-split protocol from 5 different
random seeds (seed controls both weight init and the sampler's stochastic
patch stream — the two aren't separated in the current config, noted in the
script's docstring).

| metric | mean | std | min | max |
|---|---|---|---|---|
| micro AP | 0.4665 | **0.0034** | 0.462 | 0.472 |
| micro F1* | 0.4740 | 0.0049 | 0.467 | 0.480 |
| micro ROC-AUC | 0.9543 | 0.0016 | 0.951 | 0.956 |

**Answer: no, essentially not.** Across 5 seeds on the fixed dev split, micro
AP varies by ±0.003 (< 1% relative) — negligible next to the ~0.13 std seen
across CV folds in §5.3. This isolates *where* the variance actually comes
from: **not** random initialization or sampling noise (that part is highly
reproducible), but **which cities end up in validation**. That's a
domain-generalization property of the architecture/features on this dataset,
not an optimization instability — consistent, in fact, with what the QML
side's own capacity-sweep docs report for M3: cross-city spread that persists
regardless of model capacity.

### 5.5 Evaluation Metrics — Accuracy, Change-Accuracy, No-change-Accuracy, F1

The challenge doc names four specific metrics: **Accuracy, Change accuracy,
No-change accuracy, F1**:

| | Accuracy | Change Accuracy | No-change Accuracy | F1 |
|---|---|---|---|---|
| **Pilot** (pooled, dev split) | 0.976 | 0.494 | 0.987 | 0.476 |
| **5-fold CV** (mean ± std) | 0.973 ± 0.009 | 0.350 ± 0.113 | 0.986 ± 0.005 | 0.344 ± 0.112 |
| **Seed check** (mean ± std, dev split) | 0.977 ± 0.001 | 0.476 ± 0.017 | 0.988 ± 0.001 | 0.474 ± 0.005 |

**Caveat on Accuracy specifically:** at ~2.2% true prevalence, overall
Accuracy is dominated by the majority no-change class — a trivial
"always predict no-change" classifier already scores **~97.8% Accuracy**
without detecting a single true change pixel. That's exactly why AP/F1/
Change-Accuracy were used as the headline metrics earlier (matching the QML
side's own stated priority under severe imbalance), not Accuracy — but the
challenge names Accuracy explicitly, so it's reported here in full, with this
caveat attached rather than left to read as more impressive than it is.

`train/cv.py`'s and `train/seed_check.py`'s `MICRO_KEYS` originally omitted
`accuracy`/`nochange_acc` from their own aggregate summaries too (fixed now,
for future runs) — the per-fold/per-seed `*_fullval.json` files always had
the full numbers regardless.

---

## 6. How to run

```bash
pip install -r requirements.txt
```

```bash
# model + gradient self-tests
python3 models/mlp.py
python3 train/optim.py

# train (dev 11/3 split, pilot defaults, PCA-4 representation)
python3 train/trainer.py --data_dir /path/to/raw_data \
  --representation pca --tag mlp_pca_run1

# exhaustive evaluation of a checkpoint on all 3 dev-val cities + PNG masks
python3 train/eval_full.py --data_dir /path/to/raw_data \
  --ckpt results/runs/mlp_pca_run1_bestcheap.npy --representation pca \
  --mask_dir results/runs/mlp_pca_run1_masks

# 5-fold city-grouped CV (trains 5 models, one per fold, ~75s total)
python3 train/cv.py --data_dir /path/to/raw_data --representation pca

# seed sensitivity on the fixed dev split (5 seeds, ~75s total)
python3 train/seed_check.py --data_dir /path/to/raw_data --representation pca
```

`--data_dir` should point at the `raw_data/` folder (containing `images/` and
`train_labels/`), same as the QML side expects.

Artifacts land in `results/runs/`: `<tag>.jsonl` (per-epoch log),
`<tag>_bestcheap.npy` (best pooled cheap-val AP — **use this checkpoint for
cross-model comparison**, it's the unbiased selector), `<tag>_best.npy` (best
exhaustive AP — biased when only one city is evaluated exhaustively),
`<tag>_final.npy`, `<tag>_fullval.json` (from `eval_full.py`). `cv.py` and
`seed_check.py` additionally write a `*_summary.json` with the aggregated
mean/std across folds or seeds.

---

## 7. Status / next steps

| stage | state |
|---|---|
| Architecture designed + parameter budget verified | ✅ |
| Gradient correctness verified (analytic vs. numeric) | ✅ |
| End-to-end pipeline dry-run on real data, both representations | ✅ |
| PNG mask export verified against challenge format | ✅ |
| Real training run, full protocol, exhaustive multi-city evaluation | ✅ |
| Training-length / learning-rate sensitivity checked | ✅ |
| 5-fold city-grouped CV | ✅ — mean micro AP 0.293 ± 0.129, huge per-city spread (§5.3) |
| Seed sensitivity | ✅ — negligible (std 0.003 on AP); variance is city-driven, not init-driven (§5.4) |
| Second classical baseline: 3×3 conv, 4→1, 37 params (the model designated in `QML-Binary-Segmentation/docs/model_ladder.md` as M3's literal parameter-matched twin) | ⬜ next |
| Investigate what drives the per-city AP spread (§5.3) — not explained by prevalence alone | ⬜ |
| `physical`-representation run (currently only `pca` has a real training run) | ⬜ |
| Framework choice: staying numpy-only while models remain this small; revisit PyTorch if a future architecture needs it (see §3) | noted, no action needed |

---

## Disclosure

This code and write-up were prepared with assistance from a generative AI
coding assistant (Anthropic Claude, in Claude Code). Every checkable claim
above was independently verified by actually running the referenced script
against real data or a numerical check (not asserted): the parameter counts,
the gradient correctness (finite-difference check), the pipeline dry-run, the
PNG mask format, the full training/evaluation run, and the epoch/learning-rate
sensitivity check were all executed and their console output inspected, not
merely written into this document.
