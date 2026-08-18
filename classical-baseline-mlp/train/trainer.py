"""
train/trainer.py — training loop for the pooled-8 MLP classical baseline.

DATA PIPELINE REUSE (deliberate design decision, see README.md)
------------------------------------------------------------------
This imports data/{splits,preprocess,pools,sampler}.py and train/inference.py
DIRECTLY from the sibling QML-Binary-Segmentation repo (via sys.path), rather
than re-implementing an equivalent pipeline here. Reasoning: the challenge
requires the classical model to see the SAME input features as the QML model;
re-deriving the preprocessing independently risks a silent mismatch (a
slightly different percentile, a different median-correction detail, ...)
that would quietly invalidate the "same features" comparison. Importing the
exact same functions makes that impossible by construction. The only things
NOT reused from the QML side are the model itself (models/mlp.py, this
directory) and the optimizer (train/optim.py, plain Adam, no PennyLane
dependency) — everything quantum-specific stays out of this directory.

`inference.py`'s functions are model-agnostic (they call `forward(params,
X, S) -> P` without caring what's inside `forward`), so `predict_city_center`,
`predict_coordinates`, `evaluate_predictions`, `make_fixed_val_coordinates`
work unmodified for this MLP. NOTE: the QML side's own `train/trainer.py` is
NOT imported (only its sibling `data/` and `train/inference.py` modules are) —
importing it would pull in `import pennylane as qml` as a side effect, which
this directory intentionally avoids.

TRAINING PROTOCOL — matched to the QML side's pilot defaults where possible
------------------------------------------------------------------------------
- dev 11/3 city split (`get_dev_split`), same as the QML pilot runs.
- Same city-balanced 1:1:2 (positive:hard-negative:ordinary) sampler.
- "epoch" = steps_per_epoch * batch patches (default 160*32 = 5120), same
  convention as the QML trainer, since the sampler is an infinite stream.
- Two validation tiers, same rationale as the QML side: cheap (every epoch,
  fixed natural-prevalence coordinates, fast) vs exhaustive (periodic,
  stride-1 whole-city, authoritative).
- Two checkpoints: `*_bestcheap.npy` (unbiased, pooled-cheap-AP selector —
  use this one for cross-model comparison) and `*_best.npy` (exhaustive-AP,
  biased when only one city is evaluated exhaustively).
- Default lr=0.02, batch=32, steps_per_epoch=160, epochs=20 — copied from the
  QML pilot defaults for comparability, NOT independently tuned. Revisit
  after the first real learning curve, exactly as the QML side's own
  trainer.py docstring cautions for itself.

OUTPUT CONTRACT
-----------------
The MLP predicts the CENTRE pixel of each 3x3 patch (see models/mlp.py) —
this is the QML ladder's "center_mean" readout mode, so `predict_city_center`
is used for exhaustive full-city prediction (no overlap-averaging needed:
each patch yields exactly one (row,col) prediction, its own centre).
"""
import os, sys, json, time
from dataclasses import dataclass, asdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                          # classical-baseline-mlp/
QML_ROOT = os.path.normpath(os.path.join(ROOT, "..", "QML-Binary-Segmentation"))

sys.path.insert(0, os.path.join(QML_ROOT, "data"))       # splits, preprocess, pools, sampler
sys.path.insert(0, os.path.join(QML_ROOT, "train"))       # inference.py ONLY (model-agnostic)
sys.path.insert(0, os.path.join(ROOT, "models"))          # our mlp.py
sys.path.insert(0, HERE)                                  # our optim.py

from splits import get_dev_split
from preprocess import build_fold, transform_pca4, transform_physical4, pca_zz_strength, physical_zz_strength
from pools import build_center_pools, fit_global_hard_threshold, eligible_mask
from sampler import SpatialPatchSampler
from inference import predict_city_center, predict_coordinates, evaluate_predictions, make_fixed_val_coordinates

import mlp as mlp_model
from optim import Adam


@dataclass
class TrainConfig:
    # model
    hidden: int = mlp_model.HIDDEN        # 3 -> 31 params (see models/mlp.py)
    representation: str = "pca"           # "pca" | "physical"
    # optimization (matched to the QML pilot defaults, see module docstring)
    lr: float = 0.02
    batch: int = 32
    steps_per_epoch: int = 160
    epochs: int = 20
    w_pos: float = 1.0                    # 1.0 = plain BCE (sampler already ~78:22)
    # validation
    cheap_val_per_city: int = 3000
    exhaustive_every: int = 5
    exhaustive_cities: str = "smallest"   # "smallest" | "all" | "none"
    infer_batch: int = 4096
    # misc
    seed: int = 0
    tag: str = ""
    out_dir: str = os.path.join(ROOT, "results", "runs")


def build_representation(fold, cities, representation):
    """city -> (X[H,W,4], S[H,W] or [H,W,2]). Identical logic to the QML
    side's train/trainer.py::build_representation (duplicated here, not
    imported, to avoid pulling in that module's `import pennylane` side
    effect — see module docstring). S is computed for interface parity with
    predict_city_center/predict_coordinates even though mlp.forward ignores
    it."""
    X, S = {}, {}
    for c in cities:
        D = fold.dcorr13[c]
        if representation == "pca":
            X[c] = transform_pca4(D, fold.pca_tf).astype(np.float64)
            S[c] = pca_zz_strength(D, fold.pca_tf).astype(np.float64)
        else:
            x4 = transform_physical4(D, fold.physical_tf).astype(np.float64)
            s1, s2 = physical_zz_strength(x4)
            X[c], S[c] = x4, np.stack([s1, s2], -1)
    return X, S


def make_batch(smp, X, S, labels, B, rng):
    """Sample B (3x3x4) patches + their CENTRE label (center_mean target)."""
    Xb, Sb, Yb = [], [], []
    for _ in range(B):
        c, _, r, cc = smp.sample_index(rng)
        Xb.append(X[c][r - 1:r + 2, cc - 1:cc + 2])
        Sb.append(S[c][r - 1:r + 2, cc - 1:cc + 2])
        Yb.append(labels[c][r, cc])
    return np.array(Xb), np.array(Sb), np.array(Yb, dtype=float)


def run(cfg, data_dir, train_cities=None, val_cities=None):
    """train_cities/val_cities default to the fixed dev 11/3 split
    (get_dev_split()) when not supplied — this keeps every existing CLI call
    and the pilot run's behaviour unchanged. Passing them explicitly is what
    lets train/cv.py reuse this exact function for each of the 5-fold
    city-grouped CV splits, instead of duplicating the training loop."""
    os.makedirs(cfg.out_dir, exist_ok=True)
    n_params = mlp_model.param_count(cfg.hidden)
    tag = cfg.tag or f"mlp_pool8_h{cfg.hidden}_{cfg.representation}"
    log_path = os.path.join(cfg.out_dir, f"{tag}.jsonl")
    print(f"=== MLP-pool8 (hidden={cfg.hidden}) | {cfg.representation} | "
          f"{n_params} params (QML M3 budget: 38) -> {log_path}")
    assert n_params <= 38, f"{n_params} params exceeds the 38-param QML budget"

    # ---- data (byte-identical pipeline to the QML side) ----
    if train_cities is None or val_cities is None:
        train_cities, val_cities = get_dev_split()
    fold = build_fold(train_cities, val_cities, data_dir)
    T_global = fit_global_hard_threshold(train_cities, fold.dcorr13, fold.labels, fold.valid)
    pools = {c: build_center_pools(fold.dcorr13[c], fold.labels[c], fold.valid[c], T_global)
             for c in train_cities}
    smp = SpatialPatchSampler(train_cities, pools, fold, cfg.representation, seed=cfg.seed)
    Xtr, Str = build_representation(fold, train_cities, cfg.representation)
    Xva, Sva = build_representation(fold, val_cities, cfg.representation)

    val_coords, val_y = {}, {}
    print(f"\ncheap-val ({cfg.cheap_val_per_city}/city, uniform over eligible):")
    for c in val_cities:
        co = make_fixed_val_coordinates(fold.labels[c], fold.valid[c],
                                        n=cfg.cheap_val_per_city, seed=cfg.seed + 1)
        val_coords[c] = co
        val_y[c] = fold.labels[c][co[:, 0], co[:, 1]].astype(int)
        el = eligible_mask(fold.valid[c])
        print(f"  {c:11} sampled prevalence {val_y[c].mean():.4f}   "
              f"city eligible {fold.labels[c][el].mean():.4f}")

    if cfg.exhaustive_cities == "all":
        ex_cities = list(val_cities)
    elif cfg.exhaustive_cities == "none":
        ex_cities = []
    else:
        ex_cities = [min(val_cities, key=lambda c: fold.labels[c].size)]
    print(f"exhaustive val on: {ex_cities or '(disabled)'}\n")

    # ---- model / optimizer ----
    params = mlp_model.init_params(seed=cfg.seed, hidden=cfg.hidden)
    opt = Adam(params, lr=cfg.lr)
    rng = np.random.RandomState(cfg.seed)

    def forward(p, Xb, Sb):
        return mlp_model.forward(p, Xb, Sb, hidden=cfg.hidden)

    def cheap_val(p):
        ps, ys = [], []
        for c in val_cities:
            ps.append(predict_coordinates(forward, p, Xva[c], Sva[c],
                                          val_coords[c], cfg.infer_batch))
            ys.append(val_y[c])
        return evaluate_predictions(np.concatenate(ps), np.concatenate(ys),
                                    select_threshold=True)

    def exhaustive_val(p):
        per_city, allp, ally = {}, [], []
        for c in ex_cities:
            P = predict_city_center(forward, p, Xva[c], Sva[c], cfg.infer_batch)
            m = fold.valid[c]
            per_city[c] = evaluate_predictions(P, fold.labels[c], select_threshold=True, mask=m)
            allp.append(P[m].ravel()); ally.append(fold.labels[c][m].ravel())
        pooled = evaluate_predictions(np.concatenate(allp), np.concatenate(ally),
                                      select_threshold=True) if allp else {}
        return per_city, pooled

    best = {"exhaustive_AP": -1.0, "epoch": -1}
    best_cheap = {"cheap_AP": -1.0, "epoch": -1}
    t0 = time.time()
    with open(log_path, "w") as f:
        f.write(json.dumps({"config": asdict(cfg), "n_params": n_params,
                            "model": "mlp_pool8", "train_cities": train_cities,
                            "val_cities": val_cities}) + "\n")

        for epoch in range(1, cfg.epochs + 1):
            losses = []
            for _ in range(cfg.steps_per_epoch):
                Xb, Sb, Yb = make_batch(smp, Xtr, Str, fold.labels, cfg.batch, rng)
                loss, grad = mlp_model.bce_loss_and_grad(params, Xb, Yb,
                                                         hidden=cfg.hidden, w_pos=cfg.w_pos)
                params = opt.step(params, grad)
                losses.append(loss)
            cv = cheap_val(params)
            rec = {"epoch": epoch, "train_BCE": float(np.mean(losses)),
                   "cheap_AP": cv["AP"], "cheap_best_F1": cv["F1"], "cheap_tau": cv["tau"],
                   "cheap_change_acc": cv["change_acc"],
                   "grad_norm": float(np.linalg.norm(grad)),
                   "param_norm": float(np.linalg.norm(params)),
                   "lr": cfg.lr, "wall_time": time.time() - t0}
            if cv["AP"] > best_cheap["cheap_AP"]:
                best_cheap = {"cheap_AP": cv["AP"], "epoch": epoch, "F1": cv["F1"]}
                np.save(os.path.join(cfg.out_dir, f"{tag}_bestcheap.npy"), params)
            print(f"ep {epoch:3d}  BCE {rec['train_BCE']:.4f}  cheapAP {cv['AP']:.4f}  "
                  f"F1 {cv['F1']:.4f}  |g| {rec['grad_norm']:.3e}  {rec['wall_time']:.0f}s",
                  flush=True)

            if ex_cities and (epoch % cfg.exhaustive_every == 0 or epoch == cfg.epochs):
                te = time.time()
                per_city, pooled = exhaustive_val(params)
                rec["exhaustive"] = {"per_city": per_city, "pooled": pooled,
                                     "seconds": time.time() - te}
                print(f"      exhaustive: " + "  ".join(
                    f"{c}: AP {m['AP']:.4f} F1 {m['F1']:.4f} chAcc {m['change_acc']:.3f}"
                    for c, m in per_city.items()) + f"   ({time.time()-te:.0f}s)", flush=True)
                if pooled and pooled["AP"] > best["exhaustive_AP"]:
                    best = {"exhaustive_AP": pooled["AP"], "epoch": epoch,
                            "tau": pooled["tau"], "F1": pooled["F1"]}
                    np.save(os.path.join(cfg.out_dir, f"{tag}_best.npy"), params)
            f.write(json.dumps(rec) + "\n"); f.flush()

        np.save(os.path.join(cfg.out_dir, f"{tag}_final.npy"), params)
        f.write(json.dumps({"best_exhaustive": best, "best_cheap": best_cheap,
                            "final_epoch": cfg.epochs}) + "\n")
    print(f"\nbest exhaustive AP: {best}\nbest cheap AP    : {best_cheap}")
    return {"exhaustive": best, "cheap": best_cheap}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--representation", type=str, default=None)
    ap.add_argument("--exhaustive_cities", type=str, default=None)
    ap.add_argument("--tag", type=str, default=None)
    for fld in ("hidden", "batch", "steps_per_epoch", "epochs",
                "cheap_val_per_city", "exhaustive_every", "seed", "infer_batch"):
        ap.add_argument(f"--{fld}", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    a = ap.parse_args()
    cfg = TrainConfig(**{k: v for k, v in vars(a).items()
                         if k != "data_dir" and v is not None})
    run(cfg, a.data_dir)
