"""
train/cv.py — 5-fold city-grouped cross-validation for the pooled-8 MLP.

WHY THIS IS NEEDED
--------------------
README.md §5.1 reports one result from the fixed dev 11/3 split (val = paris,
cupertino, beihai) — and that split showed a very wide spread already (AP
0.211 on paris vs 0.627 on cupertino). A single split's pooled number is not
a reliable estimate of how the model actually generalizes: which 3 cities
happen to land in validation could make the reported AP look better or worse
than the truth, by luck alone. `splits.py::get_grouped_folds(n_splits=5)`
partitions the 14 labelled cities into 5 groups; training 5 separate models
(each holding out a different group) and aggregating means every city is
evaluated exactly once as a held-out city, and the fold-to-fold spread
becomes visible instead of hidden.

WHAT THIS SCRIPT DOES
------------------------
For each of the 5 folds:
  1. train the MLP from scratch on that fold's ~11 train cities (reusing
     trainer.run(), now parameterized to accept train_cities/val_cities —
     see trainer.py's docstring on that change);
  2. exhaustively evaluate the fold's best-cheap-AP checkpoint on that fold's
     held-out cities (reusing eval_full.py::evaluate_checkpoint(), same
     reason — it needs the FOLD's own train_cities to fit the
     normalization/PCA transforms correctly, not the dev split's).
Then aggregates: per-fold micro metrics (mean +/- std across the 5 folds) and
a full 14-city table (every labelled city, since each appears in exactly one
fold's validation set).

`exhaustive_cities="none"` during training (the periodic in-training
exhaustive check is skipped — this script's own post-hoc
`evaluate_checkpoint` call is the authoritative one, exactly as for the
single-split pilot run), which keeps each fold's training fast.
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
QML_ROOT = os.path.normpath(os.path.join(ROOT, "..", "QML-Binary-Segmentation"))
sys.path.insert(0, os.path.join(QML_ROOT, "data"))
sys.path.insert(0, os.path.join(QML_ROOT, "train"))
sys.path.insert(0, os.path.join(ROOT, "models"))
sys.path.insert(0, HERE)

from splits import get_grouped_folds
from trainer import TrainConfig, run as train_run
from eval_full import evaluate_checkpoint

MICRO_KEYS = ["AP", "roc_auc", "F1", "precision", "change_acc", "prevalence"]


def main(data_dir, representation, hidden, n_splits, seed, epochs, out_dir, out):
    folds = get_grouped_folds(n_splits=n_splits, seed=seed)
    print(f"=== 5-fold city-grouped CV | {representation} | hidden={hidden} | "
          f"{n_splits} folds, split-seed={seed} ===\n")
    for i, (tr, va) in enumerate(folds):
        print(f"  fold {i}: val={va}")
    print()

    fold_results = []
    t0 = time.time()
    for i, (train_cities, val_cities) in enumerate(folds):
        tag = f"cv{n_splits}_fold{i}_{representation}"
        print(f"--- fold {i+1}/{n_splits}  train={len(train_cities)} cities  "
              f"val={val_cities} ---")
        cfg = TrainConfig(hidden=hidden, representation=representation, epochs=epochs,
                          exhaustive_cities="none", tag=tag, seed=seed, out_dir=out_dir)
        train_run(cfg, data_dir, train_cities=train_cities, val_cities=val_cities)

        ckpt = os.path.join(out_dir, f"{tag}_bestcheap.npy")
        res = evaluate_checkpoint(data_dir, ckpt, hidden, representation,
                                  train_cities=train_cities, val_cities=val_cities,
                                  out=os.path.join(out_dir, f"{tag}_fullval.json"))
        fold_results.append({"fold": i, "val_cities": val_cities,
                             "micro": res["micro"], "per_city": res["per_city"]})
        print()

    # ---- aggregate across folds ----
    micro_by_fold = {k: np.array([fr["micro"][k] for fr in fold_results]) for k in MICRO_KEYS}
    summary = {k: {"mean": float(micro_by_fold[k].mean()), "std": float(micro_by_fold[k].std()),
                   "per_fold": micro_by_fold[k].tolist()} for k in MICRO_KEYS}

    # every one of the 14 labelled cities appears in exactly one fold's val set
    all_cities = {}
    for fr in fold_results:
        all_cities.update(fr["per_city"])

    print("=" * 70)
    print(f"5-FOLD CV SUMMARY ({representation}, hidden={hidden}, {time.time()-t0:.0f}s total)")
    print("=" * 70)
    print(f"{'metric':12} {'mean':>8} {'std':>8}   per-fold")
    for k in MICRO_KEYS:
        s = summary[k]
        per_fold_str = " ".join(f"{v:.3f}" for v in s["per_fold"])
        print(f"{k:12} {s['mean']:8.4f} {s['std']:8.4f}   [{per_fold_str}]")

    print(f"\nall {len(all_cities)}/14 labelled cities (each evaluated once, as its fold's held-out city):")
    for c in sorted(all_cities, key=lambda c: -all_cities[c]["AP"]):
        m = all_cities[c]
        print(f"  {c:12} prev {m['prevalence']:.4f}  AP {m['AP']:.4f}  "
              f"F1* {m['F1']:.4f}  ROC {m['roc_auc']:.4f}  chAcc {m['change_acc']:.3f}")

    result = {"representation": representation, "hidden": hidden, "n_splits": n_splits,
              "split_seed": seed, "epochs": epochs, "micro_summary": summary,
              "all_cities": all_cities, "folds": [{"fold": fr["fold"], "val_cities": fr["val_cities"]}
                                                   for fr in fold_results]}
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nsaved {out}")
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--representation", default="pca")
    ap.add_argument("--hidden", type=int, default=3)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "results", "runs"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    out = a.out or os.path.join(a.out_dir, f"cv{a.n_splits}_{a.representation}_summary.json")
    main(a.data_dir, a.representation, a.hidden, a.n_splits, a.seed, a.epochs, a.out_dir, out)
