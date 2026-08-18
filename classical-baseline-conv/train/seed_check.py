"""
train/seed_check.py — multi-seed sensitivity check on the FIXED dev 11/3
split, for the 3x3 conv model. Port of classical-baseline-mlp's version (see
that file's docstring) — the MLP's ceiling turned out to be essentially
seed-independent (std 0.003 on AP) while being highly split-dependent (std
0.13 across CV folds); this checks whether the same pattern holds for a
structurally different (linear, full-spatial-detail) classical model.

Note: `seed` controls BOTH the parameter initialization (`init_params`) AND
the sampler's stochastic patch draws (`SpatialPatchSampler(..., seed=cfg.seed)`)
-- a joint init+sampling-stream check, not a pure weight-init ablation, same
caveat as the MLP version.
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

from splits import get_dev_split
from trainer import TrainConfig, run as train_run
from eval_full import evaluate_checkpoint

MICRO_KEYS = ["AP", "roc_auc", "F1", "accuracy", "change_acc", "nochange_acc", "precision"]
# NOTE: "accuracy"/"nochange_acc" were added after the fact -- see
# classical-baseline-mlp/train/cv.py's identical note and README.md's
# "Challenge-required metrics" section.


def main(data_dir, representation, seeds, epochs, out_dir, out):
    train_cities, val_cities = get_dev_split()
    print(f"=== seed sensitivity | Conv3x3 | {representation} | "
          f"dev split (val={val_cities}) | seeds={seeds} ===\n")

    results = []
    t0 = time.time()
    for s in seeds:
        tag = f"seedcheck_s{s}_{representation}"
        print(f"--- seed {s} ---")
        cfg = TrainConfig(representation=representation, epochs=epochs,
                          exhaustive_cities="none", tag=tag, seed=s, out_dir=out_dir)
        train_run(cfg, data_dir, train_cities=train_cities, val_cities=val_cities)

        ckpt = os.path.join(out_dir, f"{tag}_bestcheap.npy")
        res = evaluate_checkpoint(data_dir, ckpt, representation,
                                  train_cities=train_cities, val_cities=val_cities,
                                  out=os.path.join(out_dir, f"{tag}_fullval.json"))
        results.append({"seed": s, "micro": res["micro"]})
        print()

    micro_by_seed = {k: np.array([r["micro"][k] for r in results]) for k in MICRO_KEYS}
    summary = {k: {"mean": float(micro_by_seed[k].mean()), "std": float(micro_by_seed[k].std()),
                   "min": float(micro_by_seed[k].min()), "max": float(micro_by_seed[k].max()),
                   "per_seed": micro_by_seed[k].tolist()} for k in MICRO_KEYS}

    print("=" * 70)
    print(f"SEED SENSITIVITY SUMMARY (Conv3x3, {representation}, "
          f"{time.time()-t0:.0f}s total)")
    print("=" * 70)
    print(f"{'metric':12} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}   per-seed")
    for k in MICRO_KEYS:
        s = summary[k]
        per_seed_str = " ".join(f"{v:.3f}" for v in s["per_seed"])
        print(f"{k:12} {s['mean']:8.4f} {s['std']:8.4f} {s['min']:8.4f} {s['max']:8.4f}   "
              f"[{per_seed_str}]")

    result = {"representation": representation, "seeds": seeds, "epochs": epochs,
              "val_cities": val_cities, "micro_summary": summary}
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nsaved {out}")
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--representation", default="pca")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "results", "runs"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    out = a.out or os.path.join(a.out_dir, f"seedcheck_{a.representation}_summary.json")
    main(a.data_dir, a.representation, a.seeds, a.epochs, a.out_dir, out)
