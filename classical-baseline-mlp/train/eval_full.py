"""
train/eval_full.py — exhaustive full-city evaluation of a saved MLP checkpoint
on ALL validation cities, mirroring the QML side's train/eval_full.py so the
two reports are directly comparable (same metric keys, same macro/micro
aggregation).

Reports, per city and aggregated:
  prevalence, AP, ROC-AUC, F1*, precision, ChangeAcc, NoChangeAcc, tau*
  macro : mean of per-city metrics (each city counts once)
  micro : all pixels pooled, ONE global tau* (deployment-realistic; per-city
          tau* is an optimistic upper bound, reported separately)

Also saves the challenge deliverable: a pixel-aligned binary PNG mask
({0,255}) per city, via `save_mask_png` — the exact same function the QML
side's train/inference.py already provides (reused unmodified).
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
from preprocess import build_fold
from inference import predict_city_center, evaluate_predictions, save_mask_png
import mlp as mlp_model
from trainer import build_representation

KEYS = ["prevalence", "AP", "roc_auc", "F1", "precision", "change_acc", "nochange_acc", "tau"]


def main(data_dir, ckpt, hidden, representation, out, infer_batch, mask_dir):
    params = np.load(ckpt)
    n_params = mlp_model.param_count(hidden)
    assert params.shape == (n_params,), (
        f"checkpoint has {params.shape} params but hidden={hidden} implies {n_params}")
    forward = lambda p, Xb, Sb: mlp_model.forward(p, Xb, Sb, hidden=hidden)
    print(f"MLP-pool8 (hidden={hidden}) | {representation} | {n_params} params | "
          f"ckpt {os.path.basename(ckpt)}")

    train_cities, val_cities = get_dev_split()
    fold = build_fold(train_cities, val_cities, data_dir)
    Xva, Sva = build_representation(fold, val_cities, representation)

    if mask_dir:
        os.makedirs(mask_dir, exist_ok=True)

    per_city, allp, ally = {}, [], []
    for c in val_cities:
        t = time.time()
        P = predict_city_center(forward, params, Xva[c], Sva[c], infer_batch)
        m = fold.valid[c]
        per_city[c] = evaluate_predictions(P, fold.labels[c], select_threshold=True, mask=m)
        allp.append(P[m].ravel()); ally.append(fold.labels[c][m].ravel())
        print(f"  {c:11} prev {per_city[c]['prevalence']:.4f}  AP {per_city[c]['AP']:.4f}  "
              f"ROC {per_city[c]['roc_auc']:.4f}  F1* {per_city[c]['F1']:.4f}  "
              f"prec {per_city[c]['precision']:.4f}  chAcc {per_city[c]['change_acc']:.3f}  "
              f"({time.time()-t:.1f}s)", flush=True)
        if mask_dir:
            save_mask_png(P, per_city[c]["tau"], os.path.join(mask_dir, f"{c}-cm.png"))

    macro = {k: float(np.mean([per_city[c][k] for c in val_cities])) for k in KEYS}
    micro = evaluate_predictions(np.concatenate(allp), np.concatenate(ally),
                                 select_threshold=True)
    print(f"\n  macro (per-city mean, per-city tau* = optimistic):")
    print(f"    AP {macro['AP']:.4f}  ROC {macro['roc_auc']:.4f}  F1* {macro['F1']:.4f}  "
          f"prec {macro['precision']:.4f}  chAcc {macro['change_acc']:.3f}")
    print(f"  micro (pooled pixels, ONE global tau* = deployment):")
    print(f"    AP {micro['AP']:.4f}  ROC {micro['roc_auc']:.4f}  F1* {micro['F1']:.4f}  "
          f"prec {micro['precision']:.4f}  chAcc {micro['change_acc']:.3f}  "
          f"tau {micro['tau']:.3f}  prev {micro['prevalence']:.4f}")

    res = {"model": f"MLP-pool8 h={hidden}", "n_params": n_params,
           "representation": representation, "checkpoint": os.path.basename(ckpt),
           "per_city": per_city, "macro": macro, "micro": micro}
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hidden", type=int, default=mlp_model.HIDDEN)
    ap.add_argument("--representation", default="pca")
    ap.add_argument("--infer_batch", type=int, default=4096)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mask_dir", default=None, help="if set, save {0,255} PNG masks here")
    a = ap.parse_args()
    out = a.out or a.ckpt.replace("_bestcheap.npy", "_fullval.json").replace("_best.npy", "_fullval.json")
    main(a.data_dir, a.ckpt, a.hidden, a.representation, out, a.infer_batch, a.mask_dir)
