"""
train/eval_full.py — exhaustive full-city evaluation of a saved conv
checkpoint on ALL validation cities. Mirrors classical-baseline-mlp's
eval_full.py exactly, with one substantive difference: uses `predict_city`
(dense, overlap-averaging) instead of `predict_city_center`, matching this
model's dense (B,3,3) output contract — see models/conv.py and
train/trainer.py's docstrings for why.

Reports, per city and aggregated:
  prevalence, AP, ROC-AUC, F1*, precision, ChangeAcc, NoChangeAcc, tau*
  macro : mean of per-city metrics (each city counts once)
  micro : all pixels pooled, ONE global tau* (deployment-realistic)

Also saves the challenge deliverable: a pixel-aligned binary PNG mask
({0,255}) per city, via `save_mask_png`, reused unmodified from
QML-Binary-Segmentation/train/inference.py.
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
from inference import predict_city, evaluate_predictions, save_mask_png
import conv as conv_model
from trainer import build_representation

KEYS = ["prevalence", "AP", "roc_auc", "F1", "precision", "change_acc", "nochange_acc", "tau"]


def evaluate_checkpoint(data_dir, ckpt, representation, train_cities=None,
                        val_cities=None, infer_batch=4096, mask_dir=None,
                        out=None, verbose=True):
    """Core exhaustive-evaluation logic, importable so train/cv.py and
    train/seed_check.py can reuse it per-fold/per-seed. train_cities matters
    even though only val_cities are evaluated -- build_fold() fits the
    normalization/PCA transforms on train_cities ONLY (leakage discipline
    inherited from the QML side's preprocess.py), so a CV fold's own
    train_cities must be passed, not the dev split's."""
    params = np.load(ckpt)
    n_params = conv_model.param_count()
    assert params.shape == (n_params,), (
        f"checkpoint has {params.shape} params but model expects {n_params}")
    forward = lambda p, Xb, Sb: conv_model.forward(p, Xb, Sb)
    if verbose:
        print(f"Conv3x3 (4->1) | {representation} | {n_params} params | "
              f"ckpt {os.path.basename(ckpt)}")

    if train_cities is None or val_cities is None:
        train_cities, val_cities = get_dev_split()
    fold = build_fold(train_cities, val_cities, data_dir)
    Xva, Sva = build_representation(fold, val_cities, representation)

    if mask_dir:
        os.makedirs(mask_dir, exist_ok=True)

    per_city, allp, ally = {}, [], []
    for c in val_cities:
        t = time.time()
        P = predict_city(forward, params, Xva[c], Sva[c], infer_batch)
        m = fold.valid[c]
        per_city[c] = evaluate_predictions(P, fold.labels[c], select_threshold=True, mask=m)
        allp.append(P[m].ravel()); ally.append(fold.labels[c][m].ravel())
        if verbose:
            print(f"  {c:11} prev {per_city[c]['prevalence']:.4f}  AP {per_city[c]['AP']:.4f}  "
                  f"ROC {per_city[c]['roc_auc']:.4f}  F1* {per_city[c]['F1']:.4f}  "
                  f"prec {per_city[c]['precision']:.4f}  chAcc {per_city[c]['change_acc']:.3f}  "
                  f"({time.time()-t:.1f}s)", flush=True)
        if mask_dir:
            save_mask_png(P, per_city[c]["tau"], os.path.join(mask_dir, f"{c}-cm.png"))

    macro = {k: float(np.mean([per_city[c][k] for c in val_cities])) for k in KEYS}
    micro = evaluate_predictions(np.concatenate(allp), np.concatenate(ally),
                                 select_threshold=True)
    if verbose:
        print(f"\n  macro (per-city mean, per-city tau* = optimistic):")
        print(f"    AP {macro['AP']:.4f}  ROC {macro['roc_auc']:.4f}  F1* {macro['F1']:.4f}  "
              f"prec {macro['precision']:.4f}  chAcc {macro['change_acc']:.3f}")
        print(f"  micro (pooled pixels, ONE global tau* = deployment):")
        print(f"    AP {micro['AP']:.4f}  ROC {micro['roc_auc']:.4f}  F1* {micro['F1']:.4f}  "
              f"prec {micro['precision']:.4f}  chAcc {micro['change_acc']:.3f}  "
              f"tau {micro['tau']:.3f}  prev {micro['prevalence']:.4f}")

    res = {"model": "Conv3x3 4->1", "n_params": n_params,
           "representation": representation, "checkpoint": os.path.basename(ckpt),
           "val_cities": list(val_cities), "per_city": per_city, "macro": macro, "micro": micro}
    if out:
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        if verbose:
            print(f"\nsaved {out}")
    return res


def main(data_dir, ckpt, representation, out, infer_batch, mask_dir):
    evaluate_checkpoint(data_dir, ckpt, representation,
                        infer_batch=infer_batch, mask_dir=mask_dir, out=out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--representation", default="pca")
    ap.add_argument("--infer_batch", type=int, default=4096)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mask_dir", default=None, help="if set, save {0,255} PNG masks here")
    a = ap.parse_args()
    out = a.out or a.ckpt.replace("_bestcheap.npy", "_fullval.json").replace("_best.npy", "_fullval.json")
    main(a.data_dir, a.ckpt, a.representation, out, a.infer_batch, a.mask_dir)
