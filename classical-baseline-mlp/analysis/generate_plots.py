"""
analysis/generate_plots.py — confusion matrix, PR/ROC curves, qualitative
prediction maps, and the 5-fold CV per-city bar chart, for the pooled-8 MLP.

Design notes (why these forms, why this palette):
  - Confusion matrix: a 2x2 heatmap is a MAGNITUDE job (cell = a count) ->
    one sequential hue, light-to-dark, not a rainbow. Counts AND their share
    of the pooled total are both shown (a raw count alone hides how skewed
    the classes are; this task is ~2% positive).
  - PR and ROC curves: two separate single-axis panels, never one dual-axis
    chart (recall/precision and FPR/TPR are different scales; overlaying them
    on one plot would be exactly the "two y-axes" anti-pattern). Each has one
    neutral-gray dashed reference line (prevalence for PR, the diagonal for
    ROC) so the curve's actual lift over "no skill" is visible, not just its
    shape in isolation.
  - Qualitative predictions: rather than two separate binary masks (ground
    truth vs. prediction) that force the viewer to diff them by eye, the
    prediction panel is a 4-category OUTCOME overlay (TP/FP/FN/TN) -- an
    IDENTITY job, so a small fixed categorical palette with a legend, chosen
    for double redundancy (distinct hue AND distinct lightness per category,
    not relying on hue alone): TP blue (correct), FN amber (missed), FP
    magenta (false alarm), TN near-white (correct background).
  - CV bar chart: one series (AP per city) -> ONE flat color, no
    color-encodes-the-value redundancy, direct end-of-bar labels instead of
    a legend, sorted by value (not city name) so the pattern is legible.

Regenerates predictions on the fly from the saved pilot/CV artifacts (full-
city inference is ~0.2-1.5s per city for this model) rather than serializing
large prediction arrays into the repo -- consistent with the
"regenerable, don't commit" convention already used for weight checkpoints.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
QML_ROOT = os.path.normpath(os.path.join(ROOT, "..", "QML-Binary-Segmentation"))
sys.path.insert(0, os.path.join(QML_ROOT, "data"))
sys.path.insert(0, os.path.join(QML_ROOT, "train"))
sys.path.insert(0, os.path.join(ROOT, "models"))
sys.path.insert(0, os.path.join(ROOT, "train"))

from splits import get_dev_split
from preprocess import build_fold
from inference import predict_city_center, evaluate_predictions
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score
import mlp as mlp_model
from trainer import build_representation

RUNS = os.path.join(ROOT, "results", "runs")
PLOTS = os.path.join(ROOT, "results", "plots")
os.makedirs(PLOTS, exist_ok=True)

# -- fixed palette (see module docstring for the reasoning behind each choice) --
BLUE = "#2563eb"      # primary accent: model output / true positive
AMBER = "#d97706"      # false negative (missed change)
MAGENTA = "#db2777"    # false positive (false alarm)
NEARWHITE = "#f3f4f6"  # true negative (correct background)
GRAY = "#9ca3af"       # neutral reference lines / muted ink
INK = "#111827"        # primary text
SEQ_CMAP = "Blues"     # sequential magnitude colormap (confusion matrix)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRAY, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})


def get_pooled_pilot_predictions():
    """Rebuild the dev split, run the pilot bestcheap checkpoint on all 3 val
    cities, return pooled (P, Y) plus tau* and per-city P/Y/mask for the
    qualitative panel."""
    train_cities, val_cities = get_dev_split()
    fold = build_fold(train_cities, val_cities, DATA_DIR)
    Xva, Sva = build_representation(fold, val_cities, "pca")
    params = np.load(os.path.join(RUNS, "mlp_pca_pilot_bestcheap.npy"))
    forward = lambda p, Xb, Sb: mlp_model.forward(p, Xb, Sb)

    per_city = {}
    allp, ally = [], []
    for c in val_cities:
        P = predict_city_center(forward, params, Xva[c], Sva[c])
        Y = fold.labels[c]
        m = fold.valid[c]
        per_city[c] = {"P": P, "Y": Y, "mask": m}
        allp.append(P[m].ravel()); ally.append(Y[m].ravel())
    P_pool, Y_pool = np.concatenate(allp), np.concatenate(ally).astype(int)
    metrics = evaluate_predictions(P_pool, Y_pool, select_threshold=True)
    return P_pool, Y_pool, metrics["tau"], per_city, val_cities


def plot_confusion_matrix(P, Y, tau):
    yhat = (P >= tau).astype(int)
    tp = int(((yhat == 1) & (Y == 1)).sum()); fp = int(((yhat == 1) & (Y == 0)).sum())
    fn = int(((yhat == 0) & (Y == 1)).sum()); tn = int(((yhat == 0) & (Y == 0)).sum())
    total = tp + fp + fn + tn
    mat = np.array([[tn, fp], [fn, tp]])  # rows: actual [no-change, change]; cols: pred [no-change, change]

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    im = ax.imshow(mat, cmap=SEQ_CMAP, aspect="equal")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No change", "Change"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["No change", "Change"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.suptitle(f"MLP — pooled confusion matrix (dev split, τ*={tau:.3f})", fontsize=11, x=0.46)
    vmax = mat.max()
    for i in range(2):
        for j in range(2):
            val = mat[i, j]
            color = "white" if val > vmax * 0.6 else INK
            ax.text(j, i, f"{val:,}\n({100*val/total:.1f}%)", ha="center", va="center",
                    color=color, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8, label="pixel count")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print(f"  tp={tp} fp={fp} fn={fn} tn={tn}  precision={tp/(tp+fp):.3f}  "
          f"recall={tp/(tp+fn):.3f}  saved confusion_matrix.png")


def plot_pr_roc(P, Y):
    prevalence = Y.mean()
    prec, rec, _ = precision_recall_curve(Y, P)
    fpr, tpr, _ = roc_curve(Y, P)
    ap = average_precision_score(Y, P)
    auc = roc_auc_score(Y, P)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    ax = axes[0]
    ax.plot(rec, prec, color=BLUE, lw=2)
    ax.axhline(prevalence, color=GRAY, lw=1.5, ls="--", label=f"no-skill (prevalence={prevalence:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall  (AP = {ap:.3f})", fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    ax = axes[1]
    ax.plot(fpr, tpr, color=BLUE, lw=2)
    ax.plot([0, 1], [0, 1], color=GRAY, lw=1.5, ls="--", label="no-skill (diagonal)")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC  (AUC = {auc:.3f})", fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    fig.suptitle("MLP — pooled dev-split curves", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "pr_roc_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  AP={ap:.4f}  ROC-AUC={auc:.4f}  saved pr_roc_curves.png")


def _load_preview(city, which):
    p = os.path.join(DATA_DIR, "images", "Onera Satellite Change Detection dataset - Images",
                     city, "pair", f"img{which}.png")
    return np.array(Image.open(p))


def plot_qualitative(per_city, tau, val_cities):
    # pick the best- and worst-AP val cities on the pilot split for contrast
    aps = {c: evaluate_predictions(per_city[c]["P"], per_city[c]["Y"], tau=tau,
                                   mask=per_city[c]["mask"])["AP"] for c in val_cities}
    best_c = max(aps, key=aps.get); worst_c = min(aps, key=aps.get)
    cities = [best_c, worst_c]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2))
    legend_handles = [Patch(color=BLUE, label="TP (correct change)"),
                      Patch(color=AMBER, label="FN (missed change)"),
                      Patch(color=MAGENTA, label="FP (false alarm)"),
                      Patch(color=NEARWHITE, ec=GRAY, label="TN (correct background)")]

    for row, city in enumerate(cities):
        P, Y, m = per_city[city]["P"], per_city[city]["Y"], per_city[city]["mask"]
        yhat = (P >= tau).astype(int)
        outcome = np.zeros((*Y.shape, 3))
        outcome[:] = np.array([1, 1, 1])  # default white for invalid/no-data
        colors = {(1, 1): BLUE, (0, 1): AMBER, (1, 0): MAGENTA, (0, 0): NEARWHITE}
        for (yh, y), hexcol in colors.items():
            rgb = np.array(matplotlib.colors.to_rgb(hexcol))
            outcome[(yhat == yh) & (Y == y) & m] = rgb

        t1 = _load_preview(city, 1); t2 = _load_preview(city, 2)
        gt = np.where(Y[..., None].astype(bool), matplotlib.colors.to_rgb(BLUE),
                      matplotlib.colors.to_rgb(NEARWHITE))

        tag = "best" if city == best_c else "worst"
        axes[row, 0].imshow(t1); axes[row, 0].set_ylabel(f"{city} ({tag}, AP={aps[city]:.3f})",
                                                          fontsize=10)
        axes[row, 1].imshow(t2)
        axes[row, 2].imshow(gt)
        axes[row, 3].imshow(outcome)
        if row == 0:
            for ax, title in zip(axes[row], ["T1", "T2", "Ground truth", "Prediction outcome"]):
                ax.set_title(title, fontsize=11)
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])

    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False,
              bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.suptitle("MLP — qualitative predictions, dev-split best vs. worst city", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(os.path.join(PLOTS, "qualitative_predictions.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  best={best_c} (AP={aps[best_c]:.3f})  worst={worst_c} (AP={aps[worst_c]:.3f})  "
          f"saved qualitative_predictions.png")


def plot_cv_bar_chart():
    with open(os.path.join(RUNS, "cv5_pca_summary.json")) as f:
        d = json.load(f)
    cities = d["all_cities"]
    items = sorted(cities.items(), key=lambda kv: kv[1]["AP"])
    names = [k for k, _ in items]; aps = [v["AP"] for _, v in items]
    prevs = [v["prevalence"] for _, v in items]
    mean_ap = d["micro_summary"]["AP"]["mean"]

    fig, ax = plt.subplots(figsize=(8, 7))
    y = np.arange(len(names))
    ax.barh(y, aps, color=BLUE, height=0.65)
    ax.axvline(mean_ap, color=GRAY, lw=1.5, ls="--", label=f"CV mean AP = {mean_ap:.3f}")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("AP (5-fold CV, each city evaluated once as its fold's held-out city)")
    ax.set_xlim(0, max(aps) * 1.25)
    for yi, (ap, prev) in enumerate(zip(aps, prevs)):
        ax.text(ap + 0.01, yi, f"{ap:.3f}  (prev {100*prev:.1f}%)", va="center", fontsize=8, color=INK)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_title("MLP — per-city AP, 5-fold city-grouped CV (all 14 labelled cities)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "cv_per_city_ap.png"), dpi=150)
    plt.close(fig)
    print(f"  CV mean AP={mean_ap:.4f}  saved cv_per_city_ap.png")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    a = ap.parse_args()
    DATA_DIR = a.data_dir

    print("Regenerating pilot predictions (dev split, 3 val cities)...")
    P, Y, tau, per_city, val_cities = get_pooled_pilot_predictions()

    print("\n[1/4] confusion matrix")
    plot_confusion_matrix(P, Y, tau)
    print("\n[2/4] PR / ROC curves")
    plot_pr_roc(P, Y)
    print("\n[3/4] qualitative predictions")
    plot_qualitative(per_city, tau, val_cities)
    print("\n[4/4] CV per-city AP bar chart")
    plot_cv_bar_chart()

    print(f"\nAll figures saved to {PLOTS}")
