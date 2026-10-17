"""
NeuroScan -- ROC and precision-recall figure for the final model (2026-10-09). CPU; saved probabilities only (no images,
no model). Pooled internal test (cohorts A + B, predictions_test.csv of R30A s123), one-vs-rest per class, with the
operating point of the locked decision rule (argmax of p x (1, 0.574, 0.574)) marked. AUROC and average precision are
recomputed and printed so they can be checked against the final evaluation (ischemic 0.963 / 0.884, hemorrhagic 0.990 / 0.959).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

CAND = [Path("/kaggle/input/datasets/atik68/neuroscan-q-round30as123-preds"), Path("/kaggle/input/neuroscan-q-round30as123-preds")]
SRC = next(p for p in CAND if p.exists())
OUT = Path("/kaggle/working/output")
OUT.mkdir(parents=True, exist_ok=True)
LABELS = ("normal", "ischemic", "hemorrhagic")
W = np.array([1.0, 0.574, 0.574])
COL = {"normal": "#52606d", "ischemic": "#c26a1b", "hemorrhagic": "#2f6f9f"}
LS = {"normal": "-", "ischemic": "--", "hemorrhagic": "-."}

df = pd.read_csv(SRC / "predictions_test.csv")
P = df[[f"p_{l}_round30as123" for l in LABELS]].to_numpy(float)
y = df["label"].map({l: i for i, l in enumerate(LABELS)}).to_numpy()
pred = (P * W).argmax(1)
print("rows", len(df), "accuracy at locked rule", round(float((pred == y).mean()), 4), flush=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
for i, l in enumerate(LABELS):
    yb, s = (y == i).astype(int), P[:, i]
    fpr, tpr, _ = roc_curve(yb, s)
    pr, rc, _ = precision_recall_curve(yb, s)
    au, ap = roc_auc_score(yb, s), average_precision_score(yb, s)
    tp = int(((pred == i) & (yb == 1)).sum()); fp = int(((pred == i) & (yb == 0)).sum())
    fn = int(((pred != i) & (yb == 1)).sum()); tn = int(((pred != i) & (yb == 0)).sum())
    op_tpr, op_fpr, op_prec = tp / (tp + fn), fp / (fp + tn), tp / (tp + fp)
    print(f"{l}: AUROC {au:.3f}  AP {ap:.3f}  operating point recall {op_tpr:.3f} FPR {op_fpr:.3f} precision {op_prec:.3f}", flush=True)
    name = l.capitalize()
    a1.plot(fpr, tpr, LS[l], color=COL[l], lw=1.8, label=f"{name} (AUROC {au:.3f})")
    a1.scatter([op_fpr], [op_tpr], color=COL[l], s=36, zorder=3, edgecolor="white", lw=1)
    a2.plot(rc, pr, LS[l], color=COL[l], lw=1.8, label=f"{name} (AP {ap:.3f})")
    a2.scatter([op_tpr], [op_prec], color=COL[l], s=36, zorder=3, edgecolor="white", lw=1)
a1.plot([0, 1], [0, 1], color="#cbd2d9", lw=1)
a1.set_xlabel("False-positive rate (1 − specificity)")
a1.set_ylabel("True-positive rate (sensitivity)")
a2.set_xlabel("Recall (sensitivity)")
a2.set_ylabel("Precision (positive predictive value)")
a2.set_ylim(0.3, 1.01)
for a, t in ((a1, "(a) One-vs-rest ROC curves"), (a2, "(b) One-vs-rest precision–recall curves")):
    a.set_title(t, loc="left", fontsize=10)
    a.grid(True, color="#e4e7eb", lw=0.6)
    a.legend(loc="lower right" if a is a1 else "lower left", frameon=False)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
fig.text(0.5, -0.02, "Dots: operating point of the locked decision rule. Pooled internal test (cohorts A + B), 8,930 slices.",
         ha="center", fontsize=8.5, color="#52606d")
fig.tight_layout()
fig.savefig(OUT / "fig_roc_pr_final.png", dpi=220, bbox_inches="tight", facecolor="white")
print("Done.", flush=True)
