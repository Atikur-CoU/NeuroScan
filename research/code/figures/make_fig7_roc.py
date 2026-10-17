"""v11 Figure 7: one-vs-rest ROC and precision-recall curves of the final model (R30A s123) on the pooled internal test
(8,930 slices), black and white, Figure 1 style. From the saved probabilities only (predictions_test.csv, as used by
kernel neuroscan-q-fig-roc-pr); operating point of the locked rule argmax(p x (1, 0.574, 0.574)). Classes are told apart
by line style and marker shape; the ROC panel has an inset magnifying the low false-positive region."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

for f in ("NotoSans-Regular.ttf", "NotoSans-Bold.ttf"):
    font_manager.fontManager.addfont(rf"C:\Windows\Fonts\{f}")
plt.rcParams.update({"font.family": ["Noto Sans", "DejaVu Sans"], "font.size": 10.5, "text.color": "black",
                     "axes.edgecolor": "black", "axes.linewidth": 0.9})
SRC = Path(r"E:\CoU Projects\NeuroScan\NeuroScan_MultiDataset\kaggle_kernels\round30as123_preds_dataset\predictions_test.csv")
OUT = Path(r"E:\CoU Projects\NeuroScan\NeuroScan_MultiDataset\paper\figures\v11_fig7_roc_pr.png")
LABELS = ("normal", "ischemic", "hemorrhagic")
W = np.array([1.0, 0.574, 0.574])
STY = {"normal": dict(ls="-", color="black", marker="o"),
       "ischemic": dict(ls=(0, (5, 2.5)), color="black", marker="s"),
       "hemorrhagic": dict(ls=(0, (1.2, 1.8)), color="black", marker="^")}

df = pd.read_csv(SRC)
P = df[[f"p_{l}_round30as123" for l in LABELS]].to_numpy(float)
y = df["label"].map({l: i for i, l in enumerate(LABELS)}).to_numpy()
pred = (P * W).argmax(1)
print("rows", len(df), "accuracy at locked rule", round(float((pred == y).mean()), 4))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.4))
ins = inset_axes(a1, width="44%", height="44%", loc="center right", borderpad=1.6)
curves = {}
for i, l in enumerate(LABELS):
    yb, s = (y == i).astype(int), P[:, i]
    fpr, tpr, _ = roc_curve(yb, s)
    pr, rc, _ = precision_recall_curve(yb, s)
    au, ap = roc_auc_score(yb, s), average_precision_score(yb, s)
    tp = int(((pred == i) & (yb == 1)).sum()); fp = int(((pred == i) & (yb == 0)).sum())
    fn = int(((pred != i) & (yb == 1)).sum()); tn = int(((pred != i) & (yb == 0)).sum())
    op = (fp / (fp + tn), tp / (tp + fn), tp / (tp + fp))
    print(f"{l}: AUROC {au:.3f} AP {ap:.3f} op recall {op[1]:.3f} FPR {op[0]:.3f} precision {op[2]:.3f}")
    st = STY[l]
    name = l.capitalize()
    for ax in (a1, ins):
        ax.plot(fpr, tpr, ls=st["ls"], color="black", lw=1.6, label=f"{name} (AUROC {au:.3f})")
        ax.plot(op[0], op[1], marker=st["marker"], ms=8 if ax is a1 else 9, mfc="white", mec="black", mew=1.6,
                ls="", zorder=4)
    a2.plot(rc, pr, ls=st["ls"], color="black", lw=1.6, label=f"{name} (AP {ap:.3f})")
    a2.plot(op[1], op[2], marker=st["marker"], ms=8, mfc="white", mec="black", mew=1.6, ls="", zorder=4)
a1.plot([0, 1], [0, 1], color="#9e9e9e", lw=0.9, ls=(0, (2, 2)))
ins.set_xlim(0, 0.2); ins.set_ylim(0.7, 1.0)
ins.set_xticks([0, 0.1, 0.2]); ins.set_yticks([0.7, 0.8, 0.9, 1.0])
ins.tick_params(labelsize=8.5)
ins.grid(True, color="#d0d0d0", lw=0.5, ls=(0, (2, 2)))
a1.add_patch(plt.Rectangle((0, 0.7), 0.2, 0.31, fill=False, ec="black", lw=0.9, ls=(0, (2, 1.5)), zorder=5))
a1.text(0.205, 0.705, "enlarged", ha="left", va="bottom", fontsize=8.5, style="italic")
ins.set_title("Enlarged: false-positive rate 0–0.2", fontsize=8.8, pad=3)
for sp in ins.spines.values():
    sp.set_linestyle((0, (2, 1.5)))

a1.set_xlabel("False-positive rate (1 − specificity)")
a1.set_ylabel("True-positive rate (sensitivity)")
a2.set_xlabel("Recall (sensitivity)")
a2.set_ylabel("Precision (positive predictive value)")
a1.set_xlim(-0.01, 1.0); a1.set_ylim(0, 1.01)
a2.set_xlim(0, 1.01); a2.set_ylim(0.3, 1.01)
for a, t in ((a1, "(a) ROC curves"), (a2, "(b) Precision–recall curves")):
    a.text(0.5, -0.2, t, transform=a.transAxes, ha="center", va="top", fontsize=12.5, fontweight="bold")
    a.grid(True, color="#d0d0d0", lw=0.5, ls=(0, (2, 2)))
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
a1.legend(loc="lower right", frameon=False, fontsize=10, handlelength=2.8)
a2.legend(loc="lower left", frameon=False, fontsize=10, handlelength=2.8)
handles = [plt.Line2D([], [], marker=STY[l]["marker"], ls="", ms=8, mfc="white", mec="black", mew=1.6,
                      label=f"{l.capitalize()} operating point") for l in LABELS]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.03))
fig.subplots_adjust(wspace=0.22, bottom=0.2, top=0.9)
fig.savefig(OUT, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.1)
print(OUT)


