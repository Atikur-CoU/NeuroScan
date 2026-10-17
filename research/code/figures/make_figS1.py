"""Supplementary Figure S1: reliability diagrams of the final model (R30A s123) before and after the locked modality
temperatures (T_CT 1.48, T_MRI 0.80; p ~ p_raw^(1/T)), black and white, paper style. From the saved probabilities
only, with the final evaluation's own definitions (15 equal-width bins; bins with >= 10 slices plotted; top-label ECE).
The ECE values are recomputed and must match Supplementary Table S8."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

for f in ("NotoSans-Regular.ttf", "NotoSans-Bold.ttf"):
    font_manager.fontManager.addfont(rf"C:\Windows\Fonts\{f}")
plt.rcParams.update({"font.family": ["Noto Sans", "DejaVu Sans"], "font.size": 10.5, "text.color": "black",
                     "axes.edgecolor": "black", "axes.linewidth": 0.9})
D = Path(r"E:\CoU Projects\NeuroScan\NeuroScan_MultiDataset\kaggle_kernels\round30as123_preds_dataset")
OUT = Path(r"E:\CoU Projects\NeuroScan\NeuroScan_MultiDataset\paper\figures\v11_figS1_reliability.png")
LABELS = ("normal", "ischemic", "hemorrhagic")
L2I = {l: i for i, l in enumerate(LABELS)}
MRI = ("ISLES2022_dwi_adc", "SOOP_dwi_adc")
T_CT, T_MRI, NB = 1.48, 0.80, 15
PC = [f"p_{l}_round30as123" for l in LABELS]
test = pd.read_csv(D / "predictions_test.csv")
soop = pd.read_csv(D / "predictions_soop_external.csv")
PANELS = [("(a) Pooled internal test, CT", test[~test["dataset"].isin(MRI)], (5.0, 2.0)),
          ("(b) SOOP cohort, MRI", soop, (4.3, 2.5)),
          ("(c) CPAISD, CT", test[test["dataset"] == "CPAISD_v2"], (9.7, 7.9))]
EDGES = np.linspace(0, 1, NB + 1)


def probs(g):
    p = g[PC].to_numpy(float).clip(1e-12, 1)
    raw = p / p.sum(1, keepdims=True)
    t = np.where(g["dataset"].isin(MRI), T_MRI, T_CT)[:, None]
    q = p ** (1 / t)
    return raw, q / q.sum(1, keepdims=True)


def ece_and_curve(y, p):
    conf, hit = p.max(1), (p.argmax(1) == y).astype(float)
    b = np.clip(np.digitize(conf, EDGES) - 1, 0, NB - 1)
    ece = sum(abs(hit[b == k].mean() - conf[b == k].mean()) * (b == k).mean() for k in range(NB) if (b == k).any())
    ks = [k for k in range(NB) if (b == k).sum() >= 10]
    return ece, [conf[b == k].mean() for k in ks], [hit[b == k].mean() for k in ks]


fig, axs = plt.subplots(1, 3, figsize=(15, 5.6))
for ax, (title, g, expect) in zip(axs, PANELS):
    y = g["label"].map(L2I).to_numpy()
    raw, cal = probs(g)
    e_r, xr, yr = ece_and_curve(y, raw)
    e_c, xc, yc = ece_and_curve(y, cal)
    print(f"{title}: n={len(g)} ECE {100 * e_r:.1f} -> {100 * e_c:.1f} (Table S8: {expect[0]} -> {expect[1]})")
    assert (round(100 * e_r, 1), round(100 * e_c, 1)) == expect, title
    ax.plot([0, 1], [0, 1], color="#8c8c8c", lw=1.0, ls=(0, (3, 2)), label="Perfect calibration")
    ax.plot(xr, yr, color="black", lw=1.4, ls=(0, (5, 2.5)), marker="o", ms=6, mfc="white", mew=1.4,
            label="Uncalibrated")
    ax.plot(xc, yc, color="black", lw=1.6, marker="s", ms=6, mfc="black", label="Locked temperature")
    ax.set_xlim(0.3, 1.0); ax.set_ylim(0.3, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("Confidence (top-class probability)")
    ax.set_ylabel("Accuracy" if ax is axs[0] else "")
    ax.grid(True, color="#d0d0d0", lw=0.5, ls=(0, (2, 2)))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(0.03, 0.97, f"ECE {100 * e_r:.1f}% → {100 * e_c:.1f}%", transform=ax.transAxes, ha="left", va="top", fontsize=10.5,
            fontweight="bold")
    ax.text(0.97, 0.04, f"n = {len(g):,} slices", transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5)
    ax.text(0.5, -0.2, title, transform=ax.transAxes, ha="center", va="top", fontsize=12, fontweight="bold")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=10.5, handlelength=2.8, bbox_to_anchor=(0.5, 0.97))
fig.subplots_adjust(wspace=0.18, bottom=0.2, top=0.9)
fig.savefig(OUT, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.1)
print(OUT)


