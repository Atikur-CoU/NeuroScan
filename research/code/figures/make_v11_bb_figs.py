"""v11 backbone figures in one black-and-white style (Noto Sans, hatch patterns, final model first and marked by a grey
band and bold label), from backbone_comparison.json (validation partition, seed 123, plain argmax):
  Figure 5  -- (a) parameters, (b) accuracy and macro-F1, (c) macro-F1 and (d) ischemic AUROC difference from
               ConvNeXt-Tiny with 95% paired patient-cluster bootstrap CIs;
  Figure 6  -- per-class precision, recall and F1, one shared y-range;
  Figure S2 -- confusion matrices, greyscale, counts and row percentages."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Patch

for f in ("NotoSans-Regular.ttf", "NotoSans-Bold.ttf", "NotoSans-Italic.ttf"):
    font_manager.fontManager.addfont(rf"C:\Windows\Fonts\{f}")
plt.rcParams.update({"font.family": ["Noto Sans", "DejaVu Sans"], "font.size": 10, "text.color": "black",
                     "axes.edgecolor": "black", "axes.labelcolor": "black", "xtick.color": "black",
                     "ytick.color": "black", "hatch.linewidth": 0.7, "axes.linewidth": 0.9})
FIG = Path(r"E:\CoU Projects\NeuroScan\NeuroScan_MultiDataset\paper\figures")
d = json.loads(Path(r"E:\CoU Projects\NeuroScan\NeuroScan_MultiDataset\kaggle_kernels\phase_q_backbone_comparison"
                    r"\out\output\backbone_comparison.json").read_text())
M = {m["name"]: m for m in d["models"]}
V = d["validation"]["per_model"]
P = d["validation"]["paired_vs_final"]
ORDER = ["convnext_s123", "swint_s123", "vgg16_s123", "densenet121_s123", "resnet101_s123", "effb0_s123",
         "resnet18_s123", "mobilenetv2_s123"]          # validation macro-F1, highest first
assert [round(100 * V[n]["macro_f1_present_classes"], 2) for n in ORDER] == sorted(
    [round(100 * V[n]["macro_f1_present_classes"], 2) for n in ORDER], reverse=True)
NAMES = ["ConvNeXt-Tiny (final)" if n == "convnext_s123" else M[n]["display"] for n in ORDER]
GREY_BAND = "#e6e6e6"


def style(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis=grid_axis, color="#bdbdbd", lw=0.5, ls=(0, (2, 2)))
    ax.set_axisbelow(True)


def xlabels(ax, x, rot=35):
    ax.set_xticks(x)
    ax.set_xticklabels(NAMES, rotation=rot, ha="right", rotation_mode="anchor", fontsize=9.5)
    ax.get_xticklabels()[0].set_fontweight("bold")


def panel_title(ax, s, y=-0.2):
    """Panel title centred under the panel (below the axis labels)."""
    ax.text(0.5, y, s, transform=ax.transAxes, ha="center", va="top", fontsize=12, fontweight="bold")


# ================================================================== Figure 5
fig = plt.figure(figsize=(13, 9.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.95], hspace=0.78, wspace=0.62)
x = np.arange(len(ORDER))

ax = fig.add_subplot(gs[0, 0])
par = [M[n]["params_M"] for n in ORDER]
ax.axvspan(-0.5, 0.5, color=GREY_BAND, zorder=0)
ax.bar(x, par, width=0.6, color="white", edgecolor="black", hatch="///", lw=1.0, zorder=2)
ax.bar(x[:1], par[:1], width=0.6, color="#404040", edgecolor="black", lw=1.0, zorder=3)
ax.set_yscale("log")
ax.set_ylim(1, 400)
ax.set_yticks([1, 2, 5, 10, 20, 50, 100, 200])
ax.set_yticklabels(["1", "2", "5", "10", "20", "50", "100", "200"])
for xi, v in zip(x, par):
    ax.text(xi, v * 1.12, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Parameters (millions, log scale)")
xlabels(ax, x)
style(ax)
panel_title(ax, "(a) Model size", -0.46)

ax = fig.add_subplot(gs[0, 1])
acc = [100 * V[n]["accuracy"] for n in ORDER]
mf1 = [100 * V[n]["macro_f1_present_classes"] for n in ORDER]
w = 0.36
ax.axvspan(-0.5, 0.5, color=GREY_BAND, zorder=0)
ax.bar(x - w / 2, acc, w, color="white", edgecolor="black", hatch="...", lw=1.0, zorder=2, label="Accuracy")
ax.bar(x + w / 2, mf1, w, color="#9e9e9e", edgecolor="black", hatch="xx", lw=1.0, zorder=2, label="Macro-F1")
for xi, a, f in zip(x, acc, mf1):
    ax.text(xi - w / 2, a + 0.08, f"{a:.1f}", ha="center", va="bottom", fontsize=8.3, rotation=90)
    ax.text(xi + w / 2, f + 0.08, f"{f:.1f}", ha="center", va="bottom", fontsize=8.3, rotation=90)
ax.set_ylim(84, 91)
ax.set_ylabel("Validation score (%)")
ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=9.5, handlelength=1.6, handleheight=1.1)
xlabels(ax, x)
style(ax)
panel_title(ax, "(b) Accuracy and macro-F1 (axis from 84%)", -0.46)

OTHERS = ORDER[1:]
yy = np.arange(len(OTHERS))[::-1]
for col, (key, scale, xl, title) in enumerate((("macro_f1_present_classes", 100, "Difference in macro-F1 (percentage points)",
                                                 "(c) Macro-F1 vs ConvNeXt-Tiny"),
                                                ("ischemic_auroc", 1, "Difference in ischemic AUROC",
                                                 "(d) Ischemic AUROC vs ConvNeXt-Tiny"))):
    ax = fig.add_subplot(gs[1, col])
    for yi, n in zip(yy, OTHERS):
        r = P[n][key]
        est, lo, hi = r["diff_other_minus_ref"] * scale, r["ci95"][0] * scale, r["ci95"][1] * scale
        sig = hi < 0 or lo > 0
        ax.plot([lo, hi], [yi, yi], color="black", lw=1.4, zorder=2)
        for e in (lo, hi):
            ax.plot([e, e], [yi - 0.12, yi + 0.12], color="black", lw=1.4, zorder=2)
        ax.plot(est, yi, marker="s", ms=8, mfc="black" if sig else "white", mec="black", mew=1.4, zorder=3)
        txt = f"{est:+.2f} ({lo:+.2f}, {hi:+.2f})" if scale == 100 else f"{est:+.3f} ({lo:+.3f}, {hi:+.3f})"
        ax.text(1.02, yi, txt.replace("-", "−"), transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8.8)
    ax.axvline(0, color="black", lw=1.0, ls=(0, (4, 3)), zorder=1)
    if scale == 1:
        ax.set_xticks([-0.03, -0.02, -0.01, 0.0])
        ax.set_xticklabels(["−0.03", "−0.02", "−0.01", "0"])
    ax.set_yticks(yy)
    ax.set_yticklabels([M[n]["display"] for n in OTHERS] if col == 0 else [], fontsize=9.5)
    ax.set_ylim(-0.7, len(OTHERS) - 0.3)
    ax.set_xlabel(xl)
    style(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    panel_title(ax, title, -0.27)
fig.legend(handles=[plt.Line2D([], [], marker="s", ls="", ms=8, mfc="black", mec="black", label="95% CI excludes zero"),
                    plt.Line2D([], [], marker="s", ls="", ms=8, mfc="white", mec="black", mew=1.4,
                               label="95% CI includes zero")],
           loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=9.5)
fig.subplots_adjust(right=0.84, bottom=0.15)
fig.savefig(FIG / "v11_fig5_backbones.png", dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.1)
plt.close(fig)

# ================================================================== Figure 6
MET = [("precision", "Precision", "white", "..."), ("recall", "Recall", "#bdbdbd", "xx"), ("f1", "F1", "#5a5a5a", "///")]
fig, axs = plt.subplots(3, 1, figsize=(13, 12.5), sharey=True)
w = 0.26
for ax, (cls, title) in zip(axs, (("normal", "(a) Normal"), ("ischemic", "(b) Ischemic"),
                                   ("hemorrhagic", "(c) Hemorrhagic"))):
    ax.axvspan(-0.5, 0.5, color=GREY_BAND, zorder=0)
    for j, (k, lab, fc, h) in enumerate(MET):
        vals = [100 * V[n][f"{cls}_{k}"] for n in ORDER]
        ax.bar(x + (j - 1) * w, vals, w, color=fc, edgecolor="black", hatch=h, lw=0.9, zorder=2, label=lab)
        for xi, v in zip(x, vals):
            ax.text(xi + (j - 1) * w, v + 0.4, f"{v:.1f}", ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_ylim(70, 100)
    ax.set_ylabel("Validation score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("\n", " ") for s in NAMES], fontsize=10)
    ax.get_xticklabels()[0].set_fontweight("bold")
    style(ax)
    panel_title(ax, title, -0.16)
axs[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=False, fontsize=10.5, handlelength=2,
              handleheight=1.2)
fig.subplots_adjust(hspace=0.5)
fig.savefig(FIG / "v11_fig_perclass_backbones.png", dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.1)
plt.close(fig)

# ================================================================== Figure S2
fig, axs = plt.subplots(2, 4, figsize=(14, 7.4))
labs = ["Normal", "Ischemic", "Hemorrhagic"]
for ax, n, nm in zip(axs.flat, ORDER, NAMES):
    cm = np.array(V[n]["confusion_matrix_rows_true_cols_pred"], float)
    rowp = cm / cm.sum(1, keepdims=True)
    ax.imshow(rowp, cmap="Greys", vmin=0, vmax=1.15)
    for i in range(3):
        for j in range(3):
            dark = rowp[i, j] > 0.55
            ax.text(j, i, f"{int(cm[i, j]):,}\n{100 * rowp[i, j]:.1f}%", ha="center", va="center", fontsize=8.6,
                    color="white" if dark else "black", fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(labs, fontsize=8.5); ax.set_yticklabels(labs, fontsize=8.5)
    ax.set_xlabel("Predicted", fontsize=9); ax.set_ylabel("Reference", fontsize=9)
    for s in ax.spines.values():
        s.set_linewidth(2.0 if n == ORDER[0] else 0.9)
    ax.text(0.5, -0.3, nm.replace("\n", " "), transform=ax.transAxes, ha="center", va="top", fontsize=11,
            fontweight="bold" if n == ORDER[0] else "normal")
fig.tight_layout(h_pad=3.5)
fig.savefig(FIG / "v11_figS2_confusion_backbones.png", dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.1)
print("ok")




