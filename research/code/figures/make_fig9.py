"""v11 Figure 9: prototype output for six held-out test cases (random by reference label, seed 2026), in colour, Figure 1
style. Inputs and overlays are the served prototype's own outputs (kernel neuroscan-q-fig3-prototype-output v2,
fig9_cases/); probabilities and tiers from results.json. Images are only resampled for display."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image

for f in ("NotoSans-Regular.ttf", "NotoSans-Bold.ttf", "NotoSans-Italic.ttf"):
    font_manager.fontManager.addfont(rf"C:\Windows\Fonts\{f}")
plt.rcParams.update({"font.family": ["Noto Sans", "DejaVu Sans"], "font.size": 10.5, "text.color": "black"})
K = Path(r"E:\CoU Projects\NeuroScan\NeuroScan_MultiDataset\kaggle_kernels\phase_q_fig3_prototype_output\out\output\fig9_cases")
OUT = Path(r"E:\CoU Projects\NeuroScan\NeuroScan_MultiDataset\paper\figures\v11_fig9_prototype_output.png")
res = json.loads((K / "results.json").read_text())
TIER = {"High alert": "#C0392B", "Moderate alert": "#E67E22", "Normal": "#2980B9", "Uncertain": "#7F8C8D",
        "Unreliable": "#7F8C8D"}
SRC = {"SICH": "SICH, CT", "AISD": "AISD, CT", "ISLES 2022": "ISLES 2022, DWI", "SOOP": "SOOP, DWI"}
S = 448


def img(p):
    return np.asarray(Image.open(p).convert("RGB").resize((S, S), Image.BICUBIC))


fig = plt.figure(figsize=(15.5, 8.6))
gs = fig.add_gridspec(2, 3, wspace=0.08, hspace=0.42, left=0.01, right=0.99, top=0.95, bottom=0.14)
for k, c in enumerate(res):
    d = K / Path(c["dir"]).name
    sub = gs[k // 3, k % 3].subgridspec(1, 2, wspace=0.03)
    for j, (im, title) in enumerate(((img(d / "main.png"), "Input"), (img(d / "overlay.png"), "Prototype output"))):
        ax = fig.add_subplot(sub[0, j])
        ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([])
        if k < 3:
            ax.set_title(title, fontsize=11, pad=5)
        if j == 0:
            p = c["probs"]
            ax.text(1.02, -0.06, f"({'abcdef'[k]}) {SRC[c['source']]}: reference {c['label']}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=11.5, fontweight="bold")
            ax.text(1.02, -0.155, f"Predicted {c['prediction']}  ·  N {p[0]:.2f}  ·  I {p[1]:.2f}  ·  H {p[2]:.2f}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=10.5)
            ax.text(1.02, -0.255, c["tier"], transform=ax.transAxes, ha="center", va="top", fontsize=10.5, color="white",
                    fontweight="bold", bbox=dict(boxstyle="round,pad=0.32", fc=TIER[c["tier"]], ec="none"))
# legend: tiers (wording as in Table 2); the outline marks the region of model evidence for lesion predictions
from matplotlib.patches import Patch
handles = [Patch(fc=TIER[n], ec="none", label=l) for n, l in (("High alert", "High alert"), ("Moderate alert", "Moderate alert"),
                                                              ("Normal", "No alert: normal"), ("Uncertain", "No alert: uncertain"))]
handles.append(plt.Line2D([], [], color="#555555", lw=2, label="Outline: region of model evidence (lesion predictions)"))
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5, frameon=False, fontsize=10.5,
           handlelength=1.6, handleheight=1.1, columnspacing=1.6)
fig.savefig(OUT, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.1)
print(OUT)


