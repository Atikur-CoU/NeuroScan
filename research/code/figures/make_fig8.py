"""v11 Figure 8: Grad-CAM of the final model against reference lesion masks, in colour, Figure 1 style. Six held-out
cases (the cases of the earlier figure), arrays from kernel neuroscan-q-fig8-gradcam-cases. Per case: input slice,
reference lesion outline, stage-4 Grad-CAM (ischemic class, identity view; raw 7 x 7 map, max-normalised, bilinear
upsampling) as a semi-transparent inferno overlay with the reference outline. Image pixels are not altered beyond
uniform resampling for display."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, font_manager
from matplotlib.colors import Normalize
from PIL import Image

for f in ("NotoSans-Regular.ttf", "NotoSans-Bold.ttf", "NotoSans-Italic.ttf"):
    font_manager.fontManager.addfont(rf"C:\Windows\Fonts\{f}")
plt.rcParams.update({"font.family": ["Noto Sans", "DejaVu Sans"], "font.size": 10.5, "text.color": "black"})
K = Path(r"E:\CoU Projects\NeuroScan\NeuroScan_MultiDataset\kaggle_kernels\phase_q_fig8_gradcam_cases\out\output")
OUT = Path(r"E:\CoU Projects\NeuroScan\NeuroScan_MultiDataset\paper\figures\v11_fig8_gradcam_vs_masks.png")
cases = {c["name"]: c for c in json.loads((K / "fig8_cases.json").read_text())}
ORDER = ["AISD_0", "AISD_1", "CPAISD_v2_2", "CPAISD_v2_0", "ISLES2022_dwi_adc_0", "ISLES2022_dwi_adc_1"]
SRC = {"AISD": "AISD, non-contrast CT", "CPAISD_v2": "CPAISD, non-contrast CT", "ISLES2022_dwi_adc": "ISLES 2022, DWI"}
S = 448
OUTLINE = (0, 229, 255)                      # cyan, contrasts with inferno and grey
CMAP = matplotlib.colormaps["inferno"]


def load(prefix):
    name = next(n for n in cases if n.startswith(prefix + "_"))
    z = np.load(K / "fig8_cases" / f"{name}.npz")
    img = np.asarray(Image.fromarray(z["img"].clip(0, 255).astype(np.uint8)).resize((S, S), Image.BICUBIC), float)
    mask = np.asarray(Image.fromarray(z["mask"].astype(np.uint8) * 255).resize((S, S), Image.NEAREST)) > 127
    camu = np.asarray(Image.fromarray(z["cam_up"].astype(np.float32), mode="F").resize((S, S), Image.BILINEAR))
    return cases[name], img, mask, np.clip(camu, 0, 1)


def edge(m, t=2):
    e = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
    for _ in range(t - 1):
        e = e | np.roll(e, 1, 0) | np.roll(e, 1, 1)
    return e


fig = plt.figure(figsize=(15.5, 9.6))
gs = fig.add_gridspec(3, 2, wspace=0.06, hspace=0.32, left=0.01, right=0.9, top=0.95, bottom=0.07)
letters = "abcdef"
for k, pre in enumerate(ORDER):
    c, img, mask, cam = load(pre)
    sub = gs[k // 2, k % 2].subgridspec(1, 3, wspace=0.03)
    grey = np.stack([img] * 3, -1) / 255
    ref = grey.copy()
    ref[edge(mask)] = np.array(OUTLINE) / 255
    heat = CMAP(cam)[..., :3]
    a = (0.15 + 0.6 * cam)[..., None]
    over = grey * (1 - a) + heat * a
    over[edge(mask)] = np.array(OUTLINE) / 255
    for j, (im, title) in enumerate(((grey, "Input"), (ref, "Reference lesion"), (over, "Grad-CAM"))):
        ax = fig.add_subplot(sub[0, j])
        ax.imshow(im, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_linewidth(0.8)
        if k < 2:
            ax.set_title(title, fontsize=11, pad=5)
        if j == 1:
            pi = c["probs"][1]
            status = "detected" if c["pred_locked"] == "ischemic" else f"missed (predicted {c['pred_locked']})"
            ax.text(0.5, -0.07, f"({letters[k]}) {SRC[c['dataset']]}: {status}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=11.5, fontweight="bold")
            ax.text(0.5, -0.17, "P(ischemic) > 0.99" if pi > 0.995 else f"P(ischemic) {pi:.2f}", transform=ax.transAxes, ha="center", va="top", fontsize=10.5)
# colour bar and outline legend
cax = fig.add_axes((0.915, 0.30, 0.013, 0.45))
cb = fig.colorbar(cm.ScalarMappable(norm=Normalize(0, 1), cmap=CMAP), cax=cax)
cb.set_label("Grad-CAM, ischemic class (max-normalized)", fontsize=10.5)
cb.set_ticks([0, 0.5, 1])
fig.legend(handles=[plt.Line2D([], [], color=np.array(OUTLINE) / 255, lw=2.5)], labels=["Reference\nlesion outline"],
           loc="center left", bbox_to_anchor=(0.895, 0.2), frameon=False, fontsize=10.5, handlelength=1.6)
fig.savefig(OUT, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.1)
print(OUT)



