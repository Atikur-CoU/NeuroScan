"""v11 Figure 2: input representations, laid out from the real tiles written by kernel
neuroscan-q-fig2-input-representations (each composite is the exact three-channel image the network receives).
Tiles are drawn unaltered (no smoothing, warping or retouching; only uniform display scaling). Layout follows the
user's redesign: channel labels, composite wording, a joined connector and a titled shared-processing box."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

K = Path(r"E:\CoU Projects\NeuroScan\NeuroScan_MultiDataset\kaggle_kernels\phase_q_fig2_input_representations\out\output")
T = K / "fig2_tiles"
OUT = Path(r"E:\CoU Projects\NeuroScan\NeuroScan_MultiDataset\paper\figures\v11_fig2_input_representations.png")
cases = json.loads((K / "fig2_cases.json").read_text())
INK, MUTED = "#1f2933", "#52606d"
CH = {"R": ("#c0392b", "Channel 1 (R)"), "G": ("#2e8b57", "Channel 2 (G)"), "B": ("#2f6fb3", "Channel 3 (B)")}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK})

SRC = {"AISD": "AISD", "SOOP_dwi_adc": "SOOP", "RSNA_ICH": "RSNA"}
ROWS = [
    ("ct_context", "(a) CT with known slice order (AISD, CPAISD, SICH)",
     [("Previous slice", "R"), ("Current slice", "G"), ("Next slice", "B")]),
    ("mri_pair", "(b) Diffusion MRI (ISLES 2022, SOOP)",
     [("DWI", "R"), ("ADC", "G"), ("DWI (repeated)", "B")]),
    ("ct_single", "(c) CT without slice order (RSNA, Kaggle CT)",
     [("Single slice", "R"), ("Same slice", "G"), ("Same slice", "B")]),
]

FW, FH = 13.2, 9.3
fig = plt.figure(figsize=(FW, FH))
TW = 0.122; TH = TW * FW / FH               # square tiles
XS = [0.020, 0.155, 0.290]                  # input tiles
XC = 0.478                                  # composite
YS = [0.690, 0.365, 0.040]                  # row bottoms
XBUS = 0.665                                # vertical connector
centres = []


def arrow(p0, p1, **kw):
    fig.patches.append(FancyArrowPatch(p0, p1, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=15,
                                       lw=1.5, color=INK, **kw))


def line(xs, ys):
    fig.lines.append(Line2D(xs, ys, transform=fig.transFigure, color=INK, lw=1.5))


for (name, title, tiles), y in zip(ROWS, YS):
    c = cases[name]
    fig.text(XS[0], y - 0.058, title, fontsize=13, fontweight="bold", ha="left", va="top")
    fig.text(XS[0], y - 0.088, f"Example: {SRC[c['dataset']]} held-out slice; reference: {c['label']}",
             fontsize=10, ha="left", va="top")
    for (lab, ch), x in zip(tiles, XS):
        col, chlab = CH[ch]
        ax = fig.add_axes((x, y, TW, TH))
        ax.imshow(Image.open(T / f"{name}_{ch}.png"), cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(col); s.set_linewidth(2.4)
        ax.set_title(lab, fontsize=10.5, pad=4)
        fig.text(x + TW / 2, y - 0.010, chlab, ha="center", va="top", fontsize=9.5, color="white", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.28", fc=col, ec="none"))
    yc = y + TH / 2
    centres.append(yc)
    fig.text((XS[2] + TW + XC) / 2, yc + 0.010, "Stack\nchannels", ha="center", va="bottom", fontsize=10)
    arrow((XS[2] + TW + 0.012, yc), (XC - 0.012, yc))
    ax = fig.add_axes((XC, y, TW, TH))
    ax.imshow(Image.open(T / f"{name}_composite.png"))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(INK); s.set_linewidth(1.2)
    ax.set_title("Three-channel composite", fontsize=10.5, pad=4)
    note = "All three channels identical" if c["channels_identical"] else "RGB display of stacked channels"
    fig.text(XC + TW / 2, y - 0.012, note, ha="center", va="top", fontsize=9.5)
    line([XC + TW + 0.008, XBUS], [yc, yc])

# joined connector into the shared box
line([XBUS, XBUS], [centres[2], centres[0]])
bx, by, bw = 0.705, YS[2] + 0.035, 0.280
bh = YS[0] + TH + 0.02 - by
fig.patches.append(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.004,rounding_size=0.008",
                                  transform=fig.transFigure, fc="white", ec=INK, lw=1.3))
fig.text(bx + bw / 2, by + bh - 0.035, "Shared preprocessing and classifier", ha="center", va="center", fontsize=11.5,
         fontweight="bold")
steps = ["Each channel 224 × 224\nInput tensor: 3 × 224 × 224", "ImageNet\nnormalization", "One shared\nConvNeXt-Tiny",
         "Normal / Ischemic /\nHemorrhagic"]
top, bot = centres[0] - 0.02, by + bh * 0.10
sy = [top - i * (top - bot) / 3 for i in range(4)]
for i, (s, yy) in enumerate(zip(steps, sy)):
    fig.text(bx + bw / 2, yy, s, ha="center", va="center", fontsize=11, fontweight="bold" if i else "normal",
             linespacing=1.4, bbox=dict(boxstyle="round,pad=0.6,rounding_size=0.3", fc="white", ec=INK, lw=1.3))
    if i < len(steps) - 1:
        arrow((bx + bw / 2, yy - 0.060), (bx + bw / 2, sy[i + 1] + 0.052))
arrow((XBUS, sy[0]), (bx + 0.045, sy[0]))
line([XBUS, XBUS], [sy[0], centres[0]])
fig.text(bx + bw / 2, by - 0.014, "MRI coverage: normal versus ischemic;\nhemorrhage evaluated on CT only.",
         ha="center", va="top", fontsize=9, color=MUTED)
fig.savefig(OUT, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.08)
print(OUT)

