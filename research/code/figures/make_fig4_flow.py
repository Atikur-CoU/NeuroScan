"""v11 Figure 4: development of NeuroScan as a vertical flow diagram (CONSORT/PRISMA-like), black and white, Figure 1
style. Main column: the four development stages with the changes adopted in each and the stage outcome in a footer
band; side boxes (sized to their content, centred on the stage): changes not adopted; dashed line: test freeze;
bottom box: the one-time final scoring. No calendar dates. Content as in Sections 4.1-4.2."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

for f in ("NotoSans-Regular.ttf", "NotoSans-Bold.ttf", "NotoSans-Italic.ttf"):
    font_manager.fontManager.addfont(rf"C:\Windows\Fonts\{f}")
plt.rcParams.update({"font.family": ["Noto Sans", "DejaVu Sans"], "text.color": "black"})
OUT = Path(r"E:\CoU Projects\NeuroScan\NeuroScan_MultiDataset\paper\figures\v11_fig4_development_flow.png")
K, LW = "black", 1.4

STAGES = [
    ("Slice-level partition", "ResNet18",
     ["Single softmax replaces the two-stage cascade", "Adjacent CT slices as channels",
      "CPAISD re-rendered in HU (ischemic F1 +1.1)", "DWI paired with ADC (ischemic F1 +3.1)"],
     "Test accuracy 95.2% on a slice-level split",
     []),
    ("Patient-level partition", "ResNet18",
     ["Patient-level partition (accuracy 95.2% → 91.5%)", "Second leakage audit (accuracy → 91.3%)",
      "Within-source normals (CPAISD 43.5% → 81.4%)", "SOOP diffusion cohort added", "MRI collection and near-blank slices removed"],
     "Test set frozen",
     ["Segmentation head", "Attention gating", "CLAHE on CT", "Six-channel MRI input"]),
    ("Validation only", "ResNet18 → ConvNeXt-Tiny",
     ["ConvNeXt-Tiny in both seeds (macro-F1 +2.10, +1.41)", "Modality-specific calibration"],
     "Final model: ConvNeXt-Tiny, seed 123",
     ["EMA weights", "AdamW recipe", "Level-matched normals", "EfficientNet-B0", "2- and 3-model ensembles"]),
    ("Second development phase", "ConvNeXt-Tiny, new validation split",
     [],
     "Final model kept unchanged",
     ["Symmetry-aware fusion (macro-F1 −0.58)", "Adjacent-slice late fusion (macro-F1 −0.97)",
      "Multi-window CT (withdrawn after audit)"]),
]

MX, MW, SX, SW = 0, 64, 74, 52           # main and side columns
HEAD, LINE, PAD, FOOT = 6.2, 3.05, 2.0, 5.6
VGAP, FREEZE_EXTRA, FINAL_H = 6.0, 6.5, 9.5
FS = 11.8


def main_h(adopted, outcome):
    n = 1 + max(len(adopted), 1)          # model line + adopted lines (or "none")
    return HEAD + PAD + n * LINE + PAD + (FOOT if outcome else 0)


def side_h(rejected):
    return HEAD + PAD + max(len(rejected), 1) * LINE + PAD


blocks = [max(main_h(a, o), side_h(r)) for _, _, a, o, r in STAGES]
total = sum(blocks) + 3 * VGAP + FREEZE_EXTRA + VGAP + FINAL_H
fig, ax = plt.subplots(figsize=(12.4, total * 0.1))
ax.set_xlim(-1, SX + SW + 1); ax.set_ylim(-0.5, total + 0.5); ax.axis("off"); ax.set_aspect("equal")


def rbox(x, y, w, h, lw=LW):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc="white", ec=K, lw=lw,
                                zorder=2))


def arrow(p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=15, lw=LW, color=K, zorder=3))


def lines(x, ytop, items, none_text):
    y = ytop - PAD - LINE / 2
    if not items:
        ax.text(x, y, none_text, ha="left", va="center", fontsize=FS, style="italic", zorder=4)
        return
    for s in items:
        cont = s.startswith("  ")
        if not cont:
            ax.text(x, y, "•", ha="left", va="center", fontsize=FS, zorder=4)
        ax.text(x + 2.2, y, s.strip(), ha="left", va="center", fontsize=FS, zorder=4)
        y -= LINE


y = total
cx_main = MX + MW / 2
for k, (title, model, adopted, outcome, rejected) in enumerate(STAGES):
    bh = blocks[k]
    centre = y - bh / 2
    # main box, top-aligned with the block
    mh = bh
    my0 = y - mh
    rbox(MX, my0, MW, mh)
    ax.plot([MX, MX + MW], [y - HEAD, y - HEAD], color=K, lw=LW, zorder=3)
    ax.text(MX + 2.4, y - HEAD / 2, f"{k + 1}", ha="left", va="center", fontsize=14, fontweight="bold", zorder=4)
    ax.text(MX + 6.4, y - HEAD / 2, title, ha="left", va="center", fontsize=14, fontweight="bold", zorder=4)
    ax.text(MX + 3, y - HEAD - PAD - LINE / 2, model, ha="left", va="center", fontsize=FS, style="italic", zorder=4)
    lines(MX + 3, y - HEAD - LINE, adopted, "No change adopted")
    if outcome:
        ax.plot([MX, MX + MW], [my0 + FOOT, my0 + FOOT], color=K, lw=0.8, zorder=3)
        ax.text(MX + 3, my0 + FOOT / 2, outcome, ha="left", va="center", fontsize=FS + 0.3, fontweight="bold",
                zorder=4)
    # side box, sized to its content and centred on the stage
    sh = side_h(rejected)
    sy0 = centre - sh / 2
    rbox(SX, sy0, SW, sh)
    ax.plot([SX, SX + SW], [sy0 + sh - HEAD, sy0 + sh - HEAD], color=K, lw=LW, zorder=3)
    ax.text(SX + 2.4, sy0 + sh - HEAD / 2, "Not adopted", ha="left", va="center", fontsize=13, fontweight="bold",
            zorder=4)
    lines(SX + 3, sy0 + sh - HEAD, rejected, "None")
    arrow((MX + MW + 0.2, centre), (SX - 0.2, centre))
    y -= bh
    if k < 3:
        gap = VGAP + (FREEZE_EXTRA if k == 1 else 0)
        if k == 1:
            yf = y - gap / 2
            ax.plot([MX - 0.5, SX + SW + 0.5], [yf, yf], color=K, lw=1.4, linestyle=(0, (6, 4)), zorder=1)
            ax.text(SX + SW / 2, yf, "Test freeze: validation data only from here", ha="center", va="center",
                    fontsize=11.5, style="italic", zorder=4,
                    bbox=dict(boxstyle="square,pad=0.35", fc="white", ec="none"))
        arrow((cx_main, y - 0.2), (cx_main, y - gap + 0.2))
        y -= gap
arrow((cx_main, y - 0.2), (cx_main, y - VGAP + 0.2))
y -= VGAP
rbox(MX, y - FINAL_H, MW, FINAL_H, lw=2.2)
ax.text(cx_main, y - FINAL_H / 2, "Final model scored once on the held-out\ninternal test, CPAISD and SOOP cohorts",
        ha="center", va="center", fontsize=12.8, fontweight="bold", linespacing=1.3, zorder=4)
fig.savefig(OUT, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.1)
print(OUT)


