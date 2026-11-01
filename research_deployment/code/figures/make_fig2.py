"""Figure 2 of paper 2: the served model's head with the closed-form Grad-CAM branch, as one ONNX graph.
Black and white, Noto Sans (same conventions as the companion paper's diagrams). Usage: python -I make_fig2.py"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

for f in ("NotoSans-Regular", "NotoSans-Bold", "NotoSans-Italic", "NotoSans-BoldItalic"):
    font_manager.fontManager.addfont(f"C:/Windows/Fonts/{f}.ttf")
plt.rcParams.update({"font.family": "Noto Sans", "font.size": 8, "mathtext.fontset": "custom",
                     "mathtext.rm": "Noto Sans", "mathtext.it": "Noto Sans:italic", "mathtext.bf": "Noto Sans:bold"})
OUT = Path(__file__).resolve().parent
INK, GREY = "black", "0.88"

fig = plt.figure(figsize=(7.1, 3.0))   # 18 cm wide
ax = fig.add_axes([0, 0, 1, 1])        # the drawing fills the figure, so box sizes match the text
ax.set_xlim(0, 120)
ax.set_ylim(0, 50)
ax.axis("off")


def box(x, y, w, h, text, size=7.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc="white", ec=INK, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, linespacing=1.4)


def arrow(x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.9, color=INK, shrinkA=0, shrinkB=0,
                                                         mutation_scale=8))


def line(xs, ys):
    ax.plot(xs, ys, color=INK, lw=0.9, solid_capstyle="butt")


# regions: one ONNX graph; backbone quantized in the INT8 copies; head and branch kept in FP32
ax.add_patch(FancyBboxPatch((16.5, 1.0), 87.5, 43.0, boxstyle="round,pad=0,rounding_size=1.5", fc="none", ec=INK,
                            lw=0.8, ls=(0, (4, 3))))
ax.text(17.5, 46.8, "One ONNX graph: a single forward pass returns both outputs", fontsize=7.5, style="italic",
        va="center")
ax.add_patch(FancyBboxPatch((18.5, 21.0), 17.0, 20.0, boxstyle="round,pad=0,rounding_size=1.0", fc=GREY, ec="none"))
ax.text(27.0, 24.2, "quantized in the\nINT8 copies", fontsize=6.8, ha="center", va="center", style="italic",
        linespacing=1.2)
ax.text(76.0, 41.6, "head and Grad-CAM branch: FP32 in every copy", fontsize=6.8, ha="center", style="italic")

Y, H = 28.5, 10.5
box(0.3, Y, 14.0, H, "Input\n4 views\n3 × 224 × 224")
box(20.0, Y, 14.0, H, "ConvNeXt-Tiny\nstages 1–4")
box(38.0, Y, 10.5, H, "$A$\n768 × 7 × 7")
box(52.0, Y, 11.5, H, "Average\npool\n$p_k$")
box(67.0, Y, 17.5, H, "LayerNorm\n$\\hat{x}_k = (p_k-\\mu)/\\sigma$\n$y_k = \\gamma_k \\hat{x}_k + \\beta_k$", size=7.3)
box(88.0, Y, 14.0, H, "Linear\n$z_c = \\Sigma_k W_{ck} y_k + b_c$", size=7.3)
for x0, x1 in ((14.3, 20.0), (34.0, 38.0), (48.5, 52.0), (63.5, 67.0), (84.5, 88.0)):
    arrow(x0, Y + H / 2, x1, Y + H / 2)
arrow(102.0, Y + H / 2, 106.5, Y + H / 2)
ax.text(107.3, Y + H / 2, "logits\n3", va="center", fontsize=8, weight="bold", linespacing=1.3)

# branch
YB, HB = 9.5, 11.0
box(66.0, YB, 26.0, HB, "Closed-form weights\n"
    "$\\alpha_{ck} = (v_{ck} - \\bar{v}_c - \\hat{x}_k\\,\\overline{v_c\\hat{x}})\\,/\\,HW\\sigma$\n"
    "$v_{ck} = W_{ck}\\gamma_k$", size=7.3)
box(95.0, YB, 7.0, HB, "$\\Sigma_k \\alpha_{ck} A_k$\nReLU", size=7.3)
arrow(75.75, Y, 75.75, YB + HB)          # sigma, x-hat, gamma from the LayerNorm
ax.text(76.6, 23.6, "$\\sigma,\\ \\hat{x},\\ \\gamma$", fontsize=7.5)
arrow(95.0, Y, 88.0, YB + HB)            # W from the linear layer
ax.text(93.2, 23.6, "$W$", fontsize=7.5)
arrow(92.0, YB + HB / 2, 95.0, YB + HB / 2)
line([43.25, 43.25], [Y, 4.5])           # A, around the bottom, to the weighted sum
line([43.25, 98.5], [4.5, 4.5])
arrow(98.5, 4.5, 98.5, YB)
ax.text(44.2, 5.8, "the same feature maps $A$ (identity view)", fontsize=7, style="italic")
arrow(102.0, YB + HB / 2, 106.5, YB + HB / 2)
ax.text(107.3, YB + HB / 2, "Grad-CAM\n3 × 7 × 7", va="center", fontsize=8, weight="bold", linespacing=1.3)

for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig2_closed_form_head.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight",
                pad_inches=0.03)
print("written", OUT / "fig2_closed_form_head.png")
