"""v11 Figure 3: prototype architecture, black and white, in the visual style of Figure 1 (rounded boxes, bold header
separated by a rule, line-art icon on the left, plain text lines). Content checked against render_inference_server/app.py
(X-API-Key on /api/health and /api/predict), webapp_php/includes/api_auth.php (Android bearer tokens),
webapp_php/sql/schema.sql (10 tables) and Section 3.7."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

for f in ("NotoSans-Regular.ttf", "NotoSans-Bold.ttf"):
    font_manager.fontManager.addfont(rf"C:\Windows\Fonts\{f}")
plt.rcParams.update({"font.family": "Noto Sans", "text.color": "black"})
OUT = Path(r"E:\CoU Projects\NeuroScan\NeuroScan_MultiDataset\paper\figures\v11_fig3_prototype_architecture.png")
K, LW = "black", 1.6

fig, ax = plt.subplots(figsize=(21.4, 6.4))
ax.set_xlim(0, 214); ax.set_ylim(0, 64); ax.axis("off"); ax.set_aspect("equal")
PITCH = 3.9


def box(x, y, w, h, title, lines, icon=None, fs=13.5, head=7.0, text_x=19):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec=K, lw=LW))
    ax.plot([x, x + w], [y + h - head, y + h - head], color=K, lw=LW)
    ax.text(x + 2.5, y + h - head / 2, title, ha="left", va="center", fontsize=16, fontweight="bold")
    cy = y + (h - head) / 2
    if icon:
        icon(x + 9, cy)
    top = cy + (len(lines) - 1) * PITCH / 2
    for i, s in enumerate(lines):
        ax.text(x + text_x, top - i * PITCH, s, ha="left", va="center", fontsize=fs)


def arrow(p0, p1, both=False, dashed=False):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="<|-|>" if both else "-|>", mutation_scale=16, lw=LW, color=K,
                                 linestyle="--" if dashed else "-"))


# ---------------------------------------------------------------- icons (line art)
def ic_clients(cx, cy):
    ax.add_patch(FancyBboxPatch((cx - 6.2, cy - 1.2), 8.4, 6.0, boxstyle="round,pad=0,rounding_size=0.5", fc="white", ec=K, lw=2.2))
    ax.plot([cx - 2.0, cx - 2.0], [cy - 1.2, cy - 3.0], color=K, lw=2.2)
    ax.plot([cx - 4.3, cx + 0.3], [cy - 3.0, cy - 3.0], color=K, lw=2.2, solid_capstyle="round")
    ax.add_patch(FancyBboxPatch((cx + 3.2, cy - 3.0), 3.4, 6.8, boxstyle="round,pad=0,rounding_size=0.6", fc="white", ec=K, lw=2.2))
    ax.plot([cx + 4.3, cx + 5.5], [cy + 3.0, cy + 3.0], color=K, lw=1.6)


def ic_server(cx, cy):
    for i in range(3):
        y = cy - 4.2 + i * 3.0
        ax.add_patch(FancyBboxPatch((cx - 5, y), 10, 2.4, boxstyle="round,pad=0,rounding_size=0.5", fc="white", ec=K, lw=2.2))
        ax.add_patch(Circle((cx - 3.3, y + 1.2), 0.42, color=K))
        ax.plot([cx + 0.5, cx + 3.6], [y + 1.2, y + 1.2], color=K, lw=1.6)


def ic_db(cx, cy):
    w, h = 9.0, 2.6
    ax.add_patch(Ellipse((cx, cy + 3.8), w, h, fc="white", ec=K, lw=2.2))
    for y in (cy + 1.2, cy - 1.4, cy - 4.0):
        ax.add_patch(Arc((cx, y), w, h, theta1=180, theta2=360, ec=K, lw=2.2))
    ax.plot([cx - w / 2, cx - w / 2], [cy - 4.0, cy + 3.8], color=K, lw=2.2)
    ax.plot([cx + w / 2, cx + w / 2], [cy - 4.0, cy + 3.8], color=K, lw=2.2)


def ic_model(cx, cy):
    pts = [(cx - 4, cy + 3), (cx + 4, cy + 3), (cx - 4, cy - 3), (cx + 4, cy - 3), (cx, cy + 4.8), (cx, cy - 4.8), (cx, cy)]
    for a in pts[:6]:
        ax.plot([a[0], cx], [a[1], cy], color=K, lw=1.8, zorder=1)
    for a, b in ((0, 4), (1, 4), (2, 5), (3, 5), (0, 2), (1, 3)):
        ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], color=K, lw=1.8, zorder=1)
    for p in pts:
        ax.add_patch(Circle(p, 1.15, fc="white", ec=K, lw=2.0, zorder=2))


# ---------------------------------------------------------------- tiers
box(2, 36, 58, 26, "Clients", ["Web browser • Android app", "Same access control", "Alert tier set by the platform"],
    ic_clients, text_x=18)
box(78, 36, 58, 26, "Clinical platform", ["PHP • four user roles", "Upload and assignment checks",
                                          "Alert tiers, reviews, audit log"], ic_server, text_x=18)
box(154, 36, 58, 26, "Inference service", ["FastAPI + ONNX Runtime", "Final ConvNeXt-Tiny model",
                                           "Stores no patient data"], ic_model, text_x=18)
box(78, 2, 58, 24, "Database", ["MySQL • ten tables", "Scans, predictions, reviews"],
    ic_db, text_x=18)
# API panel under the inference service
ax.add_patch(FancyBboxPatch((154, 2), 58, 24, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec=K, lw=LW))
ax.plot([154, 212], [19, 19], color=K, lw=LW)
ax.text(156.5, 22.5, "API (X-API-Key)", ha="left", va="center", fontsize=16, fontweight="bold")
for yy, s, mono in ((14.4, "GET  /api/health", True), (10.5, "POST /api/predict", True),
                    (6.6, "Probabilities, warnings, overlay", False)):
    ax.text(157, yy, s, ha="left", va="center", fontsize=13.5, fontweight="bold" if mono else "normal",
            family="DejaVu Sans Mono" if mono else "Noto Sans")

# ---------------------------------------------------------------- connections
arrow((60.4, 45.5), (77.6, 45.5), both=True)
ax.text(69, 48.6, "Session / token", ha="center", va="center", fontsize=12)
arrow((136.4, 49), (153.6, 49))
ax.text(145, 52, "Request", ha="center", va="center", fontsize=12)
arrow((153.6, 42), (136.4, 42))
ax.text(145, 39, "Response", ha="center", va="center", fontsize=12)
arrow((107, 35.6), (107, 26.4), both=True)
ax.text(109, 31, "Read / write", ha="left", va="center", fontsize=12)
ax.plot([183, 183], [35.6, 26.4], color=K, lw=LW, linestyle=(0, (4, 3)))
fig.savefig(OUT, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.1)
print(OUT)


