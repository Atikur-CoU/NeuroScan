"""Figure 1 of paper 2: the deployed platform, from the repository's own configuration
(render_inference_server/start.sh, Dockerfile, app.py; webapp_php/includes/inference_client.php; android ApiService).
Black and white, Noto Sans (same conventions as Figure 2). Usage: python -I make_fig1.py"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

for f in ("NotoSans-Regular", "NotoSans-Bold", "NotoSans-Italic"):
    font_manager.fontManager.addfont(f"C:/Windows/Fonts/{f}.ttf")
plt.rcParams.update({"font.family": "Noto Sans", "font.size": 7.3})
OUT = Path(__file__).resolve().parent
INK, GREY = "black", "0.90"
DASH = (0, (4, 3))

fig = plt.figure(figsize=(7.1, 4.6))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 120)
ax.set_ylim(0, 80)
ax.axis("off")


def frame(x, y, w, h, title, sub):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5", fc=GREY, ec=INK, lw=0.9))
    ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="center", weight="bold", fontsize=7.6)
    ax.text(x + w / 2, y + h - 5.4, sub, ha="center", va="center", style="italic", fontsize=6.8)


def box(x, y, w, h, text, size=6.8, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc="white", ec=INK, lw=0.8,
                                ls=DASH if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, linespacing=1.35)


def arrow(x0, y0, x1, y1, both=False, dashed=False):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="<|-|>" if both else "-|>", lw=0.9, color=INK,
                                                         shrinkA=0, shrinkB=0, mutation_scale=8,
                                                         ls=DASH if dashed else "-"))


def note(x, y, text, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=6.6, style="italic", linespacing=1.3)


# clients
box(0.3, 51.0, 19.0, 11.0, "Web browser\npatients, doctors,\nradiologists, admins", size=7.0)
box(0.3, 36.0, 19.0, 11.0, "Android app\nsame accounts\nand roles", size=7.0)

# web application on shared hosting
frame(28.0, 10.0, 38.0, 58.0, "Web application", "shared cPanel hosting: PHP, MySQL")
box(30.0, 44.0, 34.0, 13.0, "Web pages and JSON API\nlogin, sessions, CSRF tokens, API tokens,\n"
    "role checks, upload checks, rate limits")
box(34.5, 26.0, 29.5, 13.0, "MySQL database\naccounts, roles, assignments,\nscans, predictions,\nreviews, audit log")
box(30.0, 13.0, 34.0, 9.0, "File storage\nuploaded scans and returned heatmaps")
arrow(49.25, 44.0, 49.25, 39.0, both=True)
arrow(32.25, 44.0, 32.25, 22.0, both=True)

arrow(19.3, 55.0, 30.0, 55.0)
note(24.65, 59.0, "HTTPS\nsession")
arrow(19.3, 45.5, 30.0, 45.5)
note(24.0, 40.6, "HTTPS\nbearer\ntoken")

# inference service on the free tier
frame(81.0, 31.0, 38.7, 37.0, "Inference service", "Render free tier: 0.1 CPU, 512 MB")
box(83.0, 48.0, 34.7, 10.5, "FastAPI, one uvicorn worker\nAPI-key check, size and pixel limits")
box(83.0, 33.5, 34.7, 12.0, "ONNX Runtime, one thread\nConvNeXt-Tiny with Grad-CAM branch\nSHA-256 checked at start-up")
arrow(100.35, 48.0, 100.35, 45.5, both=True)

arrow(64.0, 54.5, 83.0, 54.5)
note(73.5, 58.2, "POST /api/predict\nimage, API key")
arrow(83.0, 51.0, 64.0, 51.0)
note(73.5, 46.0, "prediction,\nprobabilities,\nheatmap")

# repository and releases
box(81.0, 12.5, 17.5, 12.0, "Public GitHub\nrepository\ncode, lock file,\nchecksums")
box(101.2, 12.5, 18.5, 12.0, "GitHub releases\nFP32 and\nINT8-dynamic\nONNX models")
arrow(89.75, 24.5, 89.75, 31.0)
note(90.6, 27.8, "build\non push", ha="left")
arrow(110.45, 24.5, 110.45, 31.0)
note(109.6, 27.8, "download\nat start-up", ha="right")

# where the measurements were taken
box(88.0, 72.5, 24.7, 6.5, "Load-test script (authors' PC)", dashed=True)
arrow(100.35, 72.5, 100.35, 68.0, dashed=True)
box(28.0, 0.5, 38.0, 7.5, "Security tests: a local copy of both parts,\nnever the public site", dashed=True)

for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig1_architecture.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight",
                pad_inches=0.03)
print("written", OUT / "fig1_architecture.png")
