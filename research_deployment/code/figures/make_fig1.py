"""Figure 1 of paper 2: the deployed platform, from the repository's own configuration
(render_inference_server/start.sh, Dockerfile, app.py; webapp_php/includes/inference_client.php; android ApiService),
and where each test ran. Black and white, Times New Roman, drawn 6.6 in wide to be printed at about the full text width
(author's request, 21 Nov 2026: larger, readable type). Usage: python -I make_fig1.py"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

for f in ("times", "timesbd", "timesi", "timesbi"):
    font_manager.fontManager.addfont(f"C:/Windows/Fonts/{f}.ttf")
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 7.6})
OUT = Path(__file__).resolve().parent
INK, GREY = "black", "0.90"
DASH = (0, (4, 3))
BOX, NOTE, TITLE, SUB = 7.6, 7.4, 9.0, 7.4

fig = plt.figure(figsize=(6.6, 4.6))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 120)
ax.set_ylim(0, 80)
ax.axis("off")


def frame(x, y, w, h, title, sub):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5", fc=GREY, ec=INK, lw=1.1))
    ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="center", weight="bold", fontsize=TITLE)
    ax.text(x + w / 2, y + h - 5.2, sub, ha="center", va="top", style="italic", fontsize=SUB, linespacing=1.15)


def box(x, y, w, h, text, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc="white", ec=INK, lw=1.0,
                                ls=DASH if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=BOX, linespacing=1.2)


def arrow(x0, y0, x1, y1, both=False, dashed=False):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="<|-|>" if both else "-|>", lw=1.0, color=INK,
                                                         shrinkA=0, shrinkB=0, mutation_scale=8,
                                                         ls=DASH if dashed else "-"))


def note(x, y, text, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=NOTE, style="italic", linespacing=1.15)


# clients
box(0.3, 50.5, 19.5, 11.5, "Web browser\npatients, doctors,\nradiologists, admins")
box(0.3, 35.5, 19.5, 11.5, "Android app\nsame accounts\nand roles")

# web application on shared hosting
frame(27.5, 10.0, 39.0, 59.0, "Web application", "shared cPanel hosting\nApache, PHP 8.2, MariaDB 10.11")
box(29.5, 43.0, 35.0, 14.0, "Web pages and JSON API\nlogin, sessions, CSRF and API tokens,\nrole and upload checks, rate limits")
box(34.5, 25.5, 30.0, 13.5, "MariaDB database\naccounts, roles, assignments,\nscans, predictions,\nreviews, audit log")
box(29.5, 12.5, 35.0, 9.5, "File storage\nuploaded scans, returned heatmaps")
arrow(49.5, 43.0, 49.5, 39.0, both=True)
arrow(31.75, 43.0, 31.75, 22.0, both=True)

arrow(19.8, 55.0, 29.5, 55.0)
note(24.65, 59.0, "HTTPS\nsession")
arrow(19.8, 45.0, 29.5, 45.0)
note(24.65, 40.2, "HTTPS\nbearer\ntoken")

# inference service on the free tier
frame(81.0, 31.0, 38.7, 38.0, "Inference service", "Render free tier: 0.1 CPU, 512 MB")
box(83.0, 49.5, 34.7, 10.0, "FastAPI, one uvicorn worker\nAPI-key check, size and pixel limits")
box(83.0, 33.0, 34.7, 12.5, "ONNX Runtime, one thread\nConvNeXt-Tiny with Grad-CAM branch\nSHA-256 checked at start-up")
arrow(100.35, 49.5, 100.35, 45.5, both=True)

arrow(64.5, 55.0, 83.0, 55.0)
note(73.75, 58.6, "POST /api/predict\nimage, API key")
arrow(83.0, 51.5, 64.5, 51.5)
note(73.75, 46.6, "prediction,\nprobabilities,\nheatmap")

# repository and releases
box(81.0, 12.0, 17.5, 12.5, "Public GitHub\nrepository\ncode, lock file,\nchecksums")
box(101.2, 12.0, 18.5, 12.5, "GitHub releases\nFP32 and\nINT8-dynamic\nONNX models")
arrow(89.75, 24.5, 89.75, 31.0)
note(90.6, 27.8, "build\non push", ha="left")
arrow(110.45, 24.5, 110.45, 31.0)
note(109.6, 27.8, "download\nat start-up", ha="right")

# where each test ran
box(88.0, 72.0, 24.7, 7.5, "Load tests\nfrom the authors' PC", dashed=True)
arrow(100.35, 72.0, 100.35, 69.0, dashed=True)
box(27.5, 0.5, 39.0, 7.5, "Security tests: local copy of both parts;\npublic site: plain requests only", dashed=True)
box(0.3, 17.5, 24.0, 10.5, "Compression and\nGrad-CAM tests: Kaggle\nCPU, published models", dashed=True)
box(0.3, 3.5, 24.0, 10.5, "Rebuild check: clean\nclone and environment\non the authors' PC", dashed=True)

for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig1_architecture.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight",
                pad_inches=0.03)
print("written", OUT / "fig1_architecture.png")
