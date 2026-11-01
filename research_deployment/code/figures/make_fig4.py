"""Figure 4 of paper 2: what the platform shows for five held-out slices under FP32, INT8-dynamic and INT8-static.
Inputs and maps come from Kaggle kernel atik68/neuroscan-p2-fig4-cases (one slice's 4 views per run, as the server);
overlays are drawn by the server's own code (render_inference_server/inference.py: display rule, tier tint, outline).
Noto Sans, labels under each panel (companion-paper conventions). Usage: python -I make_fig4.py"""
import json
import sys
import types
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
KOUT = ROOT / "kaggle_kernels" / "paper2_fig4_cases" / "out" / "output"
sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")   # the overlay code never touches it
sys.path.insert(0, str(ROOT / "render_inference_server"))
import inference as srv  # noqa: E402

for f in ("NotoSans-Regular", "NotoSans-Bold", "NotoSans-Italic"):
    font_manager.fontManager.addfont(f"C:/Windows/Fonts/{f}.ttf")
plt.rcParams.update({"font.family": "Noto Sans", "font.size": 7.5})

R = json.loads((KOUT / "fig4_cases.json").read_text())["cases"]
C = np.load(KOUT / "fig4_cases.npz")
COPIES = (("FP32", "FP32"), ("INT8-dyn", "INT8-dynamic"), ("INT8-static", "INT8-static"))
MRI = ("ISLES2022_dwi_adc", "SOOP_dwi_adc")
ROWS = {"A": "CT, ischemic (AISD)\nall copies agree",
        "B": "MRI, ischemic (ISLES 2022)\nall copies agree",
        "C": "CT, hemorrhagic (Kaggle CT)\nall copies agree",
        "D": "CT, normal (RSNA)\nchanged by INT8-dynamic",
        "E": "CT, normal (CPAISD)\nchanged by INT8-static"}


def margin(p):
    s = sorted(p)
    return s[2] - s[1]


fig, axes = plt.subplots(len(ROWS), 4, figsize=(7.1, 10.4), gridspec_kw={"wspace": 0.04, "hspace": 0.62})
for r, (name, title) in enumerate(ROWS.items()):
    case = R[name]
    rgb = Image.open(KOUT / f"case_{name}_input.png").convert("RGB")
    base = rgb.getchannel(0 if case["dataset"] in MRI else 1)   # the server's overlay base: DWI, or the current CT slice
    ax = axes[r, 0]
    ax.imshow(base, cmap="gray", vmin=0, vmax=255)
    ax.set_xlabel(f"{name}  {title}\nFP32 margin {margin(case['FP32']['probs']):.3f}", fontsize=7, linespacing=1.3)
    for c, (key, label) in enumerate(COPIES, start=1):
        v = case[key]
        pn, pi, ph = v["probs"]
        pred = v["pred"]
        unsupported = case["dataset"] in MRI and pred == "hemorrhagic"
        color = (127, 140, 141) if unsupported else srv.ThreeClassPipeline._risk_tier_color(pred, pn, pi, ph)
        cam = C[f"{name}_{key}_cams"][srv.VALID_LABELS.index(pred)]
        over = srv.ThreeClassPipeline._overlay_heatmap(base, srv.ThreeClassPipeline._display_cam(cam), color,
                                                       pred in ("ischemic", "hemorrhagic"))
        ax = axes[r, c]
        ax.imshow(over)
        p = dict(zip(srv.VALID_LABELS, v["probs"]))[pred]
        ax.set_xlabel(f"{label}\n{pred} {p:.3f}", fontsize=7, linespacing=1.3,
                      weight="bold" if pred != case["FP32"]["pred"] else "normal")
for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig4_compression_overlays.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight",
                pad_inches=0.03)
print("written", HERE / "fig4_compression_overlays.png")
