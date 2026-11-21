"""Figures 3 and 4 of paper 2 (JTCSST revision; black and white): response time (3) and throughput (4) of the two matched free-tier services under concurrent users.
Data: the load-test logs in ../e1_load/e1_fp32 and ../e1_load/e1_int8 (requests.csv, summary.json); no other input.
Times New Roman (author's request, 21 Nov 2026), labels under each panel. Usage: python -I make_fig3_4.py"""
import csv
import json
import statistics
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
E1 = HERE.parent / "e1_load"
for f in ("times", "timesbd", "timesi", "timesbi"):
    font_manager.fontManager.addfont(f"C:/Windows/Fonts/{f}.ttf")
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 7.5, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})
SERVICES = (("e1_fp32", "FP32", "black", "o", "-"), ("e1_int8", "INT8-dynamic", "0.45", "s", (0, (4, 2))))
USERS = (1, 2, 4, 8)

fig1, ax1 = plt.subplots(figsize=(3.6, 1.9))
fig2, ax2 = plt.subplots(figsize=(3.4, 1.9))
rng = np.random.default_rng(0)   # jitter only
for i, (folder, label, color, marker, ls) in enumerate(SERVICES):
    rows = list(csv.DictReader(open(E1 / folder / "requests.csv", newline="")))
    summ = json.load(open(E1 / folder / "summary.json"))
    off = -0.17 if i == 0 else 0.17
    for k, u in enumerate(USERS):
        lat = [float(r["latency_s"]) for r in rows if r["phase"] == "steps" and int(r["concurrency"]) == u]
        assert len(lat) == 40
        x = k + off + rng.uniform(-0.07, 0.07, len(lat))
        ax1.scatter(x, lat, s=7, marker=marker, facecolor="none", edgecolor=color, linewidth=0.6, zorder=3,
                    label=label if k == 0 else None)
        ax1.plot([k + off - 0.12, k + off + 0.12], [statistics.median(lat)] * 2, color="black", lw=1.6, zorder=4)
    # throughput per step, and the ceiling for one request at a time: 60 s / the service's own time per request
    tput = [summ["steps"][str(u)]["throughput_ok_per_min"] for u in USERS]
    srv = statistics.median(float(r["server_request_ms"]) / 1000 for r in rows if r["phase"] == "warm")
    ax2.plot(USERS, tput, color="black", marker=marker, ms=4, mfc="white" if i == 0 else "0.45", lw=1.0, ls=ls,
             label=label)
    ax2.hlines(60 / srv, 0.8, 9.6, color="0.3", lw=0.7, ls=(0, (1, 2)))
    ax2.text(10.0, 60 / srv, f"{60 / srv:.1f}", color="black", va="center", fontsize=6.8)
    print(f"{label}: throughput {tput}, ceiling {60 / srv:.2f}/min (server time {srv:.2f} s)")

ax1.set_xticks(range(len(USERS)), [str(u) for u in USERS])
ax1.set_ylim(0, 120)
ax1.set_ylabel("Response time (s)")
ax1.legend(frameon=False, loc="upper left", handletextpad=0.3)
ax1.set_xlabel("Simultaneous users")
ax2.set_xscale("log", base=2)
ax2.set_xticks(USERS, [str(u) for u in USERS])
ax2.minorticks_off()
ax2.set_xlim(0.8, 13)
ax2.set_ylim(0, 25)
ax2.set_ylabel("Completed requests per minute")
ax2.legend(frameon=False, loc="lower right", handletextpad=0.4)
ax2.set_xlabel("Simultaneous users")
for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", lw=0.3, color="0.85")
    ax.set_axisbelow(True)
for fig, name in ((fig1, "fig3_response_time"), (fig2, "fig4_throughput")):
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight", pad_inches=0.03)
    print("written", HERE / f"{name}.png")
