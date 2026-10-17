"""Supplementary Figure S3: entity-relationship schema of the clinical platform, from webapp_php/sql/schema.sql plus
migrations 002-005 (api_tokens; heatmap and input-file flag; context files; input pathway and warnings)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for _f in ("NotoSans-Regular.ttf", "NotoSans-Bold.ttf", "NotoSans-Italic.ttf"):
    font_manager.fontManager.addfont(rf"C:\Windows\Fonts\{_f}")
plt.rcParams.update({"font.family": ["Noto Sans", "DejaVu Sans"]})
from matplotlib.patches import Rectangle

OUT = Path(r"E:\CoU Projects\NeuroScan\NeuroScan_MultiDataset\paper\figures\v11_figS3_er_diagram.png")
INK, MUTED, HEAD = "black", "#404040", "#e6e6e6"
T = {
    "patient_profiles": ["PK/FK user_id → users", "date_of_birth, sex, mrn"],
    "doctor_patient": ["PK id", "FK patient_id → users", "FK provider_id → users", "relationship_type, is_active"],
    "api_tokens": ["PK id", "FK user_id → users", "token_hash, device_label", "expires_at"],
    "settings": ["PK id, setting_key (UQ)", "FK updated_by → users", "setting_value"],
    "users": ["PK id", "role (patient / radiologist /", "  doctor / admin)", "full_name, email", "password_hash",
              "FK created_by → users"],
    "audit_log": ["PK id", "FK actor_user_id → users", "action, target_type, target_id", "ip_address, details"],
    "login_attempts": ["PK id", "email, ip_address, success", "(no foreign key)"],
    "predictions": ["PK id", "FK/UQ scan_id → scans", "prediction, prob_*", "heatmap_path, likely_not_a_scan",
                    "pathway, evaluated_pathway", "warnings, decision_note", "inference_endpoint"],
    "scans": ["PK id, public_uuid", "FK patient_id → users", "FK uploaded_by → users", "modality, storage_path",
              "context_files", "sha256_hash, status"],
    "reviews": ["PK id", "FK scan_id → scans", "FK reviewer_id → users", "diagnosis_override, severity",
                "FK amends_review_id → reviews", "status (draft / final)"],
}
POS = {"patient_profiles": (0, 0), "doctor_patient": (1, 0), "api_tokens": (2, 0),
       "settings": (0, 1), "users": (1, 1), "audit_log": (2, 1), "login_attempts": (3, 1),
       "predictions": (0, 2), "scans": (1, 2), "reviews": (2, 2)}
W, LH, HH, GX, GY = 3.0, 0.32, 0.42, 4.0, 3.4
fig, ax = plt.subplots(figsize=(14, 9.6))
ax.set_xlim(-0.3, 3 * GX + W + 0.3)
ax.set_ylim(-(2 * GY + 3.0), 0.15)
ax.axis("off")
box = {}
for name, (cx, cy) in POS.items():
    x, y = cx * GX, -cy * GY
    h = HH + LH * len(T[name])
    ax.add_patch(Rectangle((x, y - h), W, h, fc="white", ec=INK, lw=1.2, zorder=2))
    ax.add_patch(Rectangle((x, y - HH), W, HH, fc=HEAD, ec=INK, lw=1.2, zorder=3))
    ax.text(x + W / 2, y - HH / 2, name, ha="center", va="center", color="black", fontweight="bold", fontsize=11, zorder=4)
    for i, f in enumerate(T[name]):
        ax.text(x + 0.12, y - HH - LH * (i + 0.5), f, ha="left", va="center", fontsize=9, color=INK, zorder=4,
                fontweight="bold" if f.startswith("PK") else "normal")
    box[name] = (x, y - h, W, h)


def edge(a, b, label):
    xa, ya, wa, ha = box[a]
    xb, yb, wb, hb = box[b]
    ca, cb = (xa + wa / 2, ya + ha / 2), (xb + wb / 2, yb + hb / 2)

    def border(x, y, w, h, tx, ty):
        cx, cy = x + w / 2, y + h / 2
        dx, dy = tx - cx, ty - cy
        s = min((w / 2) / abs(dx) if dx else 1e9, (h / 2) / abs(dy) if dy else 1e9)
        return cx + dx * s, cy + dy * s
    p0, p1 = border(xa, ya, wa, ha, *cb), border(xb, yb, wb, hb, *ca)
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2), zorder=1)
    ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, label, fontsize=8, color=MUTED, style="italic",
            ha="center", va="center", bbox=dict(fc="white", ec="none", pad=0.6), zorder=5)


for a, b, l in (("users", "patient_profiles", "1:1"), ("users", "doctor_patient", "1:N (two FKs)"),
                ("users", "api_tokens", "1:N"), ("users", "settings", "1:N"), ("users", "audit_log", "1:N"),
                ("users", "scans", "1:N"), ("users", "reviews", "1:N"), ("scans", "predictions", "1:1"),
                ("scans", "reviews", "1:N")):
    edge(a, b, l)

fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print("ok")


