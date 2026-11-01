"""
NeuroScan paper 2 -- E2 follow-up after the expert review (2026-10-25): where do the INT8 copies change the served
model's decisions? CPU only; no images, no model, no new inference. Input = the saved per-row probabilities of kernel
neuroscan-p2-compression-gradcam (e2_per_row_probs.csv: FP32, INT8-dynamic, INT8-static; 10,602 held-out rows).
These analyses were added after the expert review, i.e. after the E2 results were seen; they are descriptive and are
reported as such. No non-inferiority margin is used.

Deployed rule throughout: plain argmax of the 4-view TTA probabilities.
  1. Per dataset (and per cohort): one-vs-rest sensitivity and specificity for each class present, FP32 and each
     INT8 copy, with the paired difference and its patient-cluster bootstrap CI (2,000 resamples; patients resampled
     within each dataset; Kaggle_CT has no patient IDs, so its slices are their own clusters).
  2. Transitions by truth: for each true label, the 3 x 3 table FP32 prediction -> INT8 prediction; changes split into
     correct->wrong, wrong->correct and wrong->wrong.
  3. Every slice whose prediction changes: dataset, cohort, patient key, truth, both predictions and probabilities,
     and the FP32 top-1 minus top-2 margin; the margin of changed slices against all slices.
  4. Patient level: ischemic patients detected (at least one ischemic slice predicted ischemic), FP32 vs INT8, and the
     patients whose detection changes.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

T0 = time.time()
LABELS = ("normal", "ischemic", "hemorrhagic")
VARIANTS = ("INT8-dyn", "INT8-static")
N_BOOT, SEED = 2000, 2025
OUT = Path("/kaggle/working/output")
OUT.mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {m}", flush=True)


def first(*cands):
    for c in cands:
        if Path(c).exists():
            return Path(c)
    print(f"[ERROR] not found; tried {cands}")
    sys.exit(1)


SRC = first(*(f"{r}/{sub}e2_per_row_probs.csv" for r in ("/kaggle/input/notebooks/atik68/neuroscan-p2-compression-gradcam",
                                                          "/kaggle/input/neuroscan-p2-compression-gradcam")
              for sub in ("output/", "")))
df = pd.read_csv(SRC)
assert len(df) == 10602 and not df.isna().any().any(), (len(df), df.isna().sum().to_dict())
y = df["label"].map({l: i for i, l in enumerate(LABELS)}).to_numpy()
P = {v: df[[f"p_{l}_{v}" for l in LABELS]].to_numpy(float) for v in ("FP32",) + VARIANTS}
PRED = {v: p.argmax(1) for v, p in P.items()}
srt = np.sort(P["FP32"], axis=1)
margin = srt[:, 2] - srt[:, 1]
R = {"source": str(SRC), "rows": len(df), "note": "descriptive, added after the expert review; no NI margin"}
log(f"loaded {SRC}: {len(df)} rows")


def clusters(sub):
    """Patient clusters within each dataset (as in the E2 kernel)."""
    return [[g.index.to_numpy()[ix] for ix in g.groupby("pk").indices.values()] for _, g in sub.groupby("dataset")]


def boot_idx(groups, rng):
    return np.concatenate([np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))]) for cl in groups])


def sens_spec(yy, pp, c):
    pos, neg = yy == c, yy != c
    se = float((pp[pos] == c).mean()) if pos.any() else np.nan
    sp = float((pp[neg] != c).mean()) if neg.any() else np.nan
    return se, sp


def paired(sub_idx, v, seed):
    sub = df.iloc[sub_idx].reset_index(drop=True)
    yy, pa, pb = y[sub_idx], PRED["FP32"][sub_idx], PRED[v][sub_idx]
    present = [c for c in range(3) if (yy == c).any()]
    point = {}
    for c in present:
        (sa, pa_), (sb, pb_) = sens_spec(yy, pa, c), sens_spec(yy, pb, c)
        point[LABELS[c]] = {"sens_fp32": sa, "sens_int8": sb, "spec_fp32": pa_, "spec_int8": pb_}
    groups, rng = clusters(sub), np.random.default_rng(seed)
    bs = {f"{LABELS[c]}_{k}": [] for c in present for k in ("sens", "spec")}
    for _ in range(N_BOOT):
        ix = boot_idx(groups, rng)
        for c in present:
            (sa, pa_), (sb, pb_) = sens_spec(yy[ix], pa[ix], c), sens_spec(yy[ix], pb[ix], c)
            if sa == sa and sb == sb:
                bs[f"{LABELS[c]}_sens"].append(sb - sa)
            if pa_ == pa_ and pb_ == pb_:
                bs[f"{LABELS[c]}_spec"].append(pb_ - pa_)
    for c in present:
        for k in ("sens", "spec"):
            v_ = np.array(bs[f"{LABELS[c]}_{k}"])
            d = point[LABELS[c]][f"{k}_int8"] - point[LABELS[c]][f"{k}_fp32"]
            point[LABELS[c]][f"{k}_diff"] = d
            point[LABELS[c]][f"{k}_diff_ci95"] = ([float(np.percentile(v_, 2.5)), float(np.percentile(v_, 97.5))]
                                                  if len(v_) > 50 else None)
    return {"n": int(len(sub)), "patients_or_clusters": int(sub["pk"].nunique()),
            "positives": {LABELS[c]: int((yy == c).sum()) for c in present}, "by_class": point}


# 1. per dataset and per cohort
R["sens_spec"] = {}
seed = SEED
groups_def = [(f"dataset:{d}", df.index[df["dataset"] == d].to_numpy()) for d in sorted(df["dataset"].unique())]
groups_def += [("cohort:pooled_internal_test", df.index[df["cohort"] != "SOOP"].to_numpy()),
               ("cohort:SOOP_holdout", df.index[df["cohort"] == "SOOP"].to_numpy())]
for v in VARIANTS:
    R["sens_spec"][v] = {}
    for name, idx in groups_def:
        seed += 1
        r = paired(idx, v, seed)
        R["sens_spec"][v][name] = r
        log(f"{v} {name}: n={r['n']} " + "; ".join(
            f"{c} sens {b['sens_fp32']:.4f}->{b['sens_int8']:.4f} ({b['sens_diff']:+.4f} {b['sens_diff_ci95']}), "
            f"spec {b['spec_fp32']:.4f}->{b['spec_int8']:.4f} ({b['spec_diff']:+.4f})" for c, b in r["by_class"].items()))
    json.dump(R, open(OUT / "fidelity_detail.json", "w"), indent=2, default=float)

# 2 + 3. transitions by truth, and every changed slice
R["transitions"], changed_rows = {}, []
for v in VARIANTS:
    ch = PRED["FP32"] != PRED[v]
    kind = np.where(~ch, "same", np.where(PRED["FP32"] == y, "correct_to_wrong",
                                           np.where(PRED[v] == y, "wrong_to_correct", "wrong_to_wrong")))
    t = {}
    for c in range(3):
        m = y == c
        if m.any():
            t[f"true_{LABELS[c]}"] = np.bincount(PRED["FP32"][m] * 3 + PRED[v][m], minlength=9).reshape(3, 3).tolist()
    R["transitions"][v] = {"tables_rows_fp32_cols_int8": t, "changed": int(ch.sum()),
                           "kinds": pd.Series(kind[ch]).value_counts().to_dict(),
                           "kinds_by_truth": pd.crosstab(df["label"][ch], kind[ch]).to_dict(),
                           "fp32_margin_median_all": float(np.median(margin)),
                           "fp32_margin_median_changed": float(np.median(margin[ch])) if ch.any() else None,
                           "fp32_margin_max_changed": float(margin[ch].max()) if ch.any() else None,
                           "share_changed_with_fp32_margin_below_0.1": float((margin[ch] < 0.1).mean()) if ch.any() else None}
    log(f"{v} transitions: {json.dumps(R['transitions'][v])}")
    for i in np.flatnonzero(ch):
        changed_rows.append({"variant": v, "dataset": df.at[i, "dataset"], "cohort": df.at[i, "cohort"],
                             "pk": df.at[i, "pk"], "out_path": df.at[i, "out_path"], "truth": df.at[i, "label"],
                             "pred_fp32": LABELS[PRED["FP32"][i]], "pred_int8": LABELS[PRED[v][i]], "kind": kind[i],
                             "fp32_margin": round(float(margin[i]), 5),
                             **{f"p_{l}_fp32": round(float(P["FP32"][i, j]), 5) for j, l in enumerate(LABELS)},
                             **{f"p_{l}_int8": round(float(P[v][i, j]), 5) for j, l in enumerate(LABELS)}})
pd.DataFrame(changed_rows).sort_values(["variant", "kind", "truth", "fp32_margin"]).to_csv(OUT / "changed_slices.csv", index=False)
log(f"changed_slices.csv: {len(changed_rows)} rows")

# 4. patient-level ischemic detection
isch = df[df["label"] == "ischemic"]
R["patient_level_ischemic"] = {}
for v in VARIANTS:
    out = {}
    for cname, sel in (("pooled_internal_test", isch["cohort"] != "SOOP"), ("SOOP_holdout", isch["cohort"] == "SOOP")):
        g = isch[sel]
        ix = g.index.to_numpy()
        det = pd.DataFrame({"pk": g["pk"].to_numpy(), "dataset": g["dataset"].to_numpy(),
                            "a": PRED["FP32"][ix] == 1, "b": PRED[v][ix] == 1}).groupby(["dataset", "pk"])[["a", "b"]].any()
        lost = det[det["a"] & ~det["b"]].index.tolist()
        gained = det[~det["a"] & det["b"]].index.tolist()
        out[cname] = {"ischemic_patients": int(len(det)), "detected_fp32": int(det["a"].sum()),
                      "detected_int8": int(det["b"].sum()), "lost": [list(x) for x in lost],
                      "gained": [list(x) for x in gained]}
    R["patient_level_ischemic"][v] = out
    log(f"{v} patient-level ischemic detection: {json.dumps(out)[:600]}")
json.dump(R, open(OUT / "fidelity_detail.json", "w"), indent=2, default=float)
log("Done.")
