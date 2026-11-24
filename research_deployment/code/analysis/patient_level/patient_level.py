"""K9 (JTCSST second revision, comment 3): patient-level fidelity of every compressed copy, from the saved per-slice
outputs in this repository (no images needed).

    python -I patient_level.py <research_deployment/outputs> <output folder>

Inputs: int8dyn_served/e2_per_row_probs_served.csv (FP32, INT8-dynamic served one request per slice, INT8-static
min-max), static_calibration/per_row_probs.csv (percentile, entropy, first layers in FP32) and
mixed_precision/per_row_probs.csv (M1-M3). Every copy is scored under the deployed rule (argmax of the four-view mean
probabilities, as stored). Patients are the `pk` keys; Kaggle_CT has no patient identifiers, so each of its slices is
its own cluster and every patient-level result is also given for patients with identifiers only.

Per cohort group (pooled internal test = internal test + CPAISD; SOOP hold-out) and copy:
  1. patients with at least one changed slice (any change of the predicted class against FP32);
  2. patient-level calls under a triage rule: a patient is flagged hemorrhagic if any slice is predicted hemorrhagic,
     otherwise ischemic if any slice is predicted ischemic, otherwise normal; the truth uses the same rule on the labels.
     Reported: agreement of the patient-level call with FP32's; detection of ischemic and of hemorrhagic patients (at
     least one slice predicted the true class, as in the paper's patient-level ischemic detection); specificity on
     patients whose slices are all normal (no slice flagged);
  3. paired patient-cluster bootstrap 95% intervals (2,000 resamples, patients resampled within each dataset, seed 0)
     for the share of patients with a changed slice, call agreement, and the differences in detection and specificity.
Check: FP32 / INT8-dynamic ischemic detection must reproduce 265 / 263 of 281 (pooled internal test) and 63 / 63 of 66
(SOOP), the numbers in the paper."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
CLS = ["normal", "ischemic", "hemorrhagic"]

served = pd.read_csv(IN / "int8dyn_served" / "e2_per_row_probs_served.csv")
static = pd.read_csv(IN / "static_calibration" / "per_row_probs.csv")
mixed = pd.read_csv(IN / "mixed_precision" / "per_row_probs.csv")
df = served.copy()
for src, names in ((static, ["S1-percentile", "S2-entropy", "S3-minmax-first-fp32"]),
                   (mixed, ["M1-sens10", "M2-sens25", "M3-sens10-first"])):
    m = df[["out_path"]].merge(src, on="out_path", how="left", validate="one_to_one")
    assert m["out_path"].tolist() == df["out_path"].tolist() and m[f"p_normal_{names[0]}"].notna().all()
    # FP32 and INT8-static in the other runs must equal the served file's (same files, same rule)
    for k in ("FP32", "INT8-static"):
        assert (np.argmax(m[[f"p_{c}_{k}" for c in CLS]].to_numpy(), 1) ==
                np.argmax(df[[f"p_{c}_{k}" for c in CLS]].to_numpy(), 1)).all(), k
    for n in names:
        for c in CLS:
            df[f"p_{c}_{n}"] = m[f"p_{c}_{n}"].to_numpy()

COPIES = {"INT8-dynamic": "INT8-dyn", "INT8-static (min-max)": "INT8-static", "INT8-static, percentile": "S1-percentile",
          "INT8-static, entropy": "S2-entropy", "INT8-static, first layers in FP32": "S3-minmax-first-fp32",
          "M1: lowest 10% in FP32": "M1-sens10", "M2: lowest 25% in FP32": "M2-sens25",
          "M3: M1 and strided convolutions": "M3-sens10-first"}
pred = {k: np.argmax(df[[f"p_{c}_{k}" for c in CLS]].to_numpy(), 1) for k in ["FP32", *COPIES.values()]}
y = df["label"].map({c: i for i, c in enumerate(CLS)}).to_numpy()
df["key"] = df["dataset"] + "|" + df["pk"].astype(str)
df["group"] = np.where(df["cohort"] == "SOOP", "SOOP hold-out", "pooled internal test")
df["has_id"] = df["dataset"] != "Kaggle_CT"


def patient_table(rows, k):
    """One row per patient: changed slice?, patient-level call, truth, detection flags."""
    sub = df.loc[rows, ["key", "dataset"]].copy()
    a, b, yy = pred["FP32"][rows], pred[k][rows], y[rows]
    sub["changed"], sub["a_h"], sub["a_i"] = a != b, a == 2, a == 1
    sub["b_h"], sub["b_i"], sub["t_h"], sub["t_i"] = b == 2, b == 1, yy == 2, yy == 1
    sub["a_det_i"], sub["b_det_i"] = (yy == 1) & (a == 1), (yy == 1) & (b == 1)
    sub["a_det_h"], sub["b_det_h"] = (yy == 2) & (a == 2), (yy == 2) & (b == 2)
    g = sub.groupby(["dataset", "key"]).any().reset_index()

    def call(h, i):
        return np.where(g[h], 2, np.where(g[i], 1, 0))
    g["call_a"], g["call_b"], g["truth"] = call("a_h", "a_i"), call("b_h", "b_i"), call("t_h", "t_i")
    return g


def stats(g):
    isch, hem, norm = g["t_i"], g["t_h"], g["truth"] == 0
    return {"patients": len(g), "with_changed_slice": int(g["changed"].sum()),
            "call_agreement": float((g["call_a"] == g["call_b"]).mean()),
            "ischemic_patients": int(isch.sum()), "ischemic_detected_fp32": int(g.loc[isch, "a_det_i"].sum()),
            "ischemic_detected_copy": int(g.loc[isch, "b_det_i"].sum()),
            "hemorrhagic_patients": int(hem.sum()), "hemorrhagic_detected_fp32": int(g.loc[hem, "a_det_h"].sum()),
            "hemorrhagic_detected_copy": int(g.loc[hem, "b_det_h"].sum()),
            "normal_patients": int(norm.sum()), "normal_unflagged_fp32": int((g.loc[norm, "call_a"] == 0).sum()),
            "normal_unflagged_copy": int((g.loc[norm, "call_b"] == 0).sum())}


def boot(g, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    groups = [grp.index.to_numpy() for _, grp in g.groupby("dataset")]
    keys = ["share_changed", "call_agreement", "d_ischemic_detection", "d_hemorrhagic_detection", "d_normal_specificity"]
    vals = {k: [] for k in keys}
    for _ in range(n):
        ix = np.concatenate([gr[rng.integers(0, len(gr), len(gr))] for gr in groups])
        s = g.loc[ix]
        isch, hem, norm = s["t_i"].to_numpy(), s["t_h"].to_numpy(), (s["truth"] == 0).to_numpy()
        vals["share_changed"].append(s["changed"].mean())
        vals["call_agreement"].append((s["call_a"] == s["call_b"]).mean())
        vals["d_ischemic_detection"].append((s["b_det_i"][isch].mean() - s["a_det_i"][isch].mean()) if isch.any() else 0)
        vals["d_hemorrhagic_detection"].append((s["b_det_h"][hem].mean() - s["a_det_h"][hem].mean()) if hem.any() else 0)
        vals["d_normal_specificity"].append(((s["call_b"][norm] == 0).mean() - (s["call_a"][norm] == 0).mean())
                                            if norm.any() else 0)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in vals.items()}


R, rows_csv = {"rule": "patient flagged hemorrhagic if any slice predicted hemorrhagic, else ischemic if any slice "
               "ischemic, else normal; truth by the same rule on labels; detection = at least one slice of the true "
               "class predicted as that class", "bootstrap": "2,000 paired patient-cluster resamples within dataset, "
               "seed 0"}, []
for scope, idmask in (("all clusters", np.ones(len(df), bool)), ("patients with identifiers", df["has_id"].to_numpy())):
    for grp in ("pooled internal test", "SOOP hold-out"):
        rows = np.where(idmask & (df["group"] == grp).to_numpy())[0]
        for name, k in COPIES.items():
            g = patient_table(rows, k)
            s = stats(g)
            s["ci"] = boot(g)
            R.setdefault(scope, {}).setdefault(grp, {})[name] = s
            rows_csv.append({"scope": scope, "group": grp, "copy": name, **{k2: v for k2, v in s.items() if k2 != "ci"},
                             **{f"{k2}_lo": v[0] for k2, v in s["ci"].items()},
                             **{f"{k2}_hi": v[1] for k2, v in s["ci"].items()}})

chk = R["all clusters"]
assert (chk["pooled internal test"]["INT8-dynamic"]["ischemic_patients"],
        chk["pooled internal test"]["INT8-dynamic"]["ischemic_detected_fp32"],
        chk["pooled internal test"]["INT8-dynamic"]["ischemic_detected_copy"]) == (281, 265, 263), "check failed"
assert (chk["SOOP hold-out"]["INT8-dynamic"]["ischemic_patients"],
        chk["SOOP hold-out"]["INT8-dynamic"]["ischemic_detected_fp32"],
        chk["SOOP hold-out"]["INT8-dynamic"]["ischemic_detected_copy"]) == (66, 63, 63), "check failed"
json.dump(R, open(OUT / "patient_level.json", "w"), indent=2)
pd.DataFrame(rows_csv).to_csv(OUT / "patient_level.csv", index=False)
print("checks passed (281/265/263, 66/63/63); written", OUT)
