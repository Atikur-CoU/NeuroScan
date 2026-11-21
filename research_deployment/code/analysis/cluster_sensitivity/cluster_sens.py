"""
NeuroScan paper 2 -- K2, JTCSST revision (Reviewer 2, comment 2): how much do the Table 5 confidence intervals depend on
treating every Kaggle_CT slice as its own patient cluster?  CPU only; no images, no model, no new inference.

Inputs (saved per-slice TTA probabilities, 10,602 held-out rows):
  * neuroscan-p2-int8dyn-served / e2_per_row_probs.csv : FP32, INT8-dyn (served, one request per slice), INT8-static
  * neuroscan-p2-static-calibration / per_row_probs.csv : FP32, S1-percentile, S2-entropy, S3-minmax-first-fp32
The FP32 columns of the two files must agree row by row (checked).

Population: the pooled internal test (cohort != SOOP; 8,930 slices), as in Table 5. Decision rule: plain argmax.
Measures, each copy against FP32: argmax agreement; paired differences (copy minus FP32) in accuracy, balanced accuracy
and normal / ischemic / hemorrhagic recall. 95% percentile intervals from a stratified cluster bootstrap (2,000
resamples; clusters resampled with replacement within each dataset), implemented with cluster multiplicities as weights,
which is the same estimator as concatenating the resampled clusters. The same resamples are used for every copy.

Clustering schemes (only Kaggle_CT changes; every other dataset keeps its patient clusters):
  A  published: each Kaggle_CT slice is its own cluster (reference; should reproduce Table 5 to sampling error)
  B  Kaggle_CT excluded (7,914 slices)
  C  Kaggle_CT pseudo-patients: slices grouped by blocks of the original file number (number // W) for W = 20, 50,
     100, 250, 500. The source files are numbered 10005-17033 across all class folders; whether nearby numbers come
     from one patient is NOT known, so these are sensitivity scenarios of increasing cluster size, not patient IDs.
  D  Kaggle_CT as one cluster per class folder (the coarsest grouping; 4 clusters).
Descriptive sensitivity analysis added in revision; no new hypothesis test.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

T0 = time.time()
LABELS = ("normal", "ischemic", "hemorrhagic")
N_BOOT, SEED = 2000, 20261120
WIDTHS = (20, 50, 100, 250, 500)
OUT = Path("/kaggle/working/output")
OUT.mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{(time.time() - T0) / 60:5.2f} min] {m}", flush=True)


def first(*cands):
    for c in cands:
        if Path(c).exists():
            return Path(c)
    print(f"[ERROR] not found; tried {cands}")
    sys.exit(1)


def nb_file(slug, name):
    roots = (f"/kaggle/input/notebooks/atik68/{slug}", f"/kaggle/input/{slug}")
    return first(*(f"{r}{sub}/{name}" for r in roots for sub in ("/output", "")))


# ------------------------------------------------------------------ inputs: two small CSVs, read once each
t = time.time()
F_SERVED = nb_file("neuroscan-p2-int8dyn-served", "e2_per_row_probs.csv")
F_STATIC = nb_file("neuroscan-p2-static-calibration", "per_row_probs.csv")
a = pd.read_csv(F_SERVED)
b = pd.read_csv(F_STATIC)
log(f"read {F_SERVED} ({len(a)} rows) and {F_STATIC} ({len(b)} rows) in {time.time() - t:.1f} s")
assert len(a) == len(b) == 10602, (len(a), len(b))
assert (a["out_path"].to_numpy() == b["out_path"].to_numpy()).all(), "row order differs between the two files"
fp_a = a[[f"p_{l}_FP32" for l in LABELS]].to_numpy(float)
fp_b = b[[f"p_{l}_FP32" for l in LABELS]].to_numpy(float)
assert np.abs(fp_a - fp_b).max() < 1e-5 and (fp_a.argmax(1) == fp_b.argmax(1)).all(), "FP32 columns differ"

COPIES = {"INT8-dyn (served)": (a, "INT8-dyn"), "INT8-static (min-max)": (a, "INT8-static"),
          "static, percentile": (b, "S1-percentile"), "static, entropy": (b, "S2-entropy"),
          "static, first layers FP32": (b, "S3-minmax-first-fp32")}
for name, (src, col) in COPIES.items():
    cols = [f"p_{l}_{col}" for l in LABELS]
    assert all(c in src.columns for c in cols), (name, cols, list(src.columns))
    assert not src[cols].isna().any().any(), name
if "p_normal_INT8-static" in b.columns:  # the min-max copy appears in both files: check it is the same model output
    log(f"INT8-static in both files, max |dp| = {np.abs(a[[f'p_{l}_INT8-static' for l in LABELS]].to_numpy() - b[[f'p_{l}_INT8-static' for l in LABELS]].to_numpy()).max():.2e}")

pool = (a["cohort"] != "SOOP").to_numpy()
df = a.loc[pool, ["source_key", "out_path", "dataset", "label", "cohort", "pk"]].reset_index(drop=True)
y = df["label"].map({l: i for i, l in enumerate(LABELS)}).to_numpy()
PF = fp_a[pool].argmax(1)
PRED = {n: src.loc[pool, [f"p_{l}_{c}" for l in LABELS]].to_numpy(float).argmax(1) for n, (src, c) in COPIES.items()}
assert len(df) == 8930, len(df)
kag = (df["dataset"] == "Kaggle_CT").to_numpy()
num = df["source_key"].str.extract(r"/(\d+)\.png$")[0]
assert num[kag].notna().all(), "a Kaggle_CT source file is not numbered"
num = num.astype(float)
folder = df["source_key"].str.split("/").str[-3]
log(f"pooled internal test: {len(df)} slices, {df['pk'].nunique()} clusters as published; Kaggle_CT {kag.sum()} slices, "
    f"file numbers {int(num[kag].min())}-{int(num[kag].max())}, folders {folder[kag].value_counts().to_dict()}")
R = {"inputs": {"served": str(F_SERVED), "static": str(F_STATIC)}, "n_boot": N_BOOT, "seed": SEED,
     "population": "pooled internal test (cohort != SOOP)", "n_slices": int(len(df)),
     "note": "descriptive sensitivity analysis added in revision; Kaggle_CT file-number blocks are scenarios, not patient IDs",
     "changed_slices": {n: {"all": int((p != PF).sum()), "Kaggle_CT": int((p != PF)[kag].sum())} for n, p in PRED.items()},
     "schemes": {}}
log(f"changed slices (all / in Kaggle_CT): {R['changed_slices']}")


# ------------------------------------------------------------------ measures with bootstrap weights
def measures(w, yy, pf, pv):
    """Weighted (w = cluster multiplicity per slice) agreement and copy-minus-FP32 metric differences."""
    sw = w.sum()
    out = {"agreement": (w * (pf == pv)).sum() / sw,
           "accuracy": ((w * (pv == yy)).sum() - (w * (pf == yy)).sum()) / sw}
    recs = []
    for c, l in enumerate(LABELS):
        wc = (w * (yy == c)).sum()
        if wc == 0:
            continue
        r = ((w * ((yy == c) & (pv == c))).sum() - (w * ((yy == c) & (pf == c))).sum()) / wc
        out[f"{l}_recall"] = r
        recs.append(r)
    out["balanced_accuracy"] = float(np.mean(recs)) if len(recs) == 3 else np.nan
    return out


def run(scheme, keep, cluster_key):
    """keep: boolean mask over df; cluster_key: per-slice cluster label (strings), stratified by dataset."""
    t = time.time()
    d = df[keep]
    yy, pf = y[keep], PF[keep]
    key = (d["dataset"] + "|" + cluster_key[keep]).to_numpy()
    codes, uniq = pd.factorize(key)
    strata = pd.Series(uniq).str.split("|").str[0].to_numpy()
    by_stratum = [np.flatnonzero(strata == s) for s in pd.unique(strata)]
    rng = np.random.default_rng(SEED)
    point = {n: measures(np.ones(len(d)), yy, pf, PRED[n][keep]) for n in PRED}
    boots = {n: {m: [] for m in point[n]} for n in PRED}
    for _ in range(N_BOOT):
        mult = np.zeros(len(uniq))
        for cl in by_stratum:
            mult += np.bincount(cl[rng.integers(0, len(cl), len(cl))], minlength=len(uniq))
        w = mult[codes]
        for n in PRED:
            for m, v in measures(w, yy, pf, PRED[n][keep]).items():
                boots[n][m].append(v)
    res = {"n_slices": int(len(d)), "n_clusters": int(len(uniq)),
           "kaggle_ct_clusters": int(len(pd.unique(key[d["dataset"].to_numpy() == "Kaggle_CT"]))),
           "copies": {}}
    for n in PRED:
        res["copies"][n] = {}
        for m, v in point[n].items():
            bv = np.array(boots[n][m], float)
            bv = bv[~np.isnan(bv)]
            lo, hi = np.percentile(bv, [2.5, 97.5])
            res["copies"][n][m] = {"point": float(v), "ci95": [float(lo), float(hi)], "ci_width": float(hi - lo)}
    R["schemes"][scheme] = res
    ag = res["copies"]["INT8-dyn (served)"]["agreement"]
    log(f"{scheme}: {res['n_slices']} slices, {res['n_clusters']} clusters ({res['kaggle_ct_clusters']} Kaggle_CT); "
        f"INT8-dyn agreement {ag['point']:.4%} [{ag['ci95'][0]:.4%}, {ag['ci95'][1]:.4%}] ({time.time() - t:.0f} s)")
    json.dump(R, open(OUT / "cluster_sensitivity.json", "w"), indent=2, default=float)


ALLROWS = np.ones(len(df), bool)
pk = df["pk"].astype(str)
run("A_published_each_KaggleCT_slice_own_cluster", ALLROWS, pk)
run("B_KaggleCT_excluded", ~kag, pk)
for W in WIDTHS:
    run(f"C_KaggleCT_file_number_blocks_W{W}", ALLROWS,
        pk.where(~kag, "kblock:" + (num // W).astype("Int64").astype(str)))
run("D_KaggleCT_one_cluster_per_class_folder", ALLROWS, pk.where(~kag, "kfolder:" + folder))

# ------------------------------------------------------------------ flat table for the supplement
rows = []
for s, res in R["schemes"].items():
    for n, ms in res["copies"].items():
        for m, v in ms.items():
            rows.append({"scheme": s, "n_slices": res["n_slices"], "n_clusters": res["n_clusters"],
                         "kaggle_ct_clusters": res["kaggle_ct_clusters"], "copy": n, "measure": m,
                         "point": round(v["point"], 6), "ci_low": round(v["ci95"][0], 6),
                         "ci_high": round(v["ci95"][1], 6), "ci_width": round(v["ci_width"], 6),
                         "ci_excludes_zero": (None if m == "agreement" else bool(v["ci95"][0] > 0 or v["ci95"][1] < 0))})
pd.DataFrame(rows).to_csv(OUT / "cluster_sensitivity.csv", index=False)
json.dump(R, open(OUT / "cluster_sensitivity.json", "w"), indent=2, default=float)
log(f"wrote cluster_sensitivity.json and cluster_sensitivity.csv ({len(rows)} rows). Done.")
