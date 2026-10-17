"""
NeuroScan Phase Q -- POST-HOC analyses for post-hoc round 1 (helpers copied unchanged from the final evaluation).
NeuroScan Phase Q -- FINAL EVALUATION (one-time). Implements the final evaluation plan pre-registered on 2026-10-06
(paper/accuracy_improvement_options.md, row "FINAL EVALUATION PLAN"). CPU; scores probabilities saved in Phase 1;
no inference, no training, no model choice. Whatever the results, R30A s123 stays the final model.

Model: R30A s123 (primary). Decision: argmax(p_raw * (1, w_i, w_h)), weights from the locked original rule (41 x 41 log
grid, max validation macro-F1, ties -> nearest (1, 1)) -- recomputed here from the model's own Phase 1 validation
predictions and asserted to equal (0.574, 0.574) for s123. Replication: R30A s7 with weights from the same rule on its own
validation predictions. Calibration only: p ~ p_raw^(1/T), T_CT = 1.48, T_MRI = 0.80 (both models).
Cohorts: A = internal test minus CPAISD_v2 (7,406 rows), B = CPAISD_v2 test (1,524 rows / 18 patients),
C = SOOP 15% hold-out (1,672 rows / 66 patients); A+B pooled for the overall slice table.
Primary: slice-level metrics per cohort, per source, CT vs MRI, 5-source subset (descriptive), 95% CIs by
patient-cluster bootstrap (2,000 replicates, patients resampled within source; Kaggle_CT rows are own clusters).
Key secondary: patient-level case-detection sensitivity (any-slice; top-3 sensitivity analysis), per-patient
lesion-slice sensitivity, per-patient normal-slice false-positive fraction, detection by slices per patient.
Patient-level specificity / PPV / NPV: not estimable (no entirely normal patients) -- reported as such.
Calibration: ECE (15 bins), NLL, Brier, reliability diagrams. Descriptive: CPAISD core-only (legend confirmed in the
CPAISD paper: class 1 = stroke core, 2 = penumbra), AISD failure analysis (slice position, lesion size bins
< 500 / 500-1,500 / 1,500-3,500 / > 3,500 native px, mask value 2 vs 3), ISLES lesion-size strata.
DRY_RUN = True scores the Phase 1 VALIDATION files instead (for testing the code only; count checks relaxed).
"""
import hashlib
import json
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def ensure_pkg(import_name, pip_name=None):
    try:
        __import__(import_name)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name or import_name], check=True)


DRY_RUN = False
for mod in ("gdown", "statsmodels", "nibabel", "matplotlib"):
    try:
        ensure_pkg(mod)
    except Exception:  # noqa: BLE001  (local dry runs without internet)
        pass

import numpy as np
import pandas as pd
from PIL import Image

INPUT_ROOT = Path("/kaggle/input")
DS = INPUT_ROOT / "datasets/atik68"
MODELS = {"round30as123": DS / "neuroscan-q-round30as123-preds", "round30as7": DS / "neuroscan-q-round30as7-preds"}
PRIMARY = "round30as123"
LOCKED_W_PRIMARY = (0.574, 0.574)
PHASE2 = DS / "neuroscan-q-phase2-split-data"
CPAISD_RAW = DS / "neuroscan-cpaisd"
ISLES_RAW = DS / "neuroscan-isles2022"
AISD_MASK_ZIP_ID = "1d08fFpEvK4D6YTKfRlNuv_OlIxigZxl6"
T_CT, T_MRI = 1.48, 0.80
MRI_SOURCES = ("ISLES2022_dwi_adc", "SOOP_dwi_adc")
LABELS = ("normal", "ischemic", "hemorrhagic")
L2I = {l: i for i, l in enumerate(LABELS)}
GRID = np.exp(np.linspace(np.log(0.25), np.log(4.0), 41))
N_BOOT, N_BINS, TOPK = 2000, 15, 3
EXPECT = {"test": 8930, "cpaisd_test": 1524, "cpaisd_patients": 18, "soop": 1672, "soop_patients": 66}
AISD_BINS, AISD_BIN_NAMES = [0, 500, 1500, 3500, 10 ** 9], ["<500", "500-1500", "1500-3500", ">3500"]
ISLES_BINS = [-1, 0, 10, 25, 50, 100, 250, 1000, 10 ** 9]
ISLES_BIN_NAMES = ["0", "1-10", "11-25", "26-50", "51-100", "101-250", "251-1000", ">1000"]
SLICE_COUNT_BINS, SLICE_COUNT_NAMES = [0, 5, 10, 20, 10 ** 6], ["1-5", "6-10", "11-20", ">20"]
CPAISD_RE = re.compile(r"/([\d.]{10,})/(\d+)/raw\.dcm$")
MRI_RE = re.compile(r"^(sub-[\w]+)/(ses-\w+)/")
ISLES_KEY_RE = re.compile(r"^(sub-strokecase\d+)/(ses-\d+)/\w+/slice(\d+)$")
OUT = Path("/kaggle/working/output")
TMP = Path("/tmp/final_eval")
T0 = time.time()
REPORT = {"plan": "FINAL EVALUATION PLAN, pre-registered 2026-10-06", "dry_run": DRY_RUN}


def log(m):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {m}", flush=True)


def need(p):
    p = Path(p)
    if p.exists():
        return p
    for cand in [INPUT_ROOT / p.parent.name, *INPUT_ROOT.glob(f"*/{p.parent.name}"), *INPUT_ROOT.glob(f"*/*/{p.parent.name}")]:
        if (cand / p.name).exists():
            return cand / p.name
    print(f"[ERROR] missing {p}; mounted: {sorted(str(x) for x in INPUT_ROOT.glob('*/*/*') if x.is_dir())[:40]}")
    sys.exit(1)


def find_dir(root, name, depth=3):
    """Directory called `name` at most `depth` levels below root (directories only; no image listing)."""
    frontier = [Path(root)]
    for _ in range(depth + 1):
        nxt = []
        for d in frontier:
            if (d / name).is_dir():
                return d / name
            nxt += [c for c in d.iterdir() if c.is_dir()] if d.is_dir() and len(nxt) < 200 else []
        frontier = nxt
    return None


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and o != o:
        return None
    return o


# ------------------------------------------------------------------ metrics (numpy only)
def auroc(yb, s):
    n1 = int(yb.sum())
    n0 = len(yb) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return float((r[yb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pr_auc(yb, s):
    """Average precision (step-wise, as sklearn.average_precision_score)."""
    n1 = int(yb.sum())
    if n1 == 0 or n1 == len(yb):
        return np.nan
    o = np.argsort(-s, kind="mergesort")
    s, yb = s[o], yb[o]
    tp = np.cumsum(yb)
    idx = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tp, k = tp[idx], idx + 1
    prec, rec = tp / k, tp / n1
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


def slice_metrics(y, pred, p):
    """y, pred: int arrays; p: raw probabilities (ranking metrics are invariant to the weights and temperatures)."""
    r = {"n": int(len(y)), "accuracy": float((pred == y).mean())}
    present = [c for c in range(3) if (y == c).any()]
    recs, f1s = [], []
    cm = np.bincount(y * 3 + pred, minlength=9).reshape(3, 3)
    r["confusion_matrix_rows_true_cols_pred"] = cm.tolist()
    for c in range(3):
        name = LABELS[c]
        tp, fn = cm[c, c], cm[c].sum() - cm[c, c]
        fp, tn = cm[:, c].sum() - cm[c, c], cm.sum() - cm[c].sum() - cm[:, c].sum() + cm[c, c]
        if tp + fn:
            r[f"{name}_recall"] = tp / (tp + fn)
            r[f"{name}_f1"] = 2 * tp / (2 * tp + fp + fn)
        if tp + fp:
            r[f"{name}_precision"] = tp / (tp + fp)
        if tn + fp:
            r[f"{name}_specificity"] = tn / (tn + fp)
        if tn + fn:
            r[f"{name}_npv"] = tn / (tn + fn)
        if c in present:
            recs.append(tp / (tp + fn))
            f1s.append(2 * tp / (2 * tp + fp + fn))
    r["balanced_accuracy"] = float(np.mean(recs))
    r["macro_f1_present_classes"] = float(np.mean(f1s))
    aucs = []
    for c in present:
        yb = (y == c).astype(int)
        a = auroc(yb, p[:, c])
        if a == a:
            aucs.append(a)
        if c > 0:
            r[f"{LABELS[c]}_auroc"] = a
            r[f"{LABELS[c]}_pr_auc"] = pr_auc(yb, p[:, c])
    if len(aucs) >= 2:
        r["macro_ovr_auroc"] = float(np.mean(aucs))
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r.items()}


CI_KEYS = ["accuracy", "balanced_accuracy", "macro_f1_present_classes", "macro_ovr_auroc",
           "normal_recall", "normal_specificity", "ischemic_recall", "ischemic_precision", "ischemic_specificity",
           "hemorrhagic_recall", "hemorrhagic_precision", "hemorrhagic_specificity",
           "ischemic_auroc", "ischemic_pr_auc", "hemorrhagic_auroc", "hemorrhagic_pr_auc"]


def with_ci(df, m, w, seed):
    y = df["label"].map(L2I).to_numpy()
    p = df[pcols(m)].to_numpy(float)
    pred = decide(p, w)
    point = slice_metrics(y, pred, p)
    d0 = df.reset_index(drop=True)
    groups = [[g.index.to_numpy()[ix] for ix in g.groupby("pk").indices.values()] for _, g in d0.groupby("dataset")]
    rng = np.random.default_rng(seed)
    boot = {k: [] for k in CI_KEYS}
    for _ in range(N_BOOT):
        idx = np.concatenate([np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))]) for cl in groups])
        b = slice_metrics(y[idx], pred[idx], p[idx])
        for k in CI_KEYS:
            if k in b and k in point and b[k] == b[k]:
                boot[k].append(b[k])
    point["ci95"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items() if len(v) > 50}
    point["argmax_accuracy"] = float((p.argmax(1) == y).mean())
    point["patients"] = int(df["pk"].nunique())
    return point


# ------------------------------------------------------------------ data
def pcols(m):
    return [f"p_{l}_{m}" for l in LABELS]


def decide(p, w):
    return (p * np.array([1.0, w[0], w[1]])).argmax(1)


def macro_f1_3(y, pred):
    f = []
    for c in range(3):
        tp, pos, prd = ((pred == c) & (y == c)).sum(), (y == c).sum(), (pred == c).sum()
        f.append(2 * tp / (pos + prd) if pos + prd else 0.0)
    return float(np.mean(f))


def original_rule(y, p):
    best, key = None, None
    for wi in GRID:
        for wh in GRID:
            k = (round(macro_f1_3(y, decide(p, (wi, wh))), 10), -np.hypot(np.log(wi), np.log(wh)))
            if key is None or k > key:
                best, key = (float(wi), float(wh)), k
    return best, key[0]


RSNA_MAP = {}


def patient_key(ds, sk, op, subject):
    sk, op = str(sk), str(op)
    if ds == "AISD":
        return f"AISD:{sk.split('/')[2]}"
    if ds == "CPAISD_v2":
        m = CPAISD_RE.search(sk.replace("\\", "/"))
        return f"CPAISD:{m.group(1)}" if m else f"CPAISD_row:{op}"
    if ds == "ISLES2022_dwi_adc":
        m = MRI_RE.match(sk)
        return f"ISLES:{m.group(1)}/{m.group(2)}" if m else f"ISLES_row:{sk}"
    if ds == "SOOP_dwi_adc":
        if isinstance(subject, str) and subject:
            return f"SOOP:{subject}"
        m = MRI_RE.match(sk)
        return f"SOOP:{m.group(1)}" if m else f"SOOP_row:{sk}"
    if ds == "RSNA_ICH":
        stem = Path(op).stem
        pid = RSNA_MAP.get(stem)
        return f"RSNA:{pid}" if isinstance(pid, str) and pid else f"RSNA_row:{stem}"
    if ds == "SICH_CT":
        return f"SICH:{sk.replace(chr(92), '/').split('/')[-2]}"
    return f"{ds}_row:{op}"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(file_name):
    base = None
    for m, d in MODELS.items():
        f = need(d / file_name)
        REPORT.setdefault("sha256", {})[f"{m}/{file_name}"] = sha256(f)
        df = pd.read_csv(f)
        if base is None:
            base = df[["source_key", "dataset", "label", "out_path", "subject", "slice"] + pcols(m)].copy()
        else:
            j = base[["source_key", "out_path", "label"]].merge(df[["source_key", "out_path", "label"] + pcols(m)],
                                                               on=["source_key", "out_path"], how="inner", validate="one_to_one")
            assert len(j) == len(base) and (j["label_x"] == j["label_y"]).all(), f"{m}: rows/labels differ"
            base = base.merge(df[["source_key", "out_path"] + pcols(m)], on=["source_key", "out_path"], validate="one_to_one")
    base["pk"] = [patient_key(a, b, c, s) for a, b, c, s in zip(base["dataset"], base["source_key"], base["out_path"], base["subject"])]
    return base.reset_index(drop=True)


# ------------------------------------------------------------------ patient level
def patient_level(df, m, w):
    x = df[~df["pk"].str.contains("_row:")].copy()
    p = x[pcols(m)].to_numpy(float) * np.array([1.0, w[0], w[1]])
    q = p / p.sum(1, keepdims=True)
    x["pred"] = np.array(LABELS)[q.argmax(1)]
    for i, l in enumerate(LABELS):
        x[f"q_{l}"] = q[:, i]
    out = {}
    for ds, g in x.groupby("dataset"):
        rows = []
        for pid, t in g.groupby("pk"):
            les = t[t["label"] != "normal"]
            if les.empty:
                continue
            cls = les["label"].mode()[0]
            nor = t[t["label"] == "normal"]
            rows.append({"slices": len(t), "any": bool((t["pred"] == cls).any()),
                         "top3": bool(np.sort(t[f"q_{cls}"].to_numpy())[::-1][:TOPK].mean() >= 0.5),
                         "lesion_slice_sens": float((les["pred"] == les["label"]).mean()),
                         "normal_fp_frac": float((nor["pred"] != "normal").mean()) if len(nor) else np.nan})
        R = pd.DataFrame(rows)
        if R.empty:
            continue
        nfp = R["normal_fp_frac"].dropna()
        r = {"lesion_patients": int(len(R)),
             "entirely_normal_patients": int(sum((t["label"] == "normal").all() for _, t in g.groupby("pk"))),
             "case_detection_any_slice": float(R["any"].mean()), "case_detection_top3_sensitivity_analysis": float(R["top3"].mean()),
             "lesion_slice_sensitivity_median_iqr": [float(R["lesion_slice_sens"].median()), float(R["lesion_slice_sens"].quantile(.25)),
                                                     float(R["lesion_slice_sens"].quantile(.75))],
             "normal_slice_fp_fraction_median_iqr": [float(nfp.median()), float(nfp.quantile(.25)), float(nfp.quantile(.75))] if len(nfp) else None,
             "patients_with_ge1_false_positive_slice": float((nfp > 0).mean()) if len(nfp) else None,
             "slices_per_patient_median_iqr": [float(R["slices"].median()), float(R["slices"].quantile(.25)), float(R["slices"].quantile(.75))],
             "case_detection_any_by_slice_count": {}}
        R["bin"] = pd.cut(R["slices"], SLICE_COUNT_BINS, labels=SLICE_COUNT_NAMES)
        for b, gg in R.groupby("bin", observed=True):
            r["case_detection_any_by_slice_count"][str(b)] = {"patients": int(len(gg)), "detected": float(gg["any"].mean())}
        out[ds] = r
    return out


# ------------------------------------------------------------------ calibration
def calibrated(df, m):
    p = df[pcols(m)].to_numpy(float).clip(1e-12, 1)
    t = np.where(df["dataset"].isin(MRI_SOURCES), T_MRI, T_CT)[:, None]
    q = p ** (1 / t)
    return q / q.sum(1, keepdims=True)


def calib_metrics(y, p):
    conf, pr = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, N_BINS + 1)

    def ece(c, hit):
        b = np.clip(np.digitize(c, edges) - 1, 0, N_BINS - 1)
        return float(sum(abs(hit[b == k].mean() - c[b == k].mean()) * (b == k).mean() for k in range(N_BINS) if (b == k).any()))
    r = {"ece_top": ece(conf, (pr == y).astype(float)),
         "nll": float(-np.log(p[np.arange(len(y)), y].clip(1e-12)).mean()),
         "brier": float(((p - np.eye(3)[y]) ** 2).sum(1).mean())}
    for c in (1, 2):
        if (y == c).any():
            r[f"ece_{LABELS[c]}"] = ece(p[:, c], (y == c).astype(float))
    return r


def reliability_plot(y, before, after, title, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    edges = np.linspace(0, 1, N_BINS + 1)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], color="0.6", lw=1)
    for p, lab in ((before, "raw"), (after, "calibrated (locked T)")):
        conf, hit = p.max(1), (p.argmax(1) == y).astype(float)
        b = np.clip(np.digitize(conf, edges) - 1, 0, N_BINS - 1)
        xs = [conf[b == k].mean() for k in range(N_BINS) if (b == k).sum() >= 10]
        ys = [hit[b == k].mean() for k in range(N_BINS) if (b == k).sum() >= 10]
        ax.plot(xs, ys, marker="o", ms=3, lw=1.5, label=lab)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ descriptive analyses
def cpaisd_core_only(df, m, w):
    x = df[df["dataset"] == "CPAISD_v2"].copy()
    vals = []
    for sk in x["source_key"]:
        p = Path(str(sk))
        cands = [p, Path(str(sk).replace("/kaggle/input/datasets/atik68/", "/kaggle/input/")),
                 Path(str(sk).replace("/kaggle/input/", "/kaggle/input/datasets/atik68/", 1))]
        mp = next((c.parent / "mask.npz" for c in cands if (c.parent / "mask.npz").exists()), None)
        if mp is None:
            vals.append(None)
            continue
        with np.load(mp) as d:
            a = np.asarray(d["mask"] if "mask" in d.files else d[d.files[0]])
        if a.ndim == 3 and a.shape[-1] <= 4 and set(np.unique(a)) <= {0, 1}:   # one-hot channels (bg, core, penumbra)
            a = a.argmax(-1)
        vals.append({int(v) for v in np.unique(a) if v > 0})
    x["mask_values"] = vals
    found = x["mask_values"].notna().mean()
    if found < 0.9:
        return {"performed": False, "reason": f"CPAISD mask.npz found for only {found:.1%} of rows"}
    isch = x["label"] == "ischemic"
    core = isch & x["mask_values"].map(lambda s: bool(s) and 1 in s)
    pen_only = isch & x["mask_values"].map(lambda s: bool(s) and 1 not in s)
    keep = (x["label"] == "normal") | core
    sub = x[keep]
    y = sub["label"].map(L2I).to_numpy()
    p = sub[pcols(m)].to_numpy(float)
    res = {"mask_files_found_fraction": float(found), "core_slices": int(core.sum()), "penumbra_only_slices": int(pen_only.sum()),
           "core_only_slice_metrics": slice_metrics(y, decide(p, w), p)}
    pp = x.loc[pen_only, pcols(m)].to_numpy(float)
    res["penumbra_only_detection_rate"] = float((decide(pp, w) == 1).mean()) if len(pp) else None
    return res


def aisd_failure(df, m, w, masks):
    import statsmodels.api as sm
    x = df[df["dataset"] == "AISD"].copy()
    x["pid"] = x["source_key"].astype(str).str.split("/").str[2]
    x["sl"] = [int(re.sub(r"\D", "", Path(str(s)).stem) or -1) for s in x["source_key"]]
    lo, hi = x.groupby("pid")["sl"].transform("min"), x.groupby("pid")["sl"].transform("max")
    x["relpos"] = (x["sl"] - lo) / (hi - lo).replace(0, np.nan)
    x["pred"] = np.array(LABELS)[decide(x[pcols(m)].to_numpy(float), w)]
    x["band"] = pd.cut(x["relpos"], [-0.01, 1 / 3, 2 / 3, 1.01], labels=["lower", "middle", "upper"])
    les_px, dom = [], []
    for pid, sl in zip(x["pid"], x["sl"]):
        f = masks.get((pid, sl))
        if f is None:
            les_px.append(np.nan)
            dom.append(np.nan)
            continue
        a = np.asarray(Image.open(f))
        a = a[..., 0] if a.ndim == 3 else a
        n2, n3 = int((a == 2).sum()), int((a == 3).sum())
        les_px.append(n2 + n3)
        dom.append(2 if n2 >= n3 else 3)
    x["lesion_px"], x["dominant_value"] = les_px, dom
    isch, norm = x[x["label"] == "ischemic"].copy(), x[x["label"] == "normal"]
    isch["hit"] = isch["pred"] == "ischemic"
    r = {"ischemic_recall_by_third": {str(b): [float(g["hit"].mean()), int(len(g))] for b, g in isch.groupby("band", observed=True)},
         "normal_fp_by_third": {str(b): [float((g["pred"] != "normal").mean()), int(len(g))] for b, g in norm.groupby("band", observed=True)},
         "ischemic_recall_by_lesion_px": {}, "ischemic_recall_by_value": {}}
    isch["size_bin"] = pd.cut(isch["lesion_px"], AISD_BINS, labels=AISD_BIN_NAMES, right=False)
    for b, g in isch.groupby("size_bin", observed=True):
        r["ischemic_recall_by_lesion_px"][str(b)] = [float(g["hit"].mean()), int(len(g))]
    for v, g in isch.dropna(subset=["dominant_value"]).groupby("dominant_value"):
        r["ischemic_recall_by_value"][str(int(v))] = [float(g["hit"].mean()), int(len(g))]
    X = isch.dropna(subset=["relpos"]).copy()
    X["miss"] = (~X["hit"]).astype(int)
    X["relpos_edge"] = np.abs(X["relpos"] - 0.5) * 2
    if X["miss"].sum() >= 5:
        try:
            fit = sm.GLM(X["miss"], sm.add_constant(X[["relpos_edge"]]), family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(X["pid"])[0]})
            ci = np.exp(fit.conf_int().loc["relpos_edge"])
            r["edge_miss_or"] = [float(np.exp(fit.params["relpos_edge"])), float(ci[0]), float(ci[1]), float(fit.pvalues["relpos_edge"])]
        except Exception as e:  # noqa: BLE001
            r["edge_miss_or"] = f"failed: {e!r}"[:150]
    return r


def isles_size(df, m, w):
    import nibabel as nib
    root = next((r for r in (ISLES_RAW, INPUT_ROOT / "neuroscan-isles2022") if r.exists()), None)
    if root is None:
        return "ISLES dataset not mounted"
    deriv = find_dir(root, "derivatives")
    if deriv is None:
        return "derivatives folder not found"
    x = df[(df["dataset"] == "ISLES2022_dwi_adc") & (df["label"] == "ischemic")].copy()
    x["hit"] = decide(x[pcols(m)].to_numpy(float), w) == 1
    cache, area = {}, []
    for sk in x["source_key"]:
        mm = ISLES_KEY_RE.match(str(sk))
        if not mm:
            area.append(np.nan)
            continue
        pat, s = (mm.group(1), mm.group(2)), int(mm.group(3))
        if pat not in cache:
            img = None
            for ext in (".nii.gz", ".nii"):
                f = deriv / pat[0] / pat[1] / f"{pat[0]}_{pat[1]}_msk{ext}"
                if f.exists():
                    img = nib.load(str(f)).get_fdata(dtype=np.float32)
                    break
            cache[pat] = img
        a = cache[pat]
        if a is None or not 0 <= s < a.shape[2]:
            area.append(np.nan)
            continue
        sl = ((a[:, :, s] > 0) * 255).astype(np.uint8)
        area.append(int((np.asarray(Image.fromarray(sl).resize((224, 224), Image.NEAREST)) > 0).sum()))
    x["area"] = area
    x["bin"] = pd.cut(x["area"], ISLES_BINS, labels=ISLES_BIN_NAMES)
    return {str(b): [float(g["hit"].mean()), int(len(g))] for b, g in x.groupby("bin", observed=True)}


def aisd_masks():
    import gdown
    TMP.mkdir(parents=True, exist_ok=True)
    z = TMP / "mask.zip"
    gdown.download(f"https://drive.google.com/uc?id={AISD_MASK_ZIP_ID}", str(z), quiet=True)
    out = {}
    with zipfile.ZipFile(z) as zf:
        zf.extractall(TMP / "m")
    for f in (TMP / "m").rglob("*.png"):
        if f.stem.isdigit():
            out[(f.parent.name, int(f.stem))] = f
    return out




# =====================================================================================================================
# POST-HOC ANALYSES (2026-10-07, post-hoc round 1). Specified after the one-time final evaluation; descriptive only.
# They use the same saved probabilities and metric code as neuroscan-q-final-evaluation and change nothing about the
# final model, its weights or its decision rule.
#  B1  pooled results excluding Kaggle CT (no patient identifiers)
#  B2  95% CIs for patient-level case detection; an examination-level rule that needs no reference labels
#  B3  seed 7 vs seed 123: paired patient-cluster bootstrap differences under (i) each model's own locked weights,
#      (ii) a common rule = argmax, (iii) a common rule = the final model's weights
#  B4  the deployed alert-tier thresholds applied to the final model's probabilities (deployment uses plain argmax)
# =====================================================================================================================
from math import sqrt

KEYS_B3 = ["accuracy", "macro_f1_present_classes", "ischemic_recall", "ischemic_precision", "hemorrhagic_recall",
           "hemorrhagic_precision", "normal_recall", "ischemic_auroc", "hemorrhagic_auroc", "ischemic_pr_auc"]


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def paired_seed(df, w123, w7, label, seed):
    """Paired patient-cluster bootstrap of seed7 - seed123 (same resampled patients for both models)."""
    y = df["label"].map(L2I).to_numpy()
    pa, pb = df[pcols("round30as7")].to_numpy(float), df[pcols(PRIMARY)].to_numpy(float)
    da, db = decide(pa, w7), decide(pb, w123)
    d0 = df.reset_index(drop=True)
    groups = [[g.index.to_numpy()[ix] for ix in g.groupby("pk").indices.values()] for _, g in d0.groupby("dataset")]
    ma, mb = slice_metrics(y, da, pa), slice_metrics(y, db, pb)
    rng = np.random.default_rng(seed)
    diffs = {k: [] for k in KEYS_B3}
    for _ in range(N_BOOT):
        idx = np.concatenate([np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))]) for cl in groups])
        a, b = slice_metrics(y[idx], da[idx], pa[idx]), slice_metrics(y[idx], db[idx], pb[idx])
        for k in KEYS_B3:
            if k in a and k in b and a[k] == a[k] and b[k] == b[k]:
                diffs[k].append(a[k] - b[k])
    out = {"rule": label}
    for k in KEYS_B3:
        if k in ma and k in mb and len(diffs[k]) > 50:
            v = np.array(diffs[k])
            out[k] = {"seed123": mb[k], "seed7": ma[k], "diff_7_minus_123": ma[k] - mb[k],
                      "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                      "p_two_sided": float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean())))}
    return out


def exam_level(df, m, w):
    """Examination-level rules executable without reference labels, evaluated on lesion-positive patients.
    q = p_raw * (1, w_i, w_h) / sum(p_raw * (1, w_i, w_h)) per slice (TTA-averaged raw probabilities).
    any_lesion_flag: >= 1 slice predicted ischemic or hemorrhagic (no class requirement).
    exam_rule_top3: s_c = mean of the min(3, n) highest q_c over the patient's slices, c in {ischemic, hemorrhagic};
                    predicted exam class = argmax_c s_c if max_c s_c >= 0.5, else normal."""
    x = df[~df["pk"].str.contains("_row:")].copy()
    p = x[pcols(m)].to_numpy(float) * np.array([1.0, w[0], w[1]])
    q = p / p.sum(1, keepdims=True)
    x["qi"], x["qh"] = q[:, 1], q[:, 2]
    x["pred"] = np.array(LABELS)[q.argmax(1)]
    res = {}
    for ds, g in x.groupby("dataset"):
        rows = []
        for _, t in g.groupby("pk"):
            les = t[t["label"] != "normal"]
            if les.empty:
                continue
            cls = les["label"].mode()[0]
            k = min(3, len(t))
            si, sh = np.sort(t["qi"].to_numpy())[::-1][:k].mean(), np.sort(t["qh"].to_numpy())[::-1][:k].mean()
            exam = ("ischemic" if si >= sh else "hemorrhagic") if max(si, sh) >= 0.5 else "normal"
            rows.append({"cls": cls, "n": len(t), "any_flag": bool((t["pred"] != "normal").any()),
                         "strict_lesion_slice": bool((les["pred"] == les["label"]).any()),
                         "any_correct": bool((t["pred"] == cls).any()), "exam_top3": exam})
        R = pd.DataFrame(rows)
        n = len(R)
        k1, k2, k3 = int(R["any_correct"].sum()), int(R["any_flag"].sum()), int((R["exam_top3"] == R["cls"]).sum())
        k4 = int(R["strict_lesion_slice"].sum())
        res[ds] = {"lesion_patients": n,
                   "case_detection_any_slice": [k1, n, k1 / n, wilson(k1, n)],
                   "any_lesion_flag_no_class": [k2, n, k2 / n, wilson(k2, n)],
                   "exam_rule_top3_correct_class": [k3, n, k3 / n, wilson(k3, n)],
                   "strict_correct_lesion_slice": [k4, n, k4 / n, wilson(k4, n)],
                   "exam_rule_top3_called_normal": int((R["exam_top3"] == "normal").sum()),
                   "exams_with_fewer_than_3_slices": int((R["n"] < 3).sum())}
    tot = {k: sum(v[k][0] for v in res.values()) for k in ("case_detection_any_slice", "any_lesion_flag_no_class",
                                                            "exam_rule_top3_correct_class", "strict_correct_lesion_slice")}
    N = sum(v["lesion_patients"] for v in res.values())
    res["ALL_identifiable"] = {k: [c, N, c / N, wilson(c, N)] for k, c in tot.items()}
    return res


def tier(cls, conf):
    if cls == "hemorrhagic":
        return "high" if conf >= 0.55 else "moderate"
    if cls == "ischemic":
        return "high" if conf >= 0.90 else ("moderate" if conf >= 0.70 else "uncertain")
    return "normal" if conf >= 0.80 else "uncertain"


def alert_tiers(df, m):
    """Deployed tier rule (plain argmax of TTA-averaged raw probabilities; confidence = winning probability)."""
    p = df[pcols(m)].to_numpy(float)
    p = p / p.sum(1, keepdims=True)
    pred = np.array(LABELS)[p.argmax(1)]
    conf = p.max(1)
    t = np.array([tier(c, v) for c, v in zip(pred, conf)])
    lab = df["label"].to_numpy()
    out = {"n": int(len(df)), "tier_by_true_class": {}}
    for c in LABELS:
        msk = lab == c
        if msk.any():
            out["tier_by_true_class"][c] = {f"{pc}|{tt}": int(((pred == pc) & (t == tt) & msk).sum())
                                            for pc in LABELS for tt in ("high", "moderate", "uncertain", "normal")
                                            if ((pred == pc) & (t == tt) & msk).any()}
    alert = (pred != "normal") & np.isin(t, ["high", "moderate"])
    high = (pred != "normal") & (t == "high")
    for c in ("ischemic", "hemorrhagic"):
        msk = lab == c
        if msk.any():
            out[f"{c}_alert_sensitivity_any_alert"] = float((alert & msk & (pred == c)).sum() / msk.sum())
            out[f"{c}_alert_sensitivity_high"] = float((high & msk & (pred == c)).sum() / msk.sum())
            out[f"{c}_predicted_normal"] = int((msk & (pred == "normal")).sum())
            out[f"{c}_predicted_normal_confident(normal tier)"] = int((msk & (pred == "normal") & (t == "normal")).sum())
    nm = lab == "normal"
    out["false_alert_burden_normal_slices_any_alert"] = float((alert & nm).sum() / nm.sum())
    out["false_alert_burden_normal_slices_high"] = float((high & nm).sum() / nm.sum())
    out["alerts_total"] = int(alert.sum())
    out["alert_ppv_any_alert"] = float(((alert) & (pred == lab)).sum() / max(alert.sum(), 1))
    out["uncertain_tier_coverage"] = float((t == "uncertain").mean())
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rmap = pd.read_csv(need(PHASE2 / "rsna_patient_map.csv"))
    RSNA_MAP.update(dict(zip(rmap["stem"].astype(str), rmap["PatientID"])))
    val = load("predictions_val.csv")
    yv = val["label"].map(L2I).to_numpy()
    W = {m: original_rule(yv, val[pcols(m)].to_numpy(float))[0] for m in MODELS}
    assert all(abs(a - b) < 1e-3 for a, b in zip(W[PRIMARY], LOCKED_W_PRIMARY))
    log(f"locked weights reproduced: {W}")
    test, soop = load("predictions_test.csv"), load("predictions_soop_external.csv")
    A, B = test[test["dataset"] != "CPAISD_v2"].reset_index(drop=True), test[test["dataset"] == "CPAISD_v2"].reset_index(drop=True)
    R = {"note": "POST-HOC analyses specified after the one-time final evaluation (post-hoc round 1); descriptive only",
         "weights": W, "sha256": REPORT.get("sha256")}

    # B1 -- excluding Kaggle CT
    R["B1_excluding_kaggle"] = {}
    for name, df in (("AB_pooled_excl_Kaggle", test[test["dataset"] != "Kaggle_CT"]), ("A_excl_Kaggle", A[A["dataset"] != "Kaggle_CT"]),
                     ("AB_pooled_incl_Kaggle", test)):
        df = df.reset_index(drop=True)
        r = with_ci(df, PRIMARY, W[PRIMARY], seed=11)
        r["identifiable_patients"] = int(df.loc[~df["pk"].str.contains("_row:"), "pk"].nunique())
        r["kaggle_images_unknown_patients"] = int((df["dataset"] == "Kaggle_CT").sum())
        R["B1_excluding_kaggle"][name] = r
        log(f"B1 {name}: n={r['n']} identifiable patients={r['identifiable_patients']} Kaggle images={r['kaggle_images_unknown_patients']} "
            f"acc={r['accuracy']:.4f} {r['ci95'].get('accuracy')} mF1={r['macro_f1_present_classes']:.4f} "
            f"isch AUROC={r.get('ischemic_auroc', float('nan')):.4f} hem AUROC={r.get('hemorrhagic_auroc', float('nan')):.4f}")

    # B2 -- patient-level CIs and an examination-level rule without reference labels
    R["B2_exam_level"] = {c: exam_level(df, PRIMARY, W[PRIMARY]) for c, df in (("A", A), ("B", B), ("C", soop))}
    allc = {}
    for c, v in R["B2_exam_level"].items():
        for ds, r in v.items():
            if ds != "ALL_identifiable":
                for k in ("case_detection_any_slice", "any_lesion_flag_no_class", "exam_rule_top3_correct_class",
                          "strict_correct_lesion_slice"):
                    a = allc.setdefault(k, [0, 0])
                    a[0] += r[k][0]
                    a[1] += r[k][1]
    R["B2_all_cohorts"] = {k: [a[0], a[1], a[0] / a[1], wilson(a[0], a[1])] for k, a in allc.items()}
    log(f"B2 all cohorts (identifiable lesion-positive patients): {R['B2_all_cohorts']}")
    log(f"B2 per cohort/source: {json.dumps(jsonable(R['B2_exam_level']))}")

    # B3 -- seed comparison
    R["B3_seed_comparison"] = {}
    i = 0
    for cname, df in (("AB_pooled", test), ("A", A), ("B_cpaisd", B), ("C_soop", soop)):
        for rule, w7, w123 in (("own_locked_weights", W["round30as7"], W[PRIMARY]), ("common_argmax", (1.0, 1.0), (1.0, 1.0)),
                               ("common_final_model_weights", W[PRIMARY], W[PRIMARY])):
            i += 1
            R["B3_seed_comparison"][f"{cname}|{rule}"] = r = paired_seed(df.reset_index(drop=True), w123, w7, rule, seed=100 + i)
            log(f"B3 {cname} {rule}: " + "; ".join(f"{k} {v['seed123']:.4f}->{v['seed7']:.4f} d={v['diff_7_minus_123']:+.4f} "
                                                   f"[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]"
                                                   for k, v in r.items() if isinstance(v, dict)))

    # B4 -- deployed alert tiers applied to the final model
    R["B4_alert_tiers_final_model_plain_argmax"] = {c: alert_tiers(df.reset_index(drop=True), PRIMARY)
                                                    for c, df in (("AB_pooled", test), ("A", A), ("B_cpaisd", B), ("C_soop", soop))}
    for c, r in R["B4_alert_tiers_final_model_plain_argmax"].items():
        log(f"B4 {c}: {json.dumps(jsonable({k: v for k, v in r.items() if k != 'tier_by_true_class'}))}")
        log(f"B4 {c} tiers by true class: {json.dumps(r['tier_by_true_class'])}")

    with open(OUT / "posthoc_round1.json", "w") as f:
        json.dump(jsonable(R), f, indent=2)
    log("Done.")


if __name__ == "__main__":
    main()
