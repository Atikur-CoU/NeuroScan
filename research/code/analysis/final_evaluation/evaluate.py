"""
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


# ------------------------------------------------------------------ main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rmap = pd.read_csv(need(PHASE2 / "rsna_patient_map.csv"))
    RSNA_MAP.update(dict(zip(rmap["stem"].astype(str), rmap["PatientID"])))
    val2_patients = set(pd.read_csv(need(PHASE2 / "phase2_val2_patients.csv"))["patient_key"])

    # locked weights, recomputed on each model's own Phase 1 validation predictions
    W = {}
    val = load("predictions_val.csv")
    yv = val["label"].map(L2I).to_numpy()
    for m in MODELS:
        W[m], f1 = original_rule(yv, val[pcols(m)].to_numpy(float))
        log(f"locked original-rule weights {m}: {tuple(round(v, 3) for v in W[m])} (Phase 1 val macro-F1 {f1:.4f})")
    assert all(abs(a - b) < 1e-3 for a, b in zip(W[PRIMARY], LOCKED_W_PRIMARY)), f"primary weights {W[PRIMARY]} != locked"
    REPORT["weights"] = W

    if DRY_RUN:
        test, soop = val[val["dataset"] != "SOOP_dwi_adc"].reset_index(drop=True), val[val["dataset"] == "SOOP_dwi_adc"].reset_index(drop=True)
        log("DRY RUN: Phase 1 VALIDATION files scored in place of the sealed cohorts (code test only)")
    else:
        test, soop = load("predictions_test.csv"), load("predictions_soop_external.csv")
        log(f"SHA-256: {REPORT['sha256']}")
        n_cp = int((test["dataset"] == "CPAISD_v2").sum())
        checks = {"test_rows": (len(test), EXPECT["test"]), "cpaisd_rows": (n_cp, EXPECT["cpaisd_test"]),
                  "cpaisd_patients": (int(test.loc[test["dataset"] == "CPAISD_v2", "pk"].nunique()), EXPECT["cpaisd_patients"]),
                  "soop_rows": (len(soop), EXPECT["soop"]), "soop_patients": (int(soop["pk"].nunique()), EXPECT["soop_patients"])}
        log(f"cohort checks (observed, expected): {checks}")
        assert all(a == b for a, b in checks.values()), "cohort counts differ from the pre-registered plan"
        overlap = (set(test["pk"]) | set(soop["pk"])) & val2_patients
        assert not overlap, f"sealed-cohort patients found in val2: {list(overlap)[:5]}"
        REPORT["cohort_checks"] = checks
    cohorts = {"A_internal_test": test[test["dataset"] != "CPAISD_v2"].reset_index(drop=True),
               "B_cpaisd_test": test[test["dataset"] == "CPAISD_v2"].reset_index(drop=True),
               "C_soop_holdout": soop, "AB_pooled": test}

    REPORT["slice_level"] = {}
    for m in MODELS:
        R = {}
        sets = {}
        for c, df in cohorts.items():
            sets[c] = df
            if c in ("A_internal_test", "AB_pooled"):
                for ds, g in df.groupby("dataset"):
                    sets[f"{c}|{ds}"] = g.reset_index(drop=True)
                sets[f"{c}|CT"] = df[~df["dataset"].isin(MRI_SOURCES)].reset_index(drop=True)
                sets[f"{c}|MRI"] = df[df["dataset"].isin(MRI_SOURCES)].reset_index(drop=True)
            if c == "AB_pooled":
                sets[f"{c}|5_sources_excl_AISD_CPAISD (descriptive)"] = df[~df["dataset"].isin(["AISD", "CPAISD_v2"])].reset_index(drop=True)
        for i, (name, df) in enumerate(sets.items()):
            if len(df) == 0 or name.startswith("AB_pooled|") and "5_sources" not in name and name.split("|")[1] not in ("CT", "MRI"):
                continue
            R[name] = with_ci(df, m, W[m], seed=i)
            r = R[name]
            log(f"[{m}] {name}: n={r['n']} acc={r['accuracy']:.4f} bal={r['balanced_accuracy']:.4f} "
                f"mF1={r['macro_f1_present_classes']:.4f} isch R/P={r.get('ischemic_recall', float('nan')):.4f}/"
                f"{r.get('ischemic_precision', float('nan')):.4f} hem R/P={r.get('hemorrhagic_recall', float('nan')):.4f}/"
                f"{r.get('hemorrhagic_precision', float('nan')):.4f} AUROC isch={r.get('ischemic_auroc', float('nan')):.4f} "
                f"hem={r.get('hemorrhagic_auroc', float('nan')):.4f}")
        REPORT["slice_level"][m] = R

    REPORT["patient_level"] = {m: {c: patient_level(df, m, W[m]) for c, df in cohorts.items() if c != "AB_pooled"} for m in MODELS}
    REPORT["patient_level_not_estimable"] = ("patient-level specificity, PPV, NPV, balanced accuracy and ROC: the cohorts contain "
                                             "essentially no entirely normal patients; within-patient normal slices are not healthy controls")
    for m in MODELS:
        for c, v in REPORT["patient_level"][m].items():
            for ds, r in v.items():
                log(f"[{m}] PATIENT {c} {ds}: lesion patients {r['lesion_patients']}, any-slice {r['case_detection_any_slice']:.3f}, "
                    f"top-3 {r['case_detection_top3_sensitivity_analysis']:.3f}, lesion-slice sens median {r['lesion_slice_sensitivity_median_iqr'][0]:.3f}, "
                    f"normal-slice FP median {r['normal_slice_fp_fraction_median_iqr'][0] if r['normal_slice_fp_fraction_median_iqr'] else 'n/a'}")

    REPORT["calibration"] = {}
    for m in MODELS:
        REPORT["calibration"][m] = {}
        for c, df in cohorts.items():
            for mod, g in (("CT", df[~df["dataset"].isin(MRI_SOURCES)]), ("MRI", df[df["dataset"].isin(MRI_SOURCES)]), ("all", df)):
                if len(g) < 50:
                    continue
                y = g["label"].map(L2I).to_numpy()
                raw = g[pcols(m)].to_numpy(float).clip(1e-12, 1)
                raw = raw / raw.sum(1, keepdims=True)
                cal = calibrated(g, m)
                REPORT["calibration"][m][f"{c}|{mod}"] = {"raw": calib_metrics(y, raw), "locked_T": calib_metrics(y, cal)}
                if m == PRIMARY and mod != "all":
                    reliability_plot(y, raw, cal, f"{m} {c} {mod}", OUT / f"reliability_{c}_{mod}.png")
        log(f"[{m}] calibration: " + "; ".join(f"{k} ECE {v['raw']['ece_top']:.4f} -> {v['locked_T']['ece_top']:.4f}"
                                              for k, v in REPORT["calibration"][m].items()))

    desc = {}
    try:
        desc["cpaisd_core_only"] = cpaisd_core_only(cohorts["B_cpaisd_test"], PRIMARY, W[PRIMARY])
    except Exception as e:  # noqa: BLE001
        desc["cpaisd_core_only"] = f"failed: {e!r}"[:200]
    try:
        masks = aisd_masks()
        desc["aisd_failure_analysis"] = aisd_failure(cohorts["A_internal_test"], PRIMARY, W[PRIMARY], masks)
    except Exception as e:  # noqa: BLE001
        desc["aisd_failure_analysis"] = f"failed: {e!r}"[:200]
    try:
        desc["isles_lesion_size_recall"] = isles_size(cohorts["A_internal_test"], PRIMARY, W[PRIMARY])
    except Exception as e:  # noqa: BLE001
        desc["isles_lesion_size_recall"] = f"failed: {e!r}"[:200]
    REPORT["descriptive_primary_model"] = desc
    log(f"DESCRIPTIVE: {json.dumps(jsonable(desc))[:4000]}")

    name = "final_evaluation_DRYRUN_validation.json" if DRY_RUN else "final_evaluation.json"
    with open(OUT / name, "w") as f:
        json.dump(jsonable(REPORT), f, indent=2)
    log(f"Done. Wrote {name}")


if __name__ == "__main__":
    main()
