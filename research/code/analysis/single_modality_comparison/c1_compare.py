"""
NeuroScan -- post-hoc round 1, experiment C1 comparison (2026-10-07). CPU; scores saved probabilities only; no training,
no model choice; the final model (R30A s123) is not changed by any result.

Question: does joint CT + MRI training change performance relative to matched single-modality models trained with the
identical recipe, seed, partition and checkpoint rule (C1 CT-only, C1 MRI-only)?
Pre-specified primary comparison (written before this kernel ran): VALIDATION rows of each modality, threshold-free
discrimination -- CT: ischemic AUROC and hemorrhagic AUROC (one-vs-rest); MRI: ischemic AUROC. Paired patient-cluster
bootstrap of (single-modality - joint), 2,000 replicates, patients resampled within source, Kaggle_CT rows own clusters.
Secondary (validation): accuracy, balanced accuracy, macro-F1, per-class recall/precision under (i) plain argmax and
(ii) each model's own class weights from the locked original rule fitted on its OWN validation rows of that modality
(in-sample for both models: optimistic but symmetric).
Descriptive only (not used for any decision): held-out internal test rows of each modality and the SOOP hold-out,
with the joint model at its locked weights (0.574, 0.574) and each C1 model at its own validation-fitted weights.
Code helpers are copied unchanged from the final-evaluation / post-hoc kernels.
"""
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_ROOT = Path("/kaggle/input")
DS = INPUT_ROOT / "datasets/atik68"
JOINT = ("round30as123", DS / "neuroscan-q-round30as123-preds")
C1 = {"CT": ("c1ctonlys123", DS / "neuroscan-q-c1ct-preds"), "MRI": ("c1mrionlys123", DS / "neuroscan-q-c1mri-preds")}
PHASE2 = DS / "neuroscan-q-phase2-split-data"
LOCKED_W_JOINT = (0.574, 0.574)
MRI_SOURCES = ("ISLES2022_dwi_adc", "SOOP_dwi_adc")
CT_SOURCES = ("AISD", "CPAISD_v2", "RSNA_ICH", "SICH_CT", "Kaggle_CT")
LABELS = ("normal", "ischemic", "hemorrhagic")
L2I = {l: i for i, l in enumerate(LABELS)}
GRID = np.exp(np.linspace(np.log(0.25), np.log(4.0), 41))
N_BOOT = 2000
KEYS = ["ischemic_auroc", "hemorrhagic_auroc", "macro_ovr_auroc", "ischemic_pr_auc", "hemorrhagic_pr_auc", "accuracy",
        "balanced_accuracy", "macro_f1_present_classes", "ischemic_recall", "ischemic_precision", "hemorrhagic_recall",
        "hemorrhagic_precision", "normal_recall"]
PRIMARY = {"CT": ["ischemic_auroc", "hemorrhagic_auroc"], "MRI": ["ischemic_auroc"]}
CPAISD_RE = re.compile(r"/([\d.]{10,})/(\d+)/raw\.dcm$")
MRI_RE = re.compile(r"^(sub-[\w]+)/(ses-\w+)/")
OUT = Path("/kaggle/working/output")
T0 = time.time()
RSNA_MAP = {}
REPORT = {}


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


def load_pair(mod, file_name):
    """Joint and C1 probabilities on the same rows of one modality (inner join; row counts and labels asserted)."""
    jm, jd = JOINT
    cm, cd = C1[mod]
    fj, fc = need(jd / file_name), need(cd / file_name)
    REPORT.setdefault("sha256", {})[f"{jm}/{file_name}"] = sha256(fj)
    REPORT["sha256"][f"{cm}/{file_name}"] = sha256(fc)
    a, b = pd.read_csv(fj), pd.read_csv(fc)
    keep = CT_SOURCES if mod == "CT" else MRI_SOURCES
    a = a[a["dataset"].isin(keep)][["source_key", "dataset", "label", "out_path", "subject"] + pcols(jm)]
    b = b[["source_key", "out_path", "label"] + pcols(cm)].rename(columns={"label": "label_c1"})
    df = a.merge(b, on=["source_key", "out_path"], how="inner", validate="one_to_one")
    info = {"joint_rows_of_modality": len(a), "c1_rows": len(b), "matched": len(df)}
    assert len(df) == len(a) == len(b), f"{mod} {file_name}: row mismatch {info}"
    assert (df["label"] == df["label_c1"]).all(), f"{mod} {file_name}: label mismatch"
    df["pk"] = [patient_key(d, s, o, u) for d, s, o, u in zip(df["dataset"], df["source_key"], df["out_path"], df["subject"])]
    return df.reset_index(drop=True), info


def paired(df, mod, wj, wc, seed):
    """Paired patient-cluster bootstrap of (C1 - joint) on identical resampled patients."""
    jm, cm = JOINT[0], C1[mod][0]
    y = df["label"].map(L2I).to_numpy()
    pj, pc = df[pcols(jm)].to_numpy(float), df[pcols(cm)].to_numpy(float)
    dj, dc = decide(pj, wj), decide(pc, wc)
    mj, mc = slice_metrics(y, dj, pj), slice_metrics(y, dc, pc)
    groups = [[g.index.to_numpy()[ix] for ix in g.groupby("pk").indices.values()] for _, g in df.groupby("dataset")]
    rng = np.random.default_rng(seed)
    diffs = {k: [] for k in KEYS}
    for _ in range(N_BOOT):
        idx = np.concatenate([np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))]) for cl in groups])
        a, b = slice_metrics(y[idx], dj[idx], pj[idx]), slice_metrics(y[idx], dc[idx], pc[idx])
        for k in KEYS:
            if k in a and k in b and a[k] == a[k] and b[k] == b[k]:
                diffs[k].append(b[k] - a[k])
    out = {"n": int(len(df)), "patients_or_clusters": int(df["pk"].nunique()),
           "weights_joint": list(wj), "weights_c1": list(wc),
           "joint_predicts_hemorrhagic_on_rows": int((dj == 2).sum()), "c1_predicts_hemorrhagic_on_rows": int((dc == 2).sum()),
           "confusion_joint": mj["confusion_matrix_rows_true_cols_pred"], "confusion_c1": mc["confusion_matrix_rows_true_cols_pred"]}
    for k in KEYS:
        if k in mj and k in mc and len(diffs[k]) > 50:
            v = np.array(diffs[k])
            out[k] = {"joint": mj[k], "c1": mc[k], "diff_c1_minus_joint": mc[k] - mj[k],
                      "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                      "p_two_sided": float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean())))}
    return out


def per_source(df, mod, wj, wc):
    jm, cm = JOINT[0], C1[mod][0]
    out = {}
    for ds, g in df.groupby("dataset"):
        y = g["label"].map(L2I).to_numpy()
        pj, pc = g[pcols(jm)].to_numpy(float), g[pcols(cm)].to_numpy(float)
        a, b = slice_metrics(y, decide(pj, wj), pj), slice_metrics(y, decide(pc, wc), pc)
        out[ds] = {"n": int(len(g)), "patients_or_clusters": int(g["pk"].nunique()),
                   **{k: {"joint": a[k], "c1": b[k], "diff": b[k] - a[k]} for k in KEYS if k in a and k in b}}
    return out


def show(tag, r, keys):
    log(f"{tag}: n={r['n']} clusters={r['patients_or_clusters']} " + "; ".join(
        f"{k} {r[k]['joint']:.4f}->{r[k]['c1']:.4f} d={r[k]['diff_c1_minus_joint']:+.4f} "
        f"[{r[k]['ci95'][0]:+.4f},{r[k]['ci95'][1]:+.4f}] p={r[k]['p_two_sided']:.3f}" for k in keys if k in r))


def show_sources(tag, ps):
    for ds, v in ps.items():
        log(f"{tag} per-source {ds}: n={v['n']} " + "; ".join(
            f"{k} {v[k]['joint']:.4f}->{v[k]['c1']:.4f}" for k in ("accuracy", "ischemic_auroc", "hemorrhagic_auroc",
                                                                  "ischemic_recall", "hemorrhagic_recall") if k in v))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rmap = pd.read_csv(need(PHASE2 / "rsna_patient_map.csv"))
    RSNA_MAP.update(dict(zip(rmap["stem"].astype(str), rmap["PatientID"])))
    R = {"note": "C1 matched single-modality baselines vs joint final model; primary = validation AUROC; test/SOOP descriptive",
         "primary_measures": PRIMARY}
    full = pd.read_csv(need(JOINT[1] / "predictions_val.csv"))
    wfull = original_rule(full["label"].map(L2I).to_numpy(), full[pcols(JOINT[0])].to_numpy(float))[0]
    assert all(abs(a - b) < 1e-3 for a, b in zip(wfull, LOCKED_W_JOINT)), wfull
    log(f"joint locked weights reproduced: {wfull}")
    seed = 500
    sec = ["macro_ovr_auroc", "accuracy", "balanced_accuracy", "macro_f1_present_classes", "ischemic_recall",
           "ischemic_precision", "hemorrhagic_recall", "hemorrhagic_precision"]
    for mod in ("CT", "MRI"):
        cm = C1[mod][0]
        val, info = load_pair(mod, "predictions_val.csv")
        yv = val["label"].map(L2I).to_numpy()
        w_c1, f1c = original_rule(yv, val[pcols(cm)].to_numpy(float))
        w_jm, f1j = original_rule(yv, val[pcols(JOINT[0])].to_numpy(float))
        R[mod] = {"val_rows": info, "c1_own_weights_on_modality_val": [w_c1, f1c],
                  "joint_weights_refit_on_modality_val": [w_jm, f1j]}
        log(f"{mod} val rows {info}; C1 own weights {w_c1} (mF1 {f1c:.4f}); joint refit on {mod} val {w_jm} (mF1 {f1j:.4f})")
        for rule, wj, wc in (("plain_argmax", (1.0, 1.0), (1.0, 1.0)), ("own_weights_fit_on_modality_val", w_jm, w_c1)):
            seed += 1
            r = paired(val, mod, wj, wc, seed)
            R[mod][f"VAL|{rule}"] = r
            show(f"{mod} VAL {rule}", r, (PRIMARY[mod] if rule == "plain_argmax" else []) + sec)
            log(f"{mod} VAL {rule}: hemorrhagic predictions joint={r['joint_predicts_hemorrhagic_on_rows']} "
                f"c1={r['c1_predicts_hemorrhagic_on_rows']}")
        R[mod]["VAL|per_source_plain_argmax"] = ps = per_source(val, mod, (1.0, 1.0), (1.0, 1.0))
        show_sources(f"{mod} VAL", ps)
        files = ["predictions_test.csv"] + (["predictions_soop_external.csv"] if mod == "MRI" else [])
        for fn in files:
            df, info = load_pair(mod, fn)
            seed += 1
            r = paired(df, mod, LOCKED_W_JOINT, w_c1, seed)
            r["rows"] = info
            r["per_source"] = per_source(df, mod, LOCKED_W_JOINT, w_c1)
            tag = "TEST" if "test" in fn else "SOOP_holdout"
            R[mod][f"{tag}|descriptive_joint_locked_vs_c1_own_weights"] = r
            show(f"{mod} {tag} (descriptive)", r, PRIMARY[mod] + sec)
            show_sources(f"{mod} {tag}", r["per_source"])
    R["sha256"] = REPORT.get("sha256")
    with open(OUT / "c1_comparison.json", "w") as f:
        json.dump(jsonable(R), f, indent=2)
    log("Done.")


if __name__ == "__main__":
    main()
