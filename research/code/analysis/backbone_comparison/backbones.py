"""
NeuroScan -- backbone comparison (post hoc, 2026-10-08), in the format of the earlier companion paper's Table 7. CPU; saved
probabilities only. Every model was trained with the final model's script and recipe (Phase 1 patient-level partition v3,
Round 24 recipe, 45 epochs, patience 15, best-validation-accuracy checkpoint, 4-view TTA), only the backbone (and seed)
differing. Decision rule for every row: plain argmax. Validation rows are the primary basis for comparing backbones; the
held-out internal test (A + B) is descriptive -- the final model was chosen before this comparison and is not changed by it.
Paired patient-cluster bootstrap (2,000 replicates) of each model minus the final model (ConvNeXt-Tiny seed 123).
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_ROOT = Path("/kaggle/input")
DS = INPUT_ROOT / "datasets"
PHASE2 = DS / "atik68/neuroscan-q-phase2-split-data"
LABELS = ("normal", "ischemic", "hemorrhagic")
L2I = {l: i for i, l in enumerate(LABELS)}
N_BOOT = 2000
CPAISD_RE = re.compile(r"/([\d.]{10,})/(\d+)/raw\.dcm$")
MRI_RE = re.compile(r"^(sub-[\w]+)/(ses-\w+)/")
OUT = Path("/kaggle/working/output")
T0 = time.time()
RSNA_MAP = {}
REF = "convnext_s123"
MODELS = [
 {
  "name": "convnext_s123",
  "key": "round30as123",
  "dataset": "atik68/neuroscan-q-round30as123-preds",
  "arch": "convnext_tiny",
  "display": "ConvNeXt-Tiny (final)",
  "seed": 123,
  "best_epoch": 9,
  "stopped_after": 24
 },
 {
  "name": "convnext_s7",
  "key": "round30as7",
  "dataset": "atik68/neuroscan-q-round30as7-preds",
  "arch": "convnext_tiny",
  "display": "ConvNeXt-Tiny",
  "seed": 7,
  "best_epoch": 41,
  "stopped_after": 45
 },
 {
  "name": "resnet18_s123",
  "key": "round24ctl",
  "dataset": "atik68/neuroscan-q-round24ctl-preds",
  "arch": "resnet18",
  "display": "ResNet18",
  "seed": 123,
  "best_epoch": 14,
  "stopped_after": 29
 },
 {
  "name": "resnet18_s7",
  "key": "round24s7",
  "dataset": "atik68/neuroscan-q-round24s7-preds",
  "arch": "resnet18",
  "display": "ResNet18",
  "seed": 7,
  "best_epoch": 25,
  "stopped_after": 40
 },
 {
  "name": "effb0_s123",
  "key": "effb0",
  "dataset": "atik68/neuroscan-q-effb0-office-preds",
  "arch": "efficientnet_b0",
  "display": "EfficientNet-B0",
  "seed": 123,
  "best_epoch": 8,
  "stopped_after": 23
 },
 {
  "name": "effb0_s7",
  "key": "effb0s7",
  "dataset": "atik68/neuroscan-q-effb0s7-office-preds",
  "arch": "efficientnet_b0",
  "display": "EfficientNet-B0",
  "seed": 7,
  "best_epoch": 21,
  "stopped_after": 36
 },
 {
  "name": "mobilenetv2_s123",
  "key": "bbmobilenetv2s123",
  "dataset": "atik68/neuroscan-q-bb-mobilenetv2-preds",
  "arch": "mobilenet_v2",
  "display": "MobileNetV2",
  "seed": 123,
  "best_epoch": 16,
  "stopped_after": 31
 },
 {
  "name": "densenet121_s123",
  "key": "bbdensenet121s123",
  "dataset": "atik68/neuroscan-q-bb-densenet121-preds",
  "arch": "densenet121",
  "display": "DenseNet121",
  "seed": 123,
  "best_epoch": 36,
  "stopped_after": 45
 },
 {
  "name": "swint_s123",
  "key": "bbswints123",
  "dataset": "atik68/neuroscan-q-bb-swint-preds",
  "arch": "swin_t",
  "display": "Swin-Tiny",
  "seed": 123,
  "best_epoch": 31,
  "stopped_after": 45
 },
 {
  "name": "resnet101_s123",
  "key": "bbresnet101s123",
  "dataset": "atik68/neuroscan-q-bb-resnet101-preds",
  "arch": "resnet101",
  "display": "ResNet101",
  "seed": 123,
  "best_epoch": 9,
  "stopped_after": 24
 },
 {
  "name": "vgg16_s123",
  "key": "bbvgg16s123",
  "dataset": "atik68/neuroscan-q-bb-vgg16-preds",
  "arch": "vgg16",
  "display": "VGG16",
  "seed": 123,
  "best_epoch": 38,
  "stopped_after": 45
 }
]


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


def load_all(file_name):
    base = None
    for m in MODELS:
        f = need(DS / m["dataset"] / file_name)
        df = pd.read_csv(f)
        cols = [f"p_{l}_{m['key']}" for l in LABELS]
        df = df[["source_key", "dataset", "label", "out_path", "subject"] + cols].rename(
            columns={c: f"p_{l}_{m['name']}" for c, l in zip(cols, LABELS)})
        if base is None:
            base = df
        else:
            n0 = len(base)
            base = base.merge(df.drop(columns=["dataset", "subject"]).rename(columns={"label": "label_chk"}),
                              on=["source_key", "out_path"], how="inner", validate="one_to_one")
            assert len(base) == n0 == len(df), f"{m['name']} {file_name}: rows differ ({n0}, {len(df)}, matched {len(base)})"
            assert (base["label"] == base["label_chk"]).all()
            base = base.drop(columns=["label_chk"])
    base["pk"] = [patient_key(a, b, c, s) for a, b, c, s in zip(base["dataset"], base["source_key"], base["out_path"], base["subject"])]
    return base.reset_index(drop=True)


def pcols(name):
    return [f"p_{l}_{name}" for l in LABELS]


def params_m(arch):
    from torchvision import models as tvm
    m = {"convnext_tiny": tvm.convnext_tiny, "resnet18": tvm.resnet18, "efficientnet_b0": tvm.efficientnet_b0,
         "vgg16": tvm.vgg16, "mobilenet_v2": tvm.mobilenet_v2, "resnet101": tvm.resnet101, "densenet121": tvm.densenet121,
         "swin_t": tvm.swin_t}[arch](weights=None, num_classes=3)
    return round(sum(p.numel() for p in m.parameters()) / 1e6, 1)


def paired(df, ref, other, seed):
    y = df["label"].map(L2I).to_numpy()
    pa, pb = df[pcols(ref)].to_numpy(float), df[pcols(other)].to_numpy(float)
    da, db = pa.argmax(1), pb.argmax(1)
    groups = [[g.index.to_numpy()[ix] for ix in g.groupby("pk").indices.values()] for _, g in df.groupby("dataset")]
    rng = np.random.default_rng(seed)
    keys = ["accuracy", "macro_f1_present_classes", "ischemic_auroc", "hemorrhagic_auroc", "ischemic_recall", "hemorrhagic_recall"]
    diffs = {k: [] for k in keys}
    for _ in range(N_BOOT):
        idx = np.concatenate([np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))]) for cl in groups])
        a, b = slice_metrics(y[idx], da[idx], pa[idx]), slice_metrics(y[idx], db[idx], pb[idx])
        for k in keys:
            if k in a and k in b and a[k] == a[k] and b[k] == b[k]:
                diffs[k].append(b[k] - a[k])
    a, b = slice_metrics(y, da, pa), slice_metrics(y, db, pb)
    return {k: {"diff_other_minus_ref": b[k] - a[k], "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
            for k, v in diffs.items() if len(v) > 50}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rmap = pd.read_csv(need(PHASE2 / "rsna_patient_map.csv"))
    RSNA_MAP.update(dict(zip(rmap["stem"].astype(str), rmap["PatientID"])))
    R = {"note": "Backbone comparison on identical data, partition, recipe and 4-view TTA; decision = plain argmax. "
                 "Validation = primary basis for comparing backbones; held-out test = descriptive (post hoc).", "models": MODELS}
    for m in MODELS:
        m["params_M"] = params_m(m["arch"])
    seed = 900
    for split, fn in (("validation", "predictions_val.csv"), ("test_A_plus_B", "predictions_test.csv")):
        df = load_all(fn)
        y = df["label"].map(L2I).to_numpy()
        R[split] = {"rows": int(len(df)), "patient_clusters": int(df["pk"].nunique()), "per_model": {}, "paired_vs_final": {}}
        log(f"== {split}: {len(df)} rows, {df['pk'].nunique()} clusters")
        for m in MODELS:
            p = df[pcols(m["name"])].to_numpy(float)
            r = slice_metrics(y, p.argmax(1), p)
            R[split]["per_model"][m["name"]] = r
            log(f"{m['display']:28s} seed {m['seed']:>3} params {m['params_M']:6.1f}M | acc {100 * r['accuracy']:.2f} mF1 {100 * r['macro_f1_present_classes']:.2f} | "
                + " | ".join(f"{c[:4]} P/R/F1 {100 * r.get(c + '_precision', np.nan):.1f}/{100 * r.get(c + '_recall', np.nan):.1f}/{100 * r.get(c + '_f1', np.nan):.1f}" for c in LABELS)
                + f" | AUROC isch {r.get('ischemic_auroc', np.nan):.3f} hem {r.get('hemorrhagic_auroc', np.nan):.3f}")
        for m in MODELS:
            if m["name"] != REF:
                seed += 1
                R[split]["paired_vs_final"][m["name"]] = d = paired(df, REF, m["name"], seed)
                log(f"  {m['display']} - final: " + "; ".join(f"{k} {100 * v['diff_other_minus_ref'] if 'auroc' not in k else v['diff_other_minus_ref']:+.3f} "
                                                           f"[{100 * v['ci95'][0] if 'auroc' not in k else v['ci95'][0]:+.3f}, {100 * v['ci95'][1] if 'auroc' not in k else v['ci95'][1]:+.3f}]"
                                                           for k, v in d.items()))
    with open(OUT / "backbone_comparison.json", "w") as f:
        json.dump(jsonable(R), f, indent=2)
    log("Done.")


if __name__ == "__main__":
    main()
