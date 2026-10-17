"""
NeuroScan -- alert tiers under the v2 deployment rule (2026-10-07). CPU; saved final-evaluation probabilities only;
no inference; descriptive. Model R30A s123, unchanged.
v2 server rule: prediction = argmax(p_raw * (1, 0.574, 0.574)) (locked rule); displayed probabilities = temperature-
calibrated p_raw^(1/T), renormalised (T_CT 1.48, T_MRI 0.80); tier (webapp_php/includes/risk_tier.php thresholds)
from the predicted class and its displayed probability. Also reproduces the post-hoc B4 numbers (plain argmax, raw
probabilities) as a check, and counts how often the predicted class is not the largest displayed probability.
Cohorts: A+B pooled internal test (8,930), A (7,406), B CPAISD (1,524), C SOOP hold-out (1,672).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_ROOT = Path("/kaggle/input")
SRC = next(p for p in (INPUT_ROOT / "datasets/atik68/neuroscan-q-round30as123-preds", INPUT_ROOT / "neuroscan-q-round30as123-preds")
           if p.exists())
LABELS = ("normal", "ischemic", "hemorrhagic")
W = np.array([1.0, 0.574, 0.574])
T = {"CT": 1.48, "MRI": 0.80}
MRI = ("ISLES2022_dwi_adc", "SOOP_dwi_adc")
OUT = Path("/kaggle/working/output")
OUT.mkdir(parents=True, exist_ok=True)
PC = [f"p_{l}_round30as123" for l in LABELS]


def tier(cls, conf):
    if cls == "hemorrhagic":
        return "high" if conf >= 0.55 else "moderate"
    if cls == "ischemic":
        return "high" if conf >= 0.90 else ("moderate" if conf >= 0.70 else "uncertain")
    return "normal" if conf >= 0.80 else "uncertain"


def summarise(lab, pred, conf):
    t = np.array([tier(c, v) for c, v in zip(pred, conf)])
    out = {"n": int(len(lab)), "tier_by_true_class": {}}
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
            out[f"{c}_alert_any"] = float((alert & msk & (pred == c)).sum() / msk.sum())
            out[f"{c}_alert_high"] = float((high & msk & (pred == c)).sum() / msk.sum())
            out[f"{c}_predicted_normal"] = int((msk & (pred == "normal")).sum())
            out[f"{c}_predicted_normal_confident"] = int((msk & (pred == "normal") & (t == "normal")).sum())
    nm = lab == "normal"
    out["normal_false_alert_any"] = float((alert & nm).sum() / nm.sum())
    out["normal_false_alert_high"] = float((high & nm).sum() / nm.sum())
    out["alerts_total"] = int(alert.sum())
    out["alert_ppv"] = float((alert & (pred == lab)).sum() / max(alert.sum(), 1))
    out["uncertain_share"] = float((t == "uncertain").mean())
    out["accuracy"] = float((pred == lab).mean())
    return out


def main():
    test = pd.read_csv(SRC / "predictions_test.csv")
    soop = pd.read_csv(SRC / "predictions_soop_external.csv")
    assert len(test) == 8930 and len(soop) == 1672, (len(test), len(soop))
    R = {}
    for name, df in (("AB_pooled", test), ("A", test[test["dataset"] != "CPAISD_v2"]), ("B_cpaisd", test[test["dataset"] == "CPAISD_v2"]),
                     ("C_soop", soop), ("AB_CT_only", test[~test["dataset"].isin(MRI)]), ("AB_MRI_only", test[test["dataset"].isin(MRI)])):
        p = df[PC].to_numpy(float)
        p = p / p.sum(1, keepdims=True)
        lab = df["label"].to_numpy()
        tt = np.where(df["dataset"].isin(MRI), T["MRI"], T["CT"])[:, None]
        q = np.power(np.clip(p, 1e-12, 1), 1 / tt)
        q = q / q.sum(1, keepdims=True)
        old_pred = np.array(LABELS)[p.argmax(1)]
        v2_idx = (p * W).argmax(1)
        v2_pred = np.array(LABELS)[v2_idx]
        R[name] = {"B4_reproduction_plain_argmax_raw": summarise(lab, old_pred, p.max(1)),
                   "v2_locked_rule_calibrated_display": summarise(lab, v2_pred, q[np.arange(len(q)), v2_idx]),
                   "v2_predicted_class_not_largest_displayed_prob": int((q.argmax(1) != v2_idx).sum())}
        for k, v in R[name].items():
            print(f"{name} {k}: {json.dumps({a: b for a, b in v.items() if a != 'tier_by_true_class'}) if isinstance(v, dict) else v}",
                  flush=True)
    with open(OUT / "deployed_tiers.json", "w") as f:
        json.dump(R, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
