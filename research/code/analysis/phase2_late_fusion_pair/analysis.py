"""
NeuroScan Phase Q -- Development Phase 2: R33 (CT adjacent-slice context by shared-encoder late fusion) seed 123 vs
R30B (Phase 2 reference) seed 123. VAL2 ONLY. No training, no images. Pre-registered 2026-10-05 (R33 row):
  primary   = AISD ischemic AUROC / PR-AUC (paired patient-cluster bootstrap);
  secondary = full-val2 ischemic AUROC / PR-AUC;
  guardrails = rule v2b (a) macro-F1 not significantly worse, (c)/(c') both recalls held, hemorrhagic AUROC / PR-AUC
               not significantly worse; criterion (d) vs the R30B s123 / s7 val2 spread; CPAISD reported only;
  mechanism = AISD (CPAISD reported) slice-position dependence of misses (patient-clustered logistic model of an
               ischemic miss on edge distance, relpos from the patient's val2 slice range) and normal false-positive
               rate by slice-position third, R30B s123 / s7 vs R33. A ranking gain with unchanged position dependence
               is not evidence that context fixed the shortcut (32 AISD val2 patients: descriptive).
Sequential seeds: seed 7 only if s123 passes v2b with no guardrail failure and a positive primary endpoint.
Note: R33 s123 was interrupted after epoch 16 and resumed (re-seeded shuffle for epochs 17-24); its best checkpoint
(epoch 9) predates the interruption.
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path


def ensure_pkg(import_name, pip_name=None):
    try:
        __import__(import_name)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name or import_name], check=True)


ensure_pkg("sklearn", "scikit-learn")
ensure_pkg("scipy")
ensure_pkg("statsmodels")

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score
import statsmodels.api as sm

INPUT_ROOT = Path("/kaggle/input")
DS = INPUT_ROOT / "datasets/atik68"
SOURCES = {"round30bs123": DS / "neuroscan-q-round30bs123-preds", "round33s123": DS / "neuroscan-q-round33s123-preds"}
OPTIONAL = {"round30bs7": DS / "neuroscan-q-round30bs7-preds"}
CAND, REF = "round33s123", "round30bs123"
PHASE2 = DS / "neuroscan-q-phase2-split-data"
OUT = Path("/kaggle/working/output")
OUT.mkdir(parents=True, exist_ok=True)
LABELS = ("normal", "ischemic", "hemorrhagic")
L2I = {l: i for i, l in enumerate(LABELS)}
GRID = np.exp(np.linspace(np.log(0.25), np.log(4.0), 41))
N_BOOT = 2000
N_VAL2 = 10360
KEY = ["source_key", "out_path"]
CPAISD_RE = re.compile(r"/([\d.]{10,})/(\d+)/raw\.dcm$")
MRI_RE = re.compile(r"^(sub-[\w]+)/(ses-\w+)/")
KEY_METRICS = ["accuracy", "macro_f1", "normal_recall", "ischemic_recall", "ischemic_precision",
               "hemorrhagic_recall", "hemorrhagic_precision"]
RANK_METRICS = ["ischemic_auroc", "ischemic_pr_auc", "hemorrhagic_auroc", "hemorrhagic_pr_auc"]
SET_DS = {"full_val2": None, "aisd_val2": "AISD", "isles_val2": "ISLES2022_dwi_adc", "cpaisd_val2": "CPAISD_v2",
          "soop_val2": "SOOP_dwi_adc", "rsna_val2": "RSNA_ICH", "sich_val2": "SICH_CT", "kagglect_val2": "Kaggle_CT"}
TARGETED = ("aisd_val2",)  # pre-registered primary source
T0 = time.time()
REPORT = {"phase": "Development Phase 2", "split": "val2 only", "candidate": CAND, "reference": REF}


def log(m):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {m}", flush=True)


def find(p):
    p = Path(p)
    if p.exists():
        return p
    for cand in [INPUT_ROOT / p.parent.name, *INPUT_ROOT.glob(f"*/{p.parent.name}"), *INPUT_ROOT.glob(f"*/*/{p.parent.name}")]:
        if (cand / p.name).exists():
            print(f"[WARN] {p} missing; using {cand / p.name}", flush=True)
            return cand / p.name
    return None


def need(p):
    q = find(p)
    if q is None:
        print(f"[ERROR] expected input missing: {p}; mounted: {sorted(str(x) for x in INPUT_ROOT.glob('*/*/*') if x.is_dir())[:40]}")
        sys.exit(1)
    return q


def pcols(m):
    return [f"p_{l}_{m}" for l in LABELS]


def load_val2():
    srcs = dict(SOURCES)
    for m, d in OPTIONAL.items():
        if find(d / "predictions_val.csv") is not None:
            srcs[m] = d
            log(f"optional source mounted: {m}")
        else:
            log(f"optional source NOT mounted: {m} (criterion (d) will be PENDING)")
    base = None
    for m, d in srcs.items():
        df = pd.read_csv(need(d / "predictions_val.csv"))
        if len(df) != N_VAL2 or df.duplicated(KEY).any():
            print(f"[ERROR] {m}: {len(df)} rows (expect {N_VAL2}), duplicates {int(df.duplicated(KEY).sum())}")
            sys.exit(1)
        if base is None:
            base = df[["dataset", "label", "subject", "slice", "lesion_px"] + KEY + pcols(m)].copy()
            continue
        chk = base[KEY + ["label"]].merge(df[KEY + ["label"]], on=KEY, suffixes=("", "_o"))
        if len(chk) != N_VAL2 or (chk["label"] != chk["label_o"]).any():
            print(f"[ERROR] {m}: val2 rows/labels do not match ({len(chk)} matched)")
            sys.exit(1)
        base = base.merge(df[KEY + pcols(m)], on=KEY, how="inner", validate="one_to_one")
        log(f"val2: merged {m} ({len(base)} rows; identical rows and labels)")
    return base, list(srcs)


def probs(df, m):
    return df[pcols(m)].to_numpy(dtype=float)


def cls_metrics(y, pred):
    cm = np.bincount(y * 3 + pred, minlength=9).reshape(3, 3)
    r = {"n": int(len(y)), "accuracy": float(np.trace(cm) / max(len(y), 1))}
    f1s = []
    for name, c in zip(LABELS, range(3)):
        tp, pos, prd = cm[c, c], cm[c].sum(), cm[:, c].sum()
        if pos:
            r[f"{name}_recall"] = float(tp / pos)
        if prd:
            r[f"{name}_precision"] = float(tp / prd)
        f1s.append(2 * tp / (pos + prd) if (pos + prd) else 0.0)
    r["macro_f1"] = float(np.mean(f1s))
    return r


def rank_metrics(y, p):
    r = {}
    for name, c in (("ischemic", 1), ("hemorrhagic", 2)):
        yb = y == c
        if 0 < yb.sum() < len(yb):
            r[f"{name}_auroc"] = float(roc_auc_score(yb, p[:, c]))
            r[f"{name}_pr_auc"] = float(average_precision_score(yb, p[:, c]))
    return r


def weighted_pred(p, w):
    return (p * np.array([1.0, w[0], w[1]])).argmax(1)


def choose(y, p, min_hem_recall=None, min_isch_recall=None):
    """Original rule (no targets) or rule v2b (both recalls held)."""
    best, best_key, feasible = None, None, 0
    for wi in GRID:
        for wh in GRID:
            m = cls_metrics(y, weighted_pred(p, (wi, wh)))
            if min_hem_recall is not None and m.get("hemorrhagic_recall", 0.0) < min_hem_recall:
                continue
            if min_isch_recall is not None and m.get("ischemic_recall", 0.0) < min_isch_recall:
                continue
            feasible += 1
            key = (round(m["macro_f1"], 10), -np.hypot(np.log(wi), np.log(wh)))
            if best_key is None or key > best_key:
                best, best_key = (float(wi), float(wh)), key
    return best, (float(best_key[0]) if best_key else None), feasible


RSNA_PAT = {}


def patient_key(ds, sk, op, subject):
    """Identical to the Phase 2 split, integrity and training kernels."""
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
        return f"SOOP:{subject}" if isinstance(subject, str) and subject else f"SOOP_row:{sk}"
    if ds == "RSNA_ICH":
        stem = Path(op).stem
        pid = RSNA_PAT.get(stem)
        return f"RSNA:{pid}" if isinstance(pid, str) and pid else f"RSNA_row:{stem}"
    if ds == "SICH_CT":
        return f"SICH:{sk.replace(chr(92), '/').split('/')[-2]}"
    return f"{ds}_row:{op}"


def add_ids(df):
    df = df.reset_index(drop=True)
    df["patient"] = [patient_key(a, b, c, s) for a, b, c, s in zip(df["dataset"], df["source_key"], df["out_path"], df["subject"])]
    return df


def cluster_groups(df):
    groups = {}
    for i, (d, c) in enumerate(zip(df["dataset"].to_numpy(), df["patient"].to_numpy())):
        groups.setdefault(d, {}).setdefault(c, []).append(i)
    return [[np.array(v) for v in g.values()] for g in groups.values()]


def all_metrics(df, m, w):
    y = df["label"].map(L2I).to_numpy()
    p = probs(df, m)
    return {**cls_metrics(y, weighted_pred(p, w)), **rank_metrics(y, p)}


def paired(df, a, b, wa, wb, seed):
    rng = np.random.default_rng(seed)
    y = df["label"].map(L2I).to_numpy()
    pa, pb = probs(df, a), probs(df, b)
    da, db = weighted_pred(pa, wa), weighted_pred(pb, wb)
    per_ds = cluster_groups(df)
    diffs = {k: [] for k in KEY_METRICS + RANK_METRICS}
    for _ in range(N_BOOT):
        idx = np.concatenate([np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))]) for cl in per_ds])
        yi = y[idx]
        ma = {**cls_metrics(yi, da[idx]), **rank_metrics(yi, pa[idx])}
        mb = {**cls_metrics(yi, db[idx]), **rank_metrics(yi, pb[idx])}
        for k in diffs:
            if k in ma and k in mb:
                diffs[k].append(ma[k] - mb[k])
    oa, ob = {**cls_metrics(y, da), **rank_metrics(y, pa)}, {**cls_metrics(y, db), **rank_metrics(y, pb)}
    out = {"n": int(len(y)), "clusters": int(sum(len(c) for c in per_ds))}
    for k, v in diffs.items():
        if k in oa and k in ob and v:
            d = np.array(v)
            out[k] = {"a": oa[k], "b": ob[k], "diff": oa[k] - ob[k],
                      "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                      "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}
    ca, cb = da == y, db == y
    nb, nc = int((ca & ~cb).sum()), int((~ca & cb).sum())
    out["mcnemar"] = {"a_only_right": nb, "b_only_right": nc,
                      "p_exact": float(binomtest(nb, nb + nc, 0.5).pvalue) if nb + nc else 1.0}
    return out


def per_patient(df, pred):
    x = df.assign(pred=np.array(LABELS)[pred])
    x = x[~x["patient"].str.contains("_row:")]
    out = {}
    for ds, g in x.groupby("dataset"):
        rows = []
        for _, t in g.groupby("patient"):
            les, nor = t[t["label"] != "normal"], t[t["label"] == "normal"]
            rows.append({"lesion": len(les), "caught": int((les["pred"] == les["label"]).sum()),
                         "normal": len(nor), "flagged": int((nor["pred"] != "normal").sum())})
        r = pd.DataFrame(rows)
        lp, nr = r[r.lesion > 0], r[r.normal > 0]
        out[ds] = {"detected": f"{int((lp.caught > 0).sum())}/{len(lp)}",
                   "normal_patients_flagged": f"{int((nr.flagged > 0).sum())}/{len(nr)}",
                   "false_alarm_rate": round(float(nr.flagged.sum() / max(nr.normal.sum(), 1)), 4)}
    return out


def scale(k):
    return 1 if k.endswith(("auroc", "pr_auc")) else 100


def fmt(k, v, a, b):
    s = scale(k)
    return (f"{k:22s} {a[:12]}={v['a'] * s:7.3f} {b[:12]}={v['b'] * s:7.3f} diff={v['diff'] * s:+7.3f} "
            f"CI95=[{v['ci95'][0] * s:+.3f}, {v['ci95'][1] * s:+.3f}] p={v['p_two_sided']:.3f}")


def worse(v):
    return v["diff"] < 0 and v["p_two_sided"] < 0.05


def better(v):
    return v["diff"] > 0 and v["p_two_sided"] < 0.05


def adoption(res, spread):
    b_sig = {k: better(res[k]) for k in RANK_METRICS if k in res}
    crit = {"a_macro_f1_not_significantly_worse": not worse(res["macro_f1"]),
            "b_threshold_free_significantly_better": b_sig,
            "c_ischemic_recall_not_significantly_worse": not worse(res["ischemic_recall"]),
            "c2_hemorrhagic_recall_not_significantly_worse": not worse(res["hemorrhagic_recall"])}
    v2b = (crit["a_macro_f1_not_significantly_worse"] and any(b_sig.values())
           and crit["c_ischemic_recall_not_significantly_worse"] and crit["c2_hemorrhagic_recall_not_significantly_worse"])
    if spread is None:
        crit["d"] = "PENDING (needs the R30B s7 val2 run spread)"
        d = None
    else:
        crit["d_macro_f1_gain_exceeds_run_range"] = res["macro_f1"]["diff"] > spread["macro_f1"]["range"]
        crit["d_threshold_free_gain_exceeds_run_range"] = {k: bool(ok and res[k]["diff"] > spread[k]["range"])
                                                           for k, ok in b_sig.items()}
        d = crit["d_macro_f1_gain_exceeds_run_range"] and any(crit["d_threshold_free_gain_exceeds_run_range"].values())
    return crit, bool(v2b), (None if d is None else bool(v2b and d))


def position_dependence(val, m, w):
    """AISD / CPAISD: ischemic recall and normal false-positive rate by slice-position third, and a patient-clustered
    logistic model of a miss on edge distance; relpos = position within the patient's val2 slice range."""
    out = {}
    for ds in ("AISD", "CPAISD_v2"):
        d = val[val["dataset"] == ds].copy()
        if ds == "AISD":
            d["sl"] = [int(re.sub(r"\D", "", Path(str(s)).stem) or -1) for s in d["source_key"]]
        else:
            d["sl"] = [int(CPAISD_RE.search(str(s).replace("\\", "/")).group(2)) for s in d["source_key"]]
        lo, hi = d.groupby("patient")["sl"].transform("min"), d.groupby("patient")["sl"].transform("max")
        d["relpos"] = (d["sl"] - lo) / (hi - lo).replace(0, np.nan)
        d = d.dropna(subset=["relpos"])
        p = probs(d, m)
        r = {}
        for tag, ww in (("argmax", (1.0, 1.0)), ("rule", w)):
            d["pred"] = np.array(LABELS)[weighted_pred(p, ww)]
            d["band"] = pd.cut(d["relpos"], [-0.01, 1 / 3, 2 / 3, 1.01], labels=["lower", "middle", "upper"])
            isch, norm = d[d["label"] == "ischemic"].copy(), d[d["label"] == "normal"]
            rr = {"ischemic_recall_by_third": {str(b): round(float((g["pred"] == "ischemic").mean()), 4)
                                               for b, g in isch.groupby("band", observed=True)},
                  "normal_fp_by_third": {str(b): round(float((g["pred"] != "normal").mean()), 4)
                                         for b, g in norm.groupby("band", observed=True)}}
            isch["miss"] = (isch["pred"] != "ischemic").astype(int)
            isch["relpos_edge"] = np.abs(isch["relpos"] - 0.5) * 2
            if isch["miss"].sum() >= 5:
                try:
                    fit = sm.GLM(isch["miss"], sm.add_constant(isch[["relpos_edge"]]), family=sm.families.Binomial()).fit(
                        cov_type="cluster", cov_kwds={"groups": pd.factorize(isch["patient"])[0]})
                    ci = np.exp(fit.conf_int().loc["relpos_edge"])
                    rr["edge_miss_or"] = {"or": round(float(np.exp(fit.params["relpos_edge"])), 2),
                                          "ci95": [round(float(ci[0]), 2), round(float(ci[1]), 2)],
                                          "p": round(float(fit.pvalues["relpos_edge"]), 4)}
                except Exception as e:  # noqa: BLE001
                    rr["edge_miss_or"] = f"failed: {e!r}"[:120]
            rr["misses"] = int(isch["miss"].sum())
            r[tag] = rr
        out[ds] = r
    return out


def main():
    rmap = pd.read_csv(need(PHASE2 / "rsna_patient_map.csv"))
    RSNA_PAT.update(dict(zip(rmap["stem"].astype(str), rmap["PatientID"])))
    v2_patients = set(pd.read_csv(need(PHASE2 / "phase2_val2_patients.csv"))["patient_key"])
    val, runs = load_val2()
    val = add_ids(val)
    outside = int((~val["patient"].isin(v2_patients)).sum())
    log(f"val2 rows {len(val)}; patients {val['patient'].nunique()}; rows whose patient key is not in the val2 list: {outside}")
    if outside:
        print("[ERROR] predictions contain rows outside the registered val2 patients")
        sys.exit(1)
    log("rows per dataset: " + str(val["dataset"].value_counts().to_dict()))
    sets = {s: (val if d is None else val[val["dataset"] == d].reset_index(drop=True)) for s, d in SET_DS.items()}
    yv = val["label"].map(L2I).to_numpy()

    W = {}
    for m in runs:
        W[m], f1, _ = choose(yv, probs(val, m))
        log(f"original rule: {m:13s} weights {tuple(round(x, 3) for x in W[m])} (val2 macro-F1 {f1:.4f})")
    REPORT["weights_original_rule"] = W

    # Descriptive per-run metrics (original-rule weights), every set.
    REPORT["per_run"] = {s: {m: all_metrics(df, m, W[m]) for m in runs} for s, df in sets.items()}
    for s in sets:
        per = REPORT["per_run"][s]
        log(f"PER RUN {s}: " + "; ".join(f"{k} " + " / ".join(f"{per[m][k] * scale(k):.3f}" for m in runs)
                                         for k in KEY_METRICS + RANK_METRICS if all(k in per[m] for m in runs))
            + f"   [{' / '.join(runs)}]")

    # Val2 run spread of the unchanged recipe (only if R30B s7 is mounted).
    spread = None
    if "round30bs7" in runs:
        ref_runs = ("round30bs123", "round30bs7")
        REPORT["run_spread_val2"] = {}
        for s in sets:
            per = REPORT["per_run"][s]
            REPORT["run_spread_val2"][s] = {k: {"runs": {m: per[m][k] for m in ref_runs},
                                                "range": float(abs(per[ref_runs[0]][k] - per[ref_runs[1]][k]))}
                                            for k in KEY_METRICS + RANK_METRICS if all(k in per[m] for m in ref_runs)}
        spread = REPORT["run_spread_val2"]["full_val2"]
        log("RUN SPREAD full_val2 (R30B s123 vs s7): " + "; ".join(f"{k} {v['range'] * scale(k):.3f}" for k, v in spread.items()))

    # Rule v2b: candidate at the operating point that holds both reference recalls on val2.
    wref = W[REF]
    ref_m = cls_metrics(yv, weighted_pred(probs(val, REF), wref))
    th, ti = ref_m["hemorrhagic_recall"], ref_m["ischemic_recall"]
    wc, f1, feas = choose(yv, probs(val, CAND), min_hem_recall=th, min_isch_recall=ti)
    rec = {"reference_weights": wref, "targets_val2_recall": {"hemorrhagic": th, "ischemic": ti},
           "candidate_weights_v2b": wc, "candidate_val2_macro_f1": f1, "feasible_grid_points": feas}
    log(f"== V2B {CAND} vs {REF} (ref weights {tuple(round(x, 3) for x in wref)}; targets hem R {th:.4f}, isch R {ti:.4f})"
        f" -> {CAND} weights {tuple(round(x, 3) for x in wc) if wc else None}, {feas} feasible")
    if wc is None:
        rec.update({"passes_v2b": False, "reason": "no grid point holds both reference recalls"})
        REPORT["v2b"] = rec
        REPORT["decision"] = {"text": "R33 s123 FAILS (no feasible operating point) -- R33 closed, seed 7 not run"}
        log(f"== DECISION: {REPORT['decision']}")
        with open(OUT / "phase2_r33_s123_pair.json", "w") as f:
            json.dump(REPORT, f, indent=2, default=str)
        return
    rec["sets"] = {}
    for i, s in enumerate(sets):
        t0 = time.time()
        res = paired(sets[s], CAND, REF, wc, wref, i)
        rec["sets"][s] = res
        tag = "" if s == "full_val2" else ("  [TARGETED source]" if s in TARGETED else "  [per-dataset, reported]")
        log(f"   {s} (n={res['n']}, clusters={res['clusters']}, {time.time() - t0:.0f}s){tag}:")
        for k in KEY_METRICS + RANK_METRICS:
            if k in res:
                log(f"     {fmt(k, res[k], CAND, REF)}")
        mc = res["mcnemar"]
        log(f"     McNemar: {CAND}-only right {mc['a_only_right']}, {REF}-only right {mc['b_only_right']}, p={mc['p_exact']:.4f}")
    crit, v2b, full = adoption(rec["sets"]["full_val2"], spread)

    # Pre-registered secondary endpoints: targeted sources and hemorrhage non-degradation.
    targeted = {s: {k: {"diff": rec["sets"][s][k]["diff"], "p": rec["sets"][s][k]["p_two_sided"],
                        "significantly_better": better(rec["sets"][s][k]), "positive": rec["sets"][s][k]["diff"] > 0}
                    for k in ("ischemic_auroc", "ischemic_pr_auc") if k in rec["sets"][s]} for s in TARGETED}
    hem = {f"{s}:{k}": {"diff": rec["sets"][s][k]["diff"], "p": rec["sets"][s][k]["p_two_sided"],
                        "significantly_worse": worse(rec["sets"][s][k])}
           for s in ("full_val2", "rsna_val2", "sich_val2") for k in ("hemorrhagic_auroc", "hemorrhagic_pr_auc", "hemorrhagic_recall")
           if k in rec["sets"][s]}
    rec.update({"criteria_on_val2": crit, "passes_v2b": v2b, "passes_v2b_and_spread": full,
                "targeted_sources": targeted, "hemorrhage_non_degradation": hem,
                "hemorrhage_not_degraded": not any(v["significantly_worse"] for v in hem.values()),
                "per_patient_val2": {CAND: per_patient(val, weighted_pred(probs(val, CAND), wc)),
                                     REF: per_patient(val, weighted_pred(probs(val, REF), wref))}})
    REPORT["v2b"] = rec
    log(f"   criteria: {crit}")
    log(f"   PRIMARY (AISD): {targeted}")
    log(f"   hemorrhage non-degradation: not degraded = {rec['hemorrhage_not_degraded']}")
    log(f"   per-patient val2: {rec['per_patient_val2']}")
    primary_positive = all(v["positive"] for v in targeted["aisd_val2"].values())
    rec["primary_aisd_positive"] = primary_positive
    rec["mechanism"] = {m: position_dependence(val, m, W[m]) for m in runs}
    for m, mres in rec["mechanism"].items():
        log(f"   MECHANISM {m}: {mres}")
    passes = v2b and rec["hemorrhage_not_degraded"] and (full is not False) and primary_positive
    REPORT["decision"] = {
        "passes_v2b": v2b, "criterion_d": "PENDING" if full is None else full, "primary_aisd_positive": primary_positive,
        "hemorrhage_not_degraded": rec["hemorrhage_not_degraded"],
        "text": (("R33 s123 PASSES v2b" + (" (criterion (d) pending)" if full is None else " and (d)")
                  + " with a positive AISD primary endpoint -- run the seed-7 pair") if passes
                 else "R33 s123 FAILS -- R33 closed under sequential seeds; seed 7 not run")}
    log(f"== DECISION: {REPORT['decision']}")
    with open(OUT / "phase2_r33_s123_pair.json", "w") as f:
        json.dump(REPORT, f, indent=2, default=str)
    log("Done.")


if __name__ == "__main__":
    main()
