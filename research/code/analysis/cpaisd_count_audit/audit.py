"""
NeuroScan -- CPAISD row-count audit (2026-10-08). CPU; reads split CSVs only (no images).
Question: the CPAISD release has 10,165 slices, but the final partition (patient split v3) holds fewer CPAISD rows.
Where, and why, were the other rows dropped? Compares the CPAISD rows of the Phase N, Phase O and Phase P (v1, v2, v3)
splits and describes every dropped row (label, context flag, study, slice position, duplicates).
v2 (2026-10-09): also audits every source across split v1 -> v2 -> v3 and joins the dropped rows with the Phase P
slice-quality flags (std < 8 or foreground < 5%), to state the near-blank training filter in the paper.
"""
import json
import re
import time
from pathlib import Path

import pandas as pd

T0 = time.time()
ROOTS = [Path("/kaggle/input/datasets/atik68"), Path("/kaggle/input")]
SPLITS = [("phase_n", "neuroscan-phase-n-3class-split"), ("phase_o", "neuroscan-phase-o-3class-split"),
          ("phase_p_v1", "neuroscan-phase-p-patient-split"), ("phase_p_v2", "neuroscan-phase-p-patient-split-v2"),
          ("phase_p_v3", "neuroscan-phase-p-patient-split-v3")]
KEY_RE = re.compile(r"/([\d.]{10,})/(\d+)/raw\.dcm$")
OUT = Path("/kaggle/working/output")
OUT.mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


def find(slug):
    for r in ROOTS:
        d = r / slug
        if d.exists():
            return d
    return None


def load(slug):
    d = find(slug)
    if d is None:
        return None
    parts = []
    for split in ("train", "val", "test"):
        hits = [d / f"{split}.csv"] if (d / f"{split}.csv").exists() else sorted(d.glob(f"*/{split}.csv"))
        if not hits:
            log(f"  {slug}: no {split}.csv")
            return None
        df = pd.read_csv(hits[0], low_memory=False)
        df["split"] = split
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


R, cp = {}, {}
for name, slug in SPLITS:
    df = load(slug)
    if df is None:
        log(f"{name}: dataset {slug} not mounted")
        continue
    c = df[df["dataset"].astype(str).str.startswith("CPAISD")].copy()
    k = c["source_key"].astype(str).str.extract(KEY_RE)
    c["study"], c["slice_idx"] = k[0], pd.to_numeric(k[1], errors="coerce")
    cp[name] = c
    R[name] = {"rows_all_datasets": int(len(df)), "cpaisd_rows": int(len(c)),
               "cpaisd_unique_source_key": int(c["source_key"].nunique()),
               "cpaisd_duplicate_source_key_rows": int(c["source_key"].duplicated().sum()),
               "cpaisd_studies": int(c["study"].nunique()),
               "labels": c["label"].value_counts().to_dict(), "by_split": c["split"].value_counts().to_dict(),
               "dataset_names": c["dataset"].value_counts().to_dict(),
               "has_context_true": int(c["has_context"].astype(str).eq("True").sum()) if "has_context" in c else None}
    log(f"{name}: {json.dumps(R[name])}")

names = [n for n, _ in SPLITS if n in cp]
R["transitions"] = {}
for a, b in zip(names, names[1:]):
    ka, kb = set(cp[a]["source_key"]), set(cp[b]["source_key"])
    gone = cp[a][cp[a]["source_key"].isin(ka - kb)]
    t = {"dropped": len(ka - kb), "added": len(kb - ka)}
    if len(gone):
        studies_all = cp[a].groupby("study").size()
        gone_s = gone.groupby("study").size()
        whole = [s for s in gone_s.index if gone_s[s] == studies_all.get(s, -1)]
        # relative slice position within the study (0 = first, 1 = last)
        rel = cp[a].groupby("study")["slice_idx"].agg(["min", "max"])
        g = gone.join(rel, on="study")
        pos = (g["slice_idx"] - g["min"]) / (g["max"] - g["min"]).replace(0, 1)
        t.update({"dropped_labels": gone["label"].value_counts().to_dict(),
                  "dropped_by_split": gone["split"].value_counts().to_dict(),
                  "dropped_has_context_true": int(gone["has_context"].astype(str).eq("True").sum()) if "has_context" in gone else None,
                  "studies_touched": int(gone_s.size), "studies_dropped_entirely": len(whole),
                  "rows_in_entirely_dropped_studies": int(gone_s[whole].sum()) if whole else 0,
                  "dropped_rel_position_quartiles": [round(float(x), 3) for x in pos.quantile([0, .25, .5, .75, 1])],
                  "dropped_at_volume_edge_(first_or_last_slice)": int(((pos == 0) | (pos == 1)).sum())})
        gone.to_csv(OUT / f"dropped_{a}_to_{b}.csv", index=False)
    R["transitions"][f"{a}->{b}"] = t
    log(f"{a} -> {b}: {json.dumps(t)}")

# ---------------------------------------------------------------- v2: all sources, joined with the quality flags
qf = next((f for r in (Path("/kaggle/input/notebooks/atik68"), Path("/kaggle/input"))
           for f in (r / "neuroscan-phase-p-slice-quality-filter").rglob("slice_quality_all_splits.csv")), None)
log(f"quality file: {qf}")
q = pd.read_csv(qf, low_memory=False)[["source_key", "low_quality", "foreground_frac", "std"]] if qf else None
full = {n: load(sl) for n, sl in SPLITS if n.startswith("phase_p")}
R["all_sources"] = {}
for a, b in (("phase_p_v1", "phase_p_v2"), ("phase_p_v2", "phase_p_v3")):
    A, B = full[a], full[b]
    gone = A[~A["source_key"].isin(set(B["source_key"]))].copy()
    if q is not None:
        gone = gone.merge(q, on="source_key", how="left")
        Aq = A.merge(q, on="source_key", how="left")
    per = {}
    for ds in sorted(set(A["dataset"])):
        g, tot = gone[gone["dataset"] == ds], A[A["dataset"] == ds]
        e = {"rows_before": int(len(tot)), "rows_after": int((B["dataset"] == ds).sum()), "dropped": int(len(g))}
        if len(g):
            e["dropped_labels"] = g["label"].value_counts().to_dict()
            e["dropped_by_old_split"] = g["split"].value_counts().to_dict()
            if q is not None:
                e["dropped_low_quality"] = g["low_quality"].astype(str).value_counts().to_dict()
                lq = Aq[(Aq["dataset"] == ds) & (Aq["low_quality"].astype(str) == "True")]
                e["low_quality_before"] = int(len(lq))
                kept = lq[lq["source_key"].isin(set(B["source_key"]))].groupby(["label", "split"]).size()
                e["low_quality_kept_by_label_split"] = {f"{k[0]}|{k[1]}": int(v) for k, v in kept.items()}
                e["dropped_foreground_median"] = float(pd.to_numeric(g["foreground_frac"], errors="coerce").median())
        per[ds] = e
    R["all_sources"][f"{a}->{b}"] = per
    log(f"{a} -> {b}: {json.dumps(per, default=str)}")
    gone.to_csv(OUT / f"all_dropped_{a}_to_{b}.csv", index=False)

with open(OUT / "cpaisd_count_audit.json", "w") as f:
    json.dump(R, f, indent=2, default=str)
log("Done.")
