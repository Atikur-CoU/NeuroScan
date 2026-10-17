"""
NeuroScan Phase Q -- RSNA_ICH expansion for Round 24 (CPU, no training).

Why: hemorrhagic recall is limited mainly by RSNA_ICH (Round 23 test: RSNA
hemorrhagic recall 86.6%, 20.6% of RSNA normal slices flagged), while the model
ranks bleeds well (hemorrhagic AUROC 0.978). The corpus uses only 8,000 of
RSNA's bleed slices. More same-source data targets this directly without adding
a new source (so no new dataset shortcut).

What: +8,000 hemorrhagic slices and +8,000 matched normal slices, ALL assigned to
TRAIN (validation and test stay exactly as in Round 23, so results stay
comparable and no patient crosses splits):
  * hemorrhagic: any == 1 slices not already in the corpus, from patients who
    are either in split v3's TRAIN set or not in the split at all (new patients
    -> train). Patients in val/test are never used. At most 4 new bleed slices
    per patient (spreads the gain over more patients), new patients preferred.
  * normal: any == 0 slices from the SAME patients as the selected bleed slices,
    not z-adjacent (+/-1 within the series) to any any == 1 slice, 1:1 with the
    bleed slices per patient where possible (within-patient normals, as in the
    Round 19 normals prep).
  * near-blank slices dropped (fraction of pixels > 10 must be >= 0.10).
Image processing is identical to Phases J/L and the RSNA-normals prep:
guiferviz/rsna_stage1_png_128 PNG -> convert("L") -> resize(224, BICUBIC).

I/O: direct paths only; the 674k-file image folder is never listed (files are
opened by name); images are written straight into the zip.
Outputs: rsna_extra_rows.csv (split-v3 schema + split + patient) and
rsna_extra_images.zip (RSNA_ICH/extra/{hemorrhagic|normal}/<id>.png).
"""
import io
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

INPUT_ROOT = Path("/kaggle/input")
SPLIT_DIR = INPUT_ROOT / "datasets/atik68/neuroscan-phase-p-patient-split-v3"
RSNA_PNG = INPUT_ROOT / "datasets/guiferviz/rsna_stage1_png_128"
IMG_DIR = RSNA_PNG / "stage_1_train_images"
RSNA_META = INPUT_ROOT / "datasets/anjum48/rsna-ich-metadata/train_metadata.parquet.gzip"
RSNA_NORMALS = INPUT_ROOT / "notebooks/atik68/neuroscan-q-rsna-normals-prep/rsna_normal_rows.csv"
OUT = Path("/kaggle/working")
TARGET = 8000
PER_PATIENT_CAP = 4
SEED = 42
BLANK_PIXEL_THRESHOLD, MIN_FOREGROUND_FRACTION = 10, 0.10
T0 = time.time()


def log(m):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {m}", flush=True)


def need(p):
    if not Path(p).exists():
        print(f"[ERROR] expected input missing: {p}")
        sys.exit(1)
    return Path(p)


def main():
    rng = random.Random(SEED)
    meta = pd.read_parquet(need(RSNA_META))
    if "Image" not in meta.columns and "SOPInstanceUID" not in meta.columns:
        meta = meta.reset_index()
    idc = "Image" if "Image" in meta.columns else "SOPInstanceUID"
    meta = meta[[idc, "PatientID", "SeriesInstanceUID", "ImagePositionPatient_2"]].rename(
        columns={idc: "image_id", "ImagePositionPatient_2": "z"})
    lab = pd.read_csv(need(RSNA_PNG / "stage_1_train.csv"))
    lab = lab[lab["ID"].str.endswith("_any")].copy()
    lab["image_id"] = lab["ID"].str.rsplit("_", n=1).str[0]
    lab = lab.groupby("image_id", as_index=False)["Label"].max().rename(columns={"Label": "any"})
    m = meta.merge(lab, on="image_id")
    log(f"RSNA slices with metadata + labels: {len(m)}")

    # existing corpus RSNA rows (split v3 + normals prep): their image ids and patient -> split
    used, pat_split = set(), {}
    img2pat = dict(zip(m["image_id"], m["PatientID"]))
    for s in ("train", "val", "test"):
        df = pd.read_csv(need(SPLIT_DIR / f"{s}.csv"))
        ids = df.loc[df["dataset"] == "RSNA_ICH", "out_path"].map(lambda p: Path(p).stem)
        used |= set(ids)
        for i in ids:
            if i in img2pat:
                pat_split[img2pat[i]] = s
    normals = pd.read_csv(need(RSNA_NORMALS))
    used |= set(normals["out_path"].map(lambda p: Path(p).stem))
    log(f"Existing RSNA image ids in corpus: {len(used)}; split patients: {len(pat_split)} "
        f"(train {sum(v == 'train' for v in pat_split.values())})")

    # eligible patients: split-train patients + patients not in the split (-> train)
    m["split_of_patient"] = m["PatientID"].map(pat_split)
    elig = m[m["split_of_patient"].isna() | (m["split_of_patient"] == "train")].copy()
    elig = elig.sort_values(["SeriesInstanceUID", "z"])
    g = elig.groupby("SeriesInstanceUID")["any"]
    elig["adj_pos"] = (g.shift(1).fillna(0) == 1) | (g.shift(-1).fillna(0) == 1)
    elig = elig[~elig["image_id"].isin(used)]
    pos = elig[elig["any"] == 1]
    neg_ok = elig[(elig["any"] == 0) & ~elig["adj_pos"]]
    log(f"Eligible new bleed slices: {len(pos)} from {pos['PatientID'].nunique()} patients; "
        f"eligible non-adjacent normal slices: {len(neg_ok)}")

    # per-patient pools; new patients first, then split-train patients; shuffled within each group
    pos_by_pat = {p: list(v) for p, v in pos.groupby("PatientID")["image_id"]}
    neg_by_pat = {p: list(v) for p, v in neg_ok.groupby("PatientID")["image_id"]}
    pats = [p for p in pos_by_pat if p in neg_by_pat]  # need within-patient normals
    new_p = [p for p in pats if p not in pat_split]
    old_p = [p for p in pats if p in pat_split]
    rng.shuffle(new_p)
    rng.shuffle(old_p)
    order = new_p + old_p
    for p in order:
        rng.shuffle(pos_by_pat[p])
        rng.shuffle(neg_by_pat[p])

    def ok(image_id):
        f = IMG_DIR / f"{image_id}.png"
        if not f.exists():
            return None
        img = Image.open(f).convert("L")
        if (np.asarray(img) > BLANK_PIXEL_THRESHOLD).mean() < MIN_FOREGROUND_FRACTION:
            return None
        return img.resize((224, 224), Image.BICUBIC)

    rows, stats = [], {"blank_or_missing": 0}
    zpath = OUT / "rsna_extra_images.zip"
    n_hem = 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as zf:
        for p in order:
            if n_hem >= TARGET:
                break
            took_h = took_n = 0
            hems, negs = pos_by_pat[p], neg_by_pat[p]
            for iid in hems:
                if took_h >= PER_PATIENT_CAP or n_hem >= TARGET:
                    break
                img = ok(iid)
                if img is None:
                    stats["blank_or_missing"] += 1
                    continue
                # matching normal from the same patient
                img_n, nid = None, None
                while negs and img_n is None:
                    nid = negs.pop()
                    img_n = ok(nid)
                    if img_n is None:
                        stats["blank_or_missing"] += 1
                if img_n is None:
                    break  # no usable normal left for this patient -> keep 1:1
                for lbl, image_id, im in (("hemorrhagic", iid, img), ("normal", nid, img_n)):
                    rel = f"RSNA_ICH/extra/{lbl}/{image_id}.png"
                    buf = io.BytesIO()
                    im.save(buf, format="PNG")
                    zf.writestr(rel, buf.getvalue())
                    rows.append({"source_key": f"rsna_stage1_png_128/stage_1_train_images/{image_id}.png",
                                 "dataset": "RSNA_ICH", "modality": "CT", "label": lbl, "out_path": rel,
                                 "prev_out_path": np.nan, "next_out_path": np.nan, "has_context": False,
                                 "dwi_out_path": np.nan, "adc_out_path": np.nan, "split": "train",
                                 "patient": p, "patient_new_to_corpus": p not in pat_split})
                took_h += 1
                took_n += 1
                n_hem += 1
            if n_hem and n_hem % 1000 == 0:
                log(f"  {n_hem} bleed + {n_hem} normal slices")
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "rsna_extra_rows.csv", index=False)
    rep = {"rows": len(out), "by_label": out["label"].value_counts().to_dict(),
           "patients": int(out["patient"].nunique()),
           "patients_new_to_corpus": int(out.loc[out["patient_new_to_corpus"], "patient"].nunique()),
           "patients_already_in_train": int(out.loc[~out["patient_new_to_corpus"], "patient"].nunique()),
           "leak_check_val_test_patients_used": int(sum(pat_split.get(p) in ("val", "test") for p in out["patient"].unique())),
           "zip_MB": round(zpath.stat().st_size / 1e6, 1), **stats}
    log(f"Done: {rep}")
    if rep["leak_check_val_test_patients_used"]:
        print("[ERROR] val/test patient used -- leakage")
        sys.exit(1)


if __name__ == "__main__":
    main()
