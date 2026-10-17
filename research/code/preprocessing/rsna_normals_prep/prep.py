"""
NeuroScan Phase Q -- RSNA_ICH within-source normal slices (CPU-only).

Why: RSNA_ICH is 100% hemorrhagic in the corpus (8,000 rows, 0 normal)
because Phases J/L sampled only any==1 slices. The shortcut check found a
linear probe separates hemorrhagic slices by source (RSNA_ICH vs Kaggle_CT
vs SICH_CT) at 99.97%, and both Round 12 and Round 18 score RSNA_ICH at
exactly 100% -- a single-class source that can be scored by recognising the
source rather than the bleed. Adding RSNA's own no-bleed slices as "normal"
removes that shortcut.

Selection rules:
  - Only patients already present in split v3 (patient = real DICOM
    PatientID from anjum48/rsna-ich-metadata, the same key split v2/v3 used)
    so every new row inherits that patient's split -- no new leakage risk.
  - any == 0 in stage_1_train.csv, AND not adjacent (+/-1 in z order within
    the same series) to any any == 1 slice -- avoids partial-bleed margins.
  - Near-blank slices dropped (fraction of pixels > 10 must be >= 0.10).
  - Capped at 1:1 with that split's existing RSNA_ICH hemorrhagic count
    (seed 42).

Output format matches Phases J/L exactly: same guiferviz/rsna_stage1_png_128
source PNGs, same convert("L").resize(224, BICUBIC), filename {image_id}.png,
has_context=False (existing RSNA rows are single-slice).

Outputs: rsna_normal_rows.csv (split v3 schema + a `split` column) and
rsna_normal_images.zip (RSNA_ICH/normal/*.png).
"""
import random
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

INPUT_ROOT = Path("/kaggle/input")
OUT_ROOT = Path("/kaggle/working")
IMG_OUT = OUT_ROOT / "RSNA_ICH" / "normal"
TARGET_SIZE = (224, 224)
SEED = 42
BLANK_PIXEL_THRESHOLD = 10
MIN_FOREGROUND_FRACTION = 0.10


def find_file(name, path_contains=None):
    c = sorted(INPUT_ROOT.rglob(name))
    if path_contains is not None:
        c = [x for x in c if path_contains in str(x)]
    print(f"Found {len(c)} candidate(s) for '{name}': {[str(x) for x in c]}")
    return c[0] if c else None


def main():
    t0 = time.time()
    IMG_OUT.mkdir(parents=True, exist_ok=True)

    # --- metadata: image -> patient, series, z ---
    meta_path = find_file("train_metadata.parquet*")
    meta = pd.read_parquet(meta_path)
    if "Image" not in meta.columns:
        meta = meta.reset_index()
    id_col = "Image" if "Image" in meta.columns else "SOPInstanceUID"
    meta = meta[[id_col, "PatientID", "SeriesInstanceUID", "ImagePositionPatient_2"]].rename(
        columns={id_col: "image_id", "ImagePositionPatient_2": "z"})
    print(f"Metadata rows: {len(meta)}")

    # --- existing RSNA rows: patient -> split, count per split ---
    existing = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(find_file(f"{split}.csv", "patient-split-v3"))
        df = df[df["dataset"] == "RSNA_ICH"].copy()
        df["split"] = split
        existing.append(df)
    existing = pd.concat(existing, ignore_index=True)
    existing["image_id"] = existing["out_path"].map(lambda p: Path(p).stem)
    existing = existing.merge(meta[["image_id", "PatientID"]], on="image_id", how="left")
    unmatched = existing["PatientID"].isna().sum()
    print(f"Existing RSNA rows: {len(existing)}, unmatched to metadata: {unmatched}")
    assert unmatched == 0, "some existing RSNA rows have no PatientID"
    spans = existing.groupby("PatientID")["split"].nunique()
    assert (spans == 1).all(), "an RSNA patient spans multiple splits -- split v3 assumption broken"
    patient_split = existing.groupby("PatientID")["split"].first().to_dict()
    target_per_split = existing["split"].value_counts().to_dict()
    print(f"Patients: {len(patient_split)}, target per split: {target_per_split}")

    # --- labels: any per image ---
    labels = pd.read_csv(find_file("stage_1_train.csv"))
    labels = labels[labels["ID"].str.endswith("_any")].copy()
    labels["image_id"] = labels["ID"].str.rsplit("_", n=1).str[0]
    labels = labels.groupby("image_id", as_index=False)["Label"].max().rename(columns={"Label": "any"})

    # --- all slices of in-split patients, with z-order adjacency to positives ---
    pts = meta[meta["PatientID"].isin(patient_split)].merge(labels, on="image_id", how="inner")
    pts = pts.sort_values(["SeriesInstanceUID", "z"]).reset_index(drop=True)
    grp = pts.groupby("SeriesInstanceUID")["any"]
    pts["prev_any"] = grp.shift(1).fillna(0)
    pts["next_any"] = grp.shift(-1).fillna(0)
    stats = {
        "slices_of_in_split_patients": int(len(pts)),
        "positive": int((pts["any"] == 1).sum()),
        "adjacent_to_positive": int(((pts["any"] == 0) & ((pts["prev_any"] == 1) | (pts["next_any"] == 1))).sum()),
    }
    cand = pts[(pts["any"] == 0) & (pts["prev_any"] == 0) & (pts["next_any"] == 0)].copy()
    cand = cand[~cand["image_id"].isin(set(existing["image_id"]))]
    cand["split"] = cand["PatientID"].map(patient_split)
    stats["eligible_before_blank_filter"] = int(len(cand))
    print(f"Selection stats: {stats}")

    images_dir = find_file("stage_1_train_images")
    rng = random.Random(SEED)
    rows, blank, missing = [], 0, 0
    for split in ("train", "val", "test"):
        pool = cand[cand["split"] == split]["image_id"].tolist()
        rng.shuffle(pool)
        kept = 0
        for image_id in pool:
            if kept >= target_per_split.get(split, 0):
                break
            src = images_dir / f"{image_id}.png"
            if not src.exists():
                missing += 1
                continue
            img = Image.open(src).convert("L")
            if (np.asarray(img) > BLANK_PIXEL_THRESHOLD).mean() < MIN_FOREGROUND_FRACTION:
                blank += 1
                continue
            img.resize(TARGET_SIZE, Image.BICUBIC).save(IMG_OUT / f"{image_id}.png")
            rows.append({
                "source_key": f"rsna_stage1_png_128/stage_1_train_images/{image_id}.png",
                "dataset": "RSNA_ICH",
                "modality": "CT",
                "label": "normal",
                "out_path": f"RSNA_ICH/normal/{image_id}.png",
                "prev_out_path": np.nan,
                "next_out_path": np.nan,
                "has_context": False,
                "dwi_out_path": np.nan,
                "adc_out_path": np.nan,
                "split": split,
            })
            kept += 1
        print(f"  [{split}] eligible={len(pool)}, target={target_per_split.get(split, 0)}, kept={kept}")
    stats.update({"blank_dropped": blank, "missing_png": missing})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_ROOT / "rsna_normal_rows.csv", index=False)
    out_pat = out.assign(image_id=out["out_path"].map(lambda p: Path(p).stem)).merge(
        meta[["image_id", "PatientID"]], on="image_id")
    print(out_pat.groupby("split").agg(rows=("out_path", "size"), patients=("PatientID", "nunique")))
    print(f"Final stats: {stats}")

    with zipfile.ZipFile(OUT_ROOT / "rsna_normal_images.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for f in IMG_OUT.glob("*.png"):
            zf.write(f, arcname=f"RSNA_ICH/normal/{f.name}")
    shutil.rmtree(OUT_ROOT / "RSNA_ICH", ignore_errors=True)
    print(f"Done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
