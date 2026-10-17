"""
NeuroScan Phase Q -- AISD within-source normal slices (CPU-only).

Why: AISD is 100% ischemic in the corpus (3,612 rows, 0 normal) because
Phase K's prep kept only slices with a visible lesion and discarded the
rest. The shortcut check (kernel neuroscan-q-shortcut-check) confirmed the
model uses "looks like AISD" as a proxy for "ischemic": a linear probe
separates AISD-ischemic from Kaggle_CT-ischemic at 99.8%, and 91.5% of the
unseen CPAISD_v2 normal slices Round 18 mislabels as ischemic sit nearest
the AISD-ischemic feature centroid. Adding AISD's own lesion-free slices as
"normal" means the source's appearance alone can no longer predict the
label.

Selection rules:
  - Only patients already present in split v3 (so every new row inherits
    that patient's existing split -- no new leakage risk).
  - Mask must be completely empty (no value at all: remote, clear, blurred,
    invisible, or infarct), AND the slice must not be adjacent (+/-1) to any
    slice with a non-empty mask -- avoids the lesion-margin labelling problem
    already documented for CPAISD_v2.
  - Near-blank slices dropped (fraction of pixels > 10 must be >= 0.10).
  - Capped at 1:1 with that split's existing AISD ischemic count (seed 42).

Output format matches Phase K exactly: same source PNGs, same
convert("L").resize(224, BICUBIC), filenames {patient}_{slice}.png, and the
same (prev, self, next) context scheme -- neighbours are the true adjacent
slices of the scan, clamped to self at the scan edge. Every context
neighbour of a selected slice is itself lesion-free by construction.

Outputs: aisd_normal_rows.csv (split v3 schema + a `split` column) and
aisd_normal_images.zip (AISD/normal/*.png).
"""
import random
import shutil
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


ensure_pkg("gdown")

import gdown
import numpy as np
import pandas as pd
from PIL import Image

INPUT_ROOT = Path("/kaggle/input")
OUT_ROOT = Path("/kaggle/working")
DL_DIR = OUT_ROOT / "aisd_raw"
IMG_OUT = OUT_ROOT / "AISD" / "normal"
IMAGE_ZIP_ID = "157f9aE3ZhRSdIuIbP2PRG8ub9JJWvMGk"
MASK_ZIP_ID = "1d08fFpEvK4D6YTKfRlNuv_OlIxigZxl6"
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


def download_and_extract(file_id, name):
    zp = DL_DIR / f"{name}.zip"
    dest = DL_DIR / name
    gdown.download(f"https://drive.google.com/uc?id={file_id}", str(zp), quiet=True)
    with zipfile.ZipFile(zp) as zf:
        zf.extractall(dest)
    zp.unlink()
    return dest


def foreground_ok(path):
    arr = np.asarray(Image.open(path).convert("L"))
    return (arr > BLANK_PIXEL_THRESHOLD).mean() >= MIN_FOREGROUND_FRACTION


def main():
    t0 = time.time()
    DL_DIR.mkdir(parents=True, exist_ok=True)
    IMG_OUT.mkdir(parents=True, exist_ok=True)

    # --- existing AISD rows: patient -> split, ischemic count per split ---
    existing = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(find_file(f"{split}.csv", "patient-split-v3"))
        df = df[df["dataset"] == "AISD"].copy()
        df["split"] = split
        existing.append(df)
    existing = pd.concat(existing, ignore_index=True)
    existing["patient"] = existing["source_key"].str.split("/").str[2]
    per_patient = existing.groupby("patient")["split"].nunique()
    assert (per_patient == 1).all(), "an AISD patient spans multiple splits -- split v3 assumption broken"
    patient_split = existing.groupby("patient")["split"].first().to_dict()
    target_per_split = existing["split"].value_counts().to_dict()
    print(f"Existing AISD rows: {len(existing)}, patients: {len(patient_split)}, per split: {target_per_split}")

    # --- download raw AISD ---
    images_dir = download_and_extract(IMAGE_ZIP_ID, "images")
    masks_dir = download_and_extract(MASK_ZIP_ID, "masks")
    image_files = sorted(images_dir.rglob("*.png"))
    mask_by_key = {(m.parent.name, m.stem): m for m in masks_dir.rglob("*.png")}
    print(f"Raw image files: {len(image_files)}, mask files: {len(mask_by_key)}")

    scans = {}
    for f in image_files:
        if f.parent.name in patient_split and f.stem.isdigit():
            scans.setdefault(f.parent.name, {})[int(f.stem)] = f

    # --- candidate selection per patient ---
    candidates = []  # (split, patient, slice_no)
    stats = {"slices_seen": 0, "nonempty_mask": 0, "adjacent_to_lesion": 0, "missing_mask": 0, "blank": 0}
    for patient, slices in scans.items():
        nonempty, has_mask = set(), set()
        for sn, f in slices.items():
            m = mask_by_key.get((patient, f.stem))
            if m is None:
                continue
            has_mask.add(sn)
            if np.asarray(Image.open(m)).any():
                nonempty.add(sn)
        for sn, f in slices.items():
            stats["slices_seen"] += 1
            if sn not in has_mask:
                stats["missing_mask"] += 1
                continue
            if sn in nonempty:
                stats["nonempty_mask"] += 1
                continue
            if (sn - 1) in nonempty or (sn + 1) in nonempty:
                stats["adjacent_to_lesion"] += 1
                continue
            candidates.append((patient_split[patient], patient, sn))
    print(f"Selection stats: {stats}; eligible before blank filter: {len(candidates)}")

    # --- blank filter + 1:1 cap per split ---
    rng = random.Random(SEED)
    selected = []
    for split in ("train", "val", "test"):
        pool = [c for c in candidates if c[0] == split]
        rng.shuffle(pool)
        kept = []
        for c in pool:
            if len(kept) >= target_per_split.get(split, 0):
                break
            if foreground_ok(scans[c[1]][c[2]]):
                kept.append(c)
            else:
                stats["blank"] += 1
        print(f"  [{split}] eligible={len(pool)}, target={target_per_split.get(split, 0)}, kept={len(kept)}")
        selected.extend(kept)

    # --- write images (selected slices + their context neighbours) and rows ---
    def out_path(patient, sn):
        return f"AISD/normal/{patient}_{scans[patient][sn].stem}.png"

    needed = set()
    rows = []
    for split, patient, sn in selected:
        prev_sn = sn - 1 if (sn - 1) in scans[patient] else sn
        next_sn = sn + 1 if (sn + 1) in scans[patient] else sn
        needed.update({(patient, sn), (patient, prev_sn), (patient, next_sn)})
        f = scans[patient][sn]
        rows.append({
            "source_key": f"AISD/image/{patient}/{f.name}",
            "dataset": "AISD",
            "modality": "CT",
            "label": "normal",
            "out_path": out_path(patient, sn),
            "prev_out_path": out_path(patient, prev_sn),
            "next_out_path": out_path(patient, next_sn),
            "has_context": True,
            "dwi_out_path": np.nan,
            "adc_out_path": np.nan,
            "split": split,
        })
    for patient, sn in needed:
        img = Image.open(scans[patient][sn]).convert("L").resize(TARGET_SIZE, Image.BICUBIC)
        img.save(OUT_ROOT / out_path(patient, sn))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_ROOT / "aisd_normal_rows.csv", index=False)
    print(f"Wrote {len(out)} rows; images written (incl. context neighbours): {len(needed)}")
    print(out.groupby("split").agg(rows=("out_path", "size"), patients=("source_key", lambda s: s.str.split('/').str[2].nunique())))

    with zipfile.ZipFile(OUT_ROOT / "aisd_normal_images.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for f in IMG_OUT.glob("*.png"):
            zf.write(f, arcname=f"AISD/normal/{f.name}")
    shutil.rmtree(DL_DIR, ignore_errors=True)
    shutil.rmtree(OUT_ROOT / "AISD", ignore_errors=True)
    print(f"Final stats: {stats}")
    print(f"Done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
