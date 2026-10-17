"""
NeuroScan Multi-Dataset -- Phase B: Preprocessing (runs on Kaggle)

Builds one unified manifest across all 7 datasets: resizes every image to a
common size, normalizes intensity, converts to single-channel grayscale, and
writes a flat PNG plus one manifest row per image.

Reuses the exact per-dataset scanning/labeling logic verified locally in
eda_all_datasets.py (see PROCESS.md for how each dataset's structure was
hand-checked -- dedup rules, mask-based labeling for CPAISD/SICH_CT/ISLES2022,
etc.). Only two things differ from the local version:
  1. Paths point at /kaggle/input/<dataset-slug>/... instead of datasets/<name>
  2. Every scanner yields a stable string `source_key` alongside the image,
     so progress can be checkpointed and resumed at single-image granularity.

Resumability: manifest.csv is opened in append mode and flushed after every
row. A killed session leaves a partial-but-valid manifest; rerunning this
script skips every source_key already present in it and continues from
there -- it does not restart from zero, and does not reprocess whole
datasets, just the images not yet done.
"""
import csv
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def ensure_pkg(import_name, pip_name=None):
    try:
        __import__(import_name)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name or import_name], check=True)


ensure_pkg("pydicom")
ensure_pkg("nibabel")

import nibabel as nib
import pydicom

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

TARGET_SIZE = (224, 224)
# Kaggle's current mount scheme nests datasets under /kaggle/input/datasets/<owner>/
# (not the classic flat /kaggle/input/<slug>/ -- confirmed by a diagnostic run,
# see KAGGLE_WORKFLOW.md Phase B notes).
INPUT_ROOT = Path("/kaggle/input/datasets/atik68")
OUT_ROOT = Path("/kaggle/working/preprocessed")
MANIFEST_PATH = Path("/kaggle/working/manifest.csv")
MANIFEST_FIELDS = [
    "source_key", "dataset", "modality", "label",
    "out_path", "orig_height", "orig_width",
]
PROGRESS_EVERY = 2000
# When True, skip writing resized PNGs (manifest rows still get written).
# Used for fast debug-only runs so /kaggle/working output stays tiny and
# `kaggle kernels output` doesn't get stuck enumerating tens of thousands of
# image files -- that's exactly what happened trying to pull the full
# version-9 run's log (see KAGGLE_WORKFLOW.md Phase B notes).
DIAGNOSTIC_MODE = False

# All 7 confirmed working: Kaggle_CT/CPAISD/MRI_Images/SICH_CT/Tekno21/
# Multimodal_Stroke in the version-4 push, ISLES2022 in version 11 (after
# fixing Kaggle's auto-decompress corruption -- see KAGGLE_WORKFLOW.md).
ONLY_RUN = []

_ALL_DATASETS = {
    "Kaggle_CT": INPUT_ROOT / "neuroscan-kaggle-ct",
    "CPAISD": INPUT_ROOT / "neuroscan-cpaisd",
    "MRI_Images": INPUT_ROOT / "neuroscan-mri-images",
    "SICH_CT": INPUT_ROOT / "neuroscan-sich-ct",
    "Tekno21": INPUT_ROOT / "neuroscan-tekno21",
    "Multimodal_Stroke": INPUT_ROOT / "neuroscan-multimodal-stroke",
    "ISLES2022": INPUT_ROOT / "neuroscan-isles2022",
}
DATASETS = {k: v for k, v in _ALL_DATASETS.items() if k in ONLY_RUN} if ONLY_RUN else _ALL_DATASETS

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".dcm")
stroke_keywords = ("stroke", "sich", "hemorrhage", "haemorrhage", "ischemic")
normal_keywords = ("normal", "control", "healthy", "neg")

# --------------------------------------------------------------------------
# Shared readers / classifiers (verbatim logic from eda_all_datasets.py)
# --------------------------------------------------------------------------


def read_image_as_array(path: Path):
    suffix = path.suffix.lower()
    try:
        if suffix == ".dcm":
            ds = pydicom.dcmread(str(path))
            return np.asarray(ds.pixel_array)
        with Image.open(path) as im:
            return np.asarray(im)
    except Exception:
        return None


def classify_by_mask_array(arr) -> str:
    return "stroke" if np.count_nonzero(arr) > 0 else "normal"


def classify_by_mask_npz(mask_path: Path):
    if not mask_path.exists():
        return None
    try:
        with np.load(mask_path) as d:
            arr = d["mask"] if "mask" in d.files else d[d.files[0]]
        return classify_by_mask_array(arr)
    except Exception:
        return None


def classify_by_mask_image(mask_path: Path):
    if not mask_path.exists():
        return None
    try:
        arr = np.asarray(Image.open(mask_path))
        return classify_by_mask_array(arr)
    except Exception:
        return None


def has_bbox_annotation(xml_path: Path) -> bool:
    """SICH_CT ships a second, independent annotation format alongside the
    pixel masks: a Pascal-VOC-style XML bounding box per slice. Manual audit
    (PROCESS.md Sec 22) found 43 slices where the mask is fully empty but
    the bounding-box annotation marks a real lesion, visually confirmed on
    a sample -- the mask has coverage gaps the bounding box doesn't."""
    if not xml_path.exists():
        return False
    try:
        return "<object>" in xml_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False


# --------------------------------------------------------------------------
# Per-dataset scanners -- each yields (source_key, item, label, modality)
# item is a Path (read via read_image_as_array) or an ndarray (ISLES2022,
# whose slices don't each have their own file on disk).
# --------------------------------------------------------------------------


def scan_kaggle_ct(base: Path):
    # Locally this was base / "Brain_Stroke_CT_Dataset" -- Kaggle's extraction
    # strips that single top-level wrapper folder, so `base` already *is* it.
    root = base
    class_to_label = {"Bleeding": "stroke", "Ischemia": "stroke", "Normal": "normal"}
    for class_folder, label in class_to_label.items():
        png_dir = root / class_folder / "PNG"
        if not png_dir.exists():
            continue
        for p in sorted(png_dir.glob("*.png")):
            yield str(p), p, label, "CT"

    ext_png = root / "External_Test" / "PNG"
    ext_masks = root / "External_Test" / "MASKS"
    if ext_png.exists():
        for p in sorted(ext_png.glob("*.png")):
            label = classify_by_mask_image(ext_masks / p.name) or "other"
            yield str(p), p, label, "CT"


def scan_cpaisd(base: Path):
    root = base  # locally base / "dataset"; wrapper stripped on Kaggle
    if not root.exists():
        return
    for p in sorted(root.rglob("raw.dcm")):
        label = classify_by_mask_npz(p.parent / "mask.npz") or "other"
        yield str(p), p, label, "CT"


def scan_mri_images(base: Path):
    root = base / "Dataset_MRI_Folder"  # locally base / "dataset" / "Dataset_MRI_Folder"
    if not root.exists():
        return
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        rel_lower = str(p.relative_to(root)).lower()
        if any(k in rel_lower for k in stroke_keywords):
            label = "stroke"
        elif any(k in rel_lower for k in normal_keywords):
            label = "normal"
        else:
            label = "other"
        yield str(p), p, label, "MRI"


def scan_sich_ct(base: Path):
    root = base / "SubdatasetC_PNG"  # locally base / "PHE-SICH-CT-IDS" / "SubdatasetC_PNG"
    set_dir = root / "set"
    label_dir = root / "label"
    annotation_dir = root / "annotation"
    if not set_dir.exists():
        return
    for image_folder in sorted(set_dir.iterdir()):
        if not image_folder.is_dir():
            continue
        study_num = image_folder.name.replace("image", "")
        mask_folder = label_dir / f"label{study_num}"
        anno_folder = annotation_dir / f"annotation{study_num}"
        for p in sorted(image_folder.glob("*.png")):
            label = classify_by_mask_image(mask_folder / p.name)
            if label != "stroke" and has_bbox_annotation(anno_folder / f"{p.stem}.xml"):
                label = "stroke"
            label = label or "other"
            yield str(p), p, label, "CT"


def scan_tekno21(base: Path):
    root = base / "images"
    if not root.exists():
        return
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        rel_lower = str(p.relative_to(root)).lower()
        if "stroke" in rel_lower:
            label = "stroke"
        elif "normal" in rel_lower:
            label = "normal"
        else:
            label = "other"
        yield str(p), p, label, "CT"


def scan_multimodal_stroke(base: Path):
    root = base  # locally base / "deep"; wrapper stripped on Kaggle
    groups = [
        (root / "test" / "normalBT", "normal", "CT"),
        (root / "test" / "normalMR", "normal", "MRI"),
        (root / "test" / "strokeBT", "stroke", "CT"),
        (root / "test" / "strokeMR", "stroke", "MRI"),
        (root / "train" / "1- Stroke" / "StrokeBT", "stroke", "CT"),
        (root / "train" / "1- Stroke" / "StrokeMR", "stroke", "MRI"),
        (root / "train" / "2- Control" / "NormalBT", "normal", "CT"),
        (root / "train" / "2- Control" / "NormalMR", "normal", "MRI"),
    ]
    for folder, label, modality in groups:
        if not folder.exists():
            continue
        for p in sorted(folder.iterdir()):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                continue
            yield str(p), p, label, modality


def _load_nifti_flexible(folder: Path, stem: str):
    """Load a NIfTI volume for `stem` under `folder`, handling three cases:
      - <stem>.nii.gz  -- original format
      - <stem>.nii     -- Kaggle auto-decompressed it on dataset creation
      - <stem>.niigz   -- deliberately disguised extension (re-uploaded this
        way specifically because Kaggle's auto-decompress step was silently
        corrupting most .nii.gz files into 0-byte files -- confirmed via a
        diagnostic run showing "Empty file" for ~245/250 subjects even after
        switching the upload from zip to tar. Renaming to an extension
        Kaggle doesn't recognize stops it from touching the file at all; we
        gunzip it ourselves into a temp file before handing it to nibabel).
    Returns a loaded nibabel image, or None if nothing is found.
    """
    for ext in (".nii.gz", ".nii"):
        p = folder / f"{stem}{ext}"
        if p.exists():
            return nib.load(str(p))

    disguised = folder / f"{stem}.niigz"
    if disguised.exists():
        import gzip
        import tempfile
        with gzip.open(disguised, "rb") as f:
            raw = f.read()
        tmp = tempfile.NamedTemporaryFile(suffix=".nii", delete=False)
        try:
            tmp.write(raw)
        finally:
            tmp.close()
        return nib.load(tmp.name)

    return None


def scan_isles2022(base: Path):
    # Memory is the concern here, not logic: 250 subjects x 3 volumes each
    # (FLAIR alone is ~281x352x352). get_fdata() defaults to float64 --
    # loading as float32 halves that. Each volume/mask is dropped (`del` +
    # periodic gc.collect()) as soon as its slices are yielded, and any
    # per-subject failure is caught and logged rather than allowed to kill
    # the whole run silently (this is exactly what happened in the run that
    # produced 0 ISLES2022 rows with no visible error -- see
    # KAGGLE_WORKFLOW.md Phase B notes).
    import gc

    root = base  # locally base / "ISLES-2022"; wrapper stripped on Kaggle
    deriv_root = root / "derivatives"
    if not root.exists():
        return

    sub_dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("sub-"))
    print(f"    scan_isles2022: {len(sub_dirs)} subjects found under {root}")

    for sub_idx, sub_dir in enumerate(sub_dirs):
        sub_id = sub_dir.name
        try:
            for ses_dir in sorted(p for p in sub_dir.iterdir() if p.is_dir() and p.name.startswith("ses-")):
                ses_id = ses_dir.name
                mask_arr = None
                try:
                    mask_img = _load_nifti_flexible(deriv_root / sub_id / ses_id, f"{sub_id}_{ses_id}_msk")
                    if mask_img is not None:
                        mask_arr = mask_img.get_fdata(dtype=np.float32)
                except Exception as e:
                    print(f"    [WARN] {sub_id}/{ses_id}: failed to load mask: {e}")
                    mask_arr = None

                sequences = [
                    (ses_dir / "dwi", f"{sub_id}_{ses_id}_dwi", True, "dwi"),
                    (ses_dir / "dwi", f"{sub_id}_{ses_id}_adc", True, "adc"),
                    (ses_dir / "anat", f"{sub_id}_{ses_id}_FLAIR", False, "flair"),
                ]
                for seq_folder, seq_stem, uses_mask, seq_name in sequences:
                    try:
                        seq_img = _load_nifti_flexible(seq_folder, seq_stem)
                        if seq_img is None:
                            continue
                        vol = seq_img.get_fdata(dtype=np.float32)
                    except Exception as e:
                        print(f"    [WARN] {sub_id}/{ses_id}/{seq_name}: failed to load volume: {e}")
                        continue

                    slice_mask = None
                    if uses_mask and mask_arr is not None and mask_arr.shape == vol.shape:
                        slice_mask = mask_arr

                    for i in range(vol.shape[2]):
                        sl = vol[:, :, i].copy()
                        if slice_mask is not None:
                            label = classify_by_mask_array(slice_mask[:, :, i])
                        else:
                            label = "other"
                        source_key = f"{sub_id}/{ses_id}/{seq_name}/slice{i:04d}"
                        yield source_key, sl, label, "MRI"

                    del vol
                del mask_arr
        except Exception as e:
            print(f"    [ERROR] {sub_id}: unhandled exception, skipping subject: {e}")

        if sub_idx % 20 == 0:
            gc.collect()
            print(f"    scan_isles2022: {sub_idx + 1}/{len(sub_dirs)} subjects done")


DATASET_SCANNERS = {
    "Kaggle_CT": scan_kaggle_ct,
    "CPAISD": scan_cpaisd,
    "MRI_Images": scan_mri_images,
    "SICH_CT": scan_sich_ct,
    "Tekno21": scan_tekno21,
    "Multimodal_Stroke": scan_multimodal_stroke,
    "ISLES2022": scan_isles2022,
}

# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Per-image percentile windowing -> 0-255 uint8. A generic, modality-
    agnostic normalization (handles CT's wide HU range and MRI's arbitrary
    scanner units the same way); refine with modality-specific windowing
    later if needed (see KAGGLE_WORKFLOW.md)."""
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def load_done_keys() -> set:
    done = set()
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, newline="") as f:
            for row in csv.DictReader(f):
                done.add(row["source_key"])
    return done


def ensure_manifest_header():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=MANIFEST_FIELDS).writeheader()


def process_one(dataset_name, source_key, item, label, modality, writer, fh):
    arr = item if isinstance(item, np.ndarray) else read_image_as_array(item)
    if arr is None:
        return False

    if arr.ndim == 3:
        arr = arr.mean(axis=2)  # collapse to single-channel grayscale
    orig_h, orig_w = arr.shape[0], arr.shape[1]

    out_name = hashlib.md5(source_key.encode()).hexdigest() + ".png"
    out_path = OUT_ROOT / dataset_name / label / out_name

    if not DIAGNOSTIC_MODE:
        img8 = normalize_to_uint8(arr)
        img = Image.fromarray(img8).resize(TARGET_SIZE, Image.BILINEAR)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Deterministic filename from source_key: a rerun overwrites
        # identically rather than piling up duplicates, and is
        # content-addressed so it never collides across datasets.
        img.save(out_path)

    writer.writerow({
        "source_key": source_key,
        "dataset": dataset_name,
        "modality": modality,
        "label": label,
        "out_path": str(out_path.relative_to(OUT_ROOT)),
        "orig_height": orig_h,
        "orig_width": orig_w,
    })
    fh.flush()
    return True


def main():
    for dataset_name, base in DATASETS.items():
        print(f"  {dataset_name}: {'FOUND' if base.exists() else 'MISSING'} at {base}")

    ensure_manifest_header()
    done = load_done_keys()
    print(f"Resuming: {len(done)} images already in manifest.csv from a previous run.")

    with open(MANIFEST_PATH, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)

        for dataset_name, base in DATASETS.items():
            if not base.exists():
                print(f"[SKIP] {dataset_name}: input not attached at {base} "
                      f"-- add it under Notebook > Add Input")
                continue

            scanner = DATASET_SCANNERS[dataset_name]
            new_count = 0
            skipped_count = 0
            try:
                for source_key, item, label, modality in scanner(base):
                    if source_key in done:
                        skipped_count += 1
                        continue
                    if process_one(dataset_name, source_key, item, label, modality, writer, fh):
                        done.add(source_key)
                        new_count += 1
                        if new_count % PROGRESS_EVERY == 0:
                            print(f"  {dataset_name}: {new_count} new, {skipped_count} already-done skipped")
            except Exception:
                import traceback
                print(f"[ERROR] {dataset_name}: scanner raised an exception after "
                      f"{new_count} new images -- see traceback below. Manifest rows "
                      f"written so far for this dataset are preserved (flushed per-row); "
                      f"rerunning will resume from here, not from zero.")
                traceback.print_exc()

            print(f"[DONE] {dataset_name}: {new_count} new images processed, "
                  f"{skipped_count} already done (resumed)")

    total_rows = len(load_done_keys())
    print(f"\nFinished. manifest.csv now has {total_rows} rows total.")

    # Zip the preprocessed images into one file, THEN DELETE the individual
    # files. `kaggle kernels output` effectively hangs trying to enumerate
    # 100K+ individual output files -- and merely adding a zip alongside
    # them doesn't fix that, since the individual files are still there to
    # enumerate (confirmed: a pull hung with no progress for 10+ minutes
    # against output containing both). Only removing the originals after
    # zipping actually makes the kernel's final output small enough to pull
    # normally -- see KAGGLE_WORKFLOW.md Phase B Sec 3.4 / Phase D notes.
    if not DIAGNOSTIC_MODE and OUT_ROOT.exists():
        import shutil
        zip_base = OUT_ROOT.parent / "preprocessed_images"
        print(f"\nZipping {OUT_ROOT} -> {zip_base}.zip ...")
        shutil.make_archive(str(zip_base), "zip", root_dir=str(OUT_ROOT))
        zip_path = Path(f"{zip_base}.zip")
        print(f"Done: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
        print(f"Removing {OUT_ROOT} now that its contents are zipped ...")
        shutil.rmtree(OUT_ROOT)
        print("Removed.")


if __name__ == "__main__":
    main()
