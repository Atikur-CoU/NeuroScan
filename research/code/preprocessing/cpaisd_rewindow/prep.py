"""
NeuroScan Multi-Dataset -- Phase N: re-preprocess CPAISD with proper
Hounsfield-Unit rescaling + a clinical brain window, replacing the
generic percentile-clip normalization every other phase has used
(runs on Kaggle, CPU-only).

Motivation (PROCESS.md Sec 32): Phase B's normalize_to_uint8() applies
a blind 0.5-99.5 percentile clip directly on RAW DICOM pixel values.
CPAISD is the only dataset in this project that supplies raw DICOM
(everything else is pre-windowed PNG/JPG or MRI NIfTI, where this
issue doesn't apply the same way). Raw CT pixel values span roughly
0-2700 (encoding HU -1024 to +1650ish, i.e. air through bone/
calcification), so a global percentile clip on that full range
compresses brain-tissue contrast -- where subtle ischemic changes
live -- into a tiny sliver of the output range. Measured directly on
3 lesion-positive CPAISD slices: brain-tissue contrast std is 0.005
under the current pipeline vs. 0.15 (30x more) once RescaleSlope/
RescaleIntercept are applied and the DICOM's own tagged brain window
(WindowCenter=35, WindowWidth=80 -- the standard clinical brain
window, also adequate for early ischemic change visibility) is used
instead.

Reprocesses ALL of CPAISD, both normal and stroke slices, not stroke
only -- fixing contrast on just the positive class would let the
model shortcut on "flat vs. textured" rather than learn real anatomy,
a shortcut-learning risk worth avoiding deliberately.

Output goes into a single leaf folder literally named
"cpaisd_v2_images" (not the "ischemic"/"normal"/"stroke" names used
elsewhere), so it gets its own dedicated supplementary-index lookup in
the next training kernel rather than colliding with the main corpus's
"normal" folders or the existing "ischemic"/"hemorrhagic"
supplementary index from Phase L/M.

Inputs (dataset_sources):
  - atik68/neuroscan-cpaisd (the original raw DICOM + mask.npz source)
"""
import csv
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np


def ensure_pkg(import_name, pip_name=None):
    try:
        __import__(import_name)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name or import_name], check=True)


ensure_pkg("pydicom")

import pydicom
from PIL import Image

INPUT_ROOT = Path("/kaggle/input")
OUT_ROOT = Path("/kaggle/working/cpaisd_v2_images")
MANIFEST_PATH = Path("/kaggle/working/cpaisd_v2_manifest.csv")
TARGET_SIZE = (224, 224)
DEFAULT_WC, DEFAULT_WW = 35.0, 80.0  # standard clinical brain window; used as fallback


def find_cpaisd_root():
    # No need to resolve an exact "base" directory (Kaggle's mount nesting
    # varies) -- rglob from /kaggle/input directly finds every raw.dcm
    # under however CPAISD ends up mounted.
    for marker in INPUT_ROOT.rglob("raw.dcm"):
        return INPUT_ROOT
    return None


def classify_by_mask(mask_path: Path) -> str:
    if not mask_path.exists():
        return "other"
    try:
        with np.load(mask_path) as d:
            arr = d["mask"] if "mask" in d.files else d[d.files[0]]
        return "stroke" if np.count_nonzero(arr) > 0 else "normal"
    except Exception:
        return "other"


def first_tag_value(tag_value, default):
    if tag_value is None:
        return default
    try:
        # pydicom MultiValue -- take the first entry (the primary/recommended window)
        return float(tag_value[0])
    except (TypeError, IndexError):
        try:
            return float(tag_value)
        except (TypeError, ValueError):
            return default


def dicom_to_windowed_uint8(ds) -> np.ndarray:
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    hu = arr * slope + intercept

    wc = first_tag_value(getattr(ds, "WindowCenter", None), DEFAULT_WC)
    ww = first_tag_value(getattr(ds, "WindowWidth", None), DEFAULT_WW)
    lo, hi = wc - ww / 2, wc + ww / 2

    windowed = np.clip((hu - lo) / (hi - lo), 0.0, 1.0)
    return (windowed * 255).astype(np.uint8)


def main():
    root = find_cpaisd_root()
    if root is None:
        print("[ERROR] Could not locate CPAISD raw.dcm files -- check dataset_sources.")
        sys.exit(1)
    print(f"CPAISD root: {root}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    fh = open(MANIFEST_PATH, "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=["source_key", "label", "out_filename"])
    writer.writeheader()

    n_total, n_stroke, n_normal, n_failed = 0, 0, 0, 0
    for raw_path in sorted(root.rglob("raw.dcm")):
        source_key = str(raw_path)
        label = classify_by_mask(raw_path.parent / "mask.npz")
        if label == "other":
            continue

        try:
            ds = pydicom.dcmread(str(raw_path))
            img8 = dicom_to_windowed_uint8(ds)
        except Exception as e:
            n_failed += 1
            print(f"  [WARN] failed to process {raw_path}: {e}")
            continue

        out_name = hashlib.md5(source_key.encode()).hexdigest() + ".png"
        img = Image.fromarray(img8).resize(TARGET_SIZE, Image.BILINEAR)
        img.save(OUT_ROOT / out_name)

        writer.writerow({"source_key": source_key, "label": label, "out_filename": out_name})
        fh.flush()

        n_total += 1
        if label == "stroke":
            n_stroke += 1
        else:
            n_normal += 1
        if n_total % 2000 == 0:
            print(f"  {n_total} done ({n_stroke} stroke, {n_normal} normal, {n_failed} failed)")

    fh.close()
    print(f"\nFinished: {n_total} images ({n_stroke} stroke, {n_normal} normal), {n_failed} failed.")

    import shutil
    zip_base = Path("/kaggle/working/cpaisd_v2_images")
    print(f"\nZipping {OUT_ROOT} -> {zip_base}.zip ...")
    shutil.make_archive(str(zip_base), "zip", root_dir=str(OUT_ROOT))
    print(f"Done: {zip_base}.zip ({(zip_base.with_suffix('.zip')).stat().st_size / 1e6:.1f} MB)")
    shutil.rmtree(OUT_ROOT)
    print("Removed uncompressed originals.")


if __name__ == "__main__":
    main()
