"""
NeuroScan Phase Q -- SOOP (OpenNeuro ds004889, CC0) slice preparation. CPU, no training.

Purpose: add SOOP stroke MRI to TRAINING while keeping ISLES2022 in the test
set unchanged, and hold back a patient-level portion of SOOP as an untouched
external MRI test set.

Preprocessing is IDENTICAL to how ISLES2022 was prepared in Phase B
(kaggle_kernels/phase_b_preprocessing/preprocess.py):
  * each axial slice vol[:, :, i] of the TRACE (DWI) and ADC volumes,
    no rotation;
  * per-slice percentile windowing 0.5-99.5 % -> uint8 (normalize_to_uint8),
    which also absorbs SOOP's scanner-dependent ADC scale;
  * bilinear resize to 224x224;
  * label = ischemic if the ACUTE lesion mask has any voxel on that slice,
    else normal.
SOOP-specific rules (SOOP has separate acute and chronic masks):
  * slices with chronic-lesion voxels but no acute voxels are EXCLUDED
    (an old infarct is neither "normal" nor an acute ischemic lesion);
  * subjects without any acute-lesion voxel (259 "DWI-negative" subjects,
    156 of them clinically diagnosed stroke) are EXCLUDED until an expert
    confirms what a missing mask means;
  * near-blank slices are dropped (fraction of DWI pixels > 10 must be
    >= 0.10), the same blank rule used for the AISD/RSNA normals.

Patient-level split of the eligible subjects (seed 42, stratified by acute
lesion-volume tertile): 60 % train, 10 % val, 30 % EXTERNAL hold-out. The
hold-out is never used for training or any decision -- it is the external
MRI test. No SOOP subject goes into the internal test set, so the internal
test stays identical to Rounds 19-21 and stays comparable.

Outputs (small, no loose images):
  soop_rows.csv        split-v3-style rows + split + lesion_px (224 canvas)
  soop_images.zip      SOOP/{ischemic|normal}/{md5(source_key)}.png  (DWI and ADC)
  soop_prep_report.json
All paths are direct: data is read from OpenNeuro's public S3 bucket over HTTPS.
"""
import hashlib
import io
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def ensure_pkg(import_name, pip_name=None):
    try:
        __import__(import_name)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name or import_name], check=True)


ensure_pkg("nibabel")
ensure_pkg("requests")

import nibabel as nib
import numpy as np
import pandas as pd
import requests
from PIL import Image

DS = "ds004889"
BUCKET = "https://s3.amazonaws.com/openneuro.org"
TMP = Path("/tmp/soop")
OUT = Path("/kaggle/working")
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
TARGET_SIZE = (224, 224)
SEED = 42
SPLIT_FRACS = (("train", 0.60), ("val", 0.10), ("external", 0.30))
BLANK_PIXEL_THRESHOLD = 10
MIN_FOREGROUND_FRACTION = 0.10
DATASET_NAME = "SOOP_dwi_adc"
T0 = time.time()
SESSION = requests.Session()


def log(m):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {m}", flush=True)


def fetch(rel, retries=5):
    url = f"{BUCKET}/{DS}/{rel}"
    for a in range(retries):
        try:
            r = SESSION.get(url, timeout=180)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            log(f"  retry {a + 1} {rel}: {e}")
            time.sleep(3 * (a + 1))
    return None


def list_all():
    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": f"{DS}/", "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = SESSION.get(BUCKET, params=params, timeout=120)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for c in root.findall("s3:Contents", NS):
            keys.append(c.find("s3:Key", NS).text[len(DS) + 1:])
        if root.findtext("s3:IsTruncated", namespaces=NS) == "true":
            token = root.findtext("s3:NextContinuationToken", namespaces=NS)
        else:
            return keys


def load_nii(rel):
    data = fetch(rel)
    if data is None:
        return None
    dest = TMP / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    a = np.asarray(nib.load(str(dest)).get_fdata(dtype=np.float32))
    dest.unlink()
    return a[..., 0] if a.ndim == 4 else a


def normalize_to_uint8(arr):
    """Verbatim from phase_b_preprocessing/preprocess.py (used for ISLES2022)."""
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def to_png(sl):
    img = Image.fromarray(normalize_to_uint8(sl)).resize(TARGET_SIZE, Image.BILINEAR)
    return img


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    report = {}
    keys = list_all()
    subj = {}
    for k in keys:
        if k.startswith("derivatives/lesion_masks/"):
            s = k.split("/")[2]
            if "desc-lesionAcute_mask" in k:
                subj.setdefault(s, {})["acute"] = k
            elif "desc-lesionChronic_mask" in k:
                subj.setdefault(s, {})["chronic"] = k
        elif k.startswith("sub-") and "/dwi/" in k and k.endswith(".nii.gz"):
            s = k.split("/")[0]
            if "rec-TRACE" in k:
                subj.setdefault(s, {})["trace"] = k
            elif "rec-ADC" in k:
                subj.setdefault(s, {})["adc"] = k
    complete = {s: v for s, v in subj.items() if "trace" in v and "adc" in v}
    log(f"Listed {len(keys)} keys; subjects with TRACE+ADC: {len(complete)}; with acute mask file: "
        f"{sum('acute' in v for v in complete.values())}")

    # --- acute lesion volume per subject (masks are tiny) -> eligibility + stratification
    def acute_voxels(item):
        s, v = item
        if "acute" not in v:
            return s, 0
        m = load_nii(v["acute"])
        return s, int((m > 0).sum()) if m is not None else 0
    with ThreadPoolExecutor(16) as ex:
        vox = dict(ex.map(acute_voxels, complete.items()))
    eligible = sorted(s for s, n in vox.items() if n > 0)
    report["subjects_with_trace_adc"] = len(complete)
    report["eligible_acute_lesion_subjects"] = len(eligible)
    report["excluded_no_acute_voxels"] = len(complete) - len(eligible)
    log(f"Eligible (acute voxels > 0): {len(eligible)}; excluded: {report['excluded_no_acute_voxels']}")

    # --- patient-level split, stratified by lesion-volume tertile
    rng = np.random.default_rng(SEED)
    vols = pd.Series({s: vox[s] for s in eligible})
    tert = pd.qcut(vols.rank(method="first"), 3, labels=False)
    assign = {}
    for t in range(3):
        members = list(rng.permutation(vols.index[tert == t]))
        n, start = len(members), 0
        for i, (name, frac) in enumerate(SPLIT_FRACS):
            end = n if i == len(SPLIT_FRACS) - 1 else start + int(round(frac * n))
            for s in members[start:end]:
                assign[s] = name
            start = end
    report["subjects_per_split"] = pd.Series(assign).value_counts().to_dict()
    log(f"Subject split: {report['subjects_per_split']}")

    # --- slice extraction, streamed straight into the zip (no loose files in /kaggle/working)
    rows, stats = [], {"slices_seen": 0, "chronic_only_excluded": 0, "blank_dropped": 0, "shape_mismatch_subjects": 0,
                       "load_failed_subjects": 0}

    def load_subject(s):
        v = complete[s]
        return s, load_nii(v["trace"]), load_nii(v["adc"]), load_nii(v["acute"]), (load_nii(v["chronic"]) if "chronic" in v else None)

    zpath = OUT / "soop_images.zip"
    def batched_loads(ex, batch=32):
        """Bounded prefetch: at most `batch` subjects' volumes are held in memory at once."""
        for b in range(0, len(eligible), batch):
            yield from ex.map(load_subject, eligible[b:b + batch])

    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as zf, ThreadPoolExecutor(8) as ex:
        for done, (s, dwi, adc, acute, chronic) in enumerate(batched_loads(ex), 1):
            if dwi is None or adc is None or acute is None:
                stats["load_failed_subjects"] += 1
                continue
            if not (dwi.shape == adc.shape == acute.shape) or (chronic is not None and chronic.shape != dwi.shape):
                stats["shape_mismatch_subjects"] += 1
                continue
            for i in range(dwi.shape[2]):
                stats["slices_seen"] += 1
                a_i = acute[:, :, i] > 0
                if chronic is not None and (chronic[:, :, i] > 0).any() and not a_i.any():
                    stats["chronic_only_excluded"] += 1
                    continue
                d_img = to_png(dwi[:, :, i])
                if (np.asarray(d_img) > BLANK_PIXEL_THRESHOLD).mean() < MIN_FOREGROUND_FRACTION:
                    stats["blank_dropped"] += 1
                    continue
                label = "ischemic" if a_i.any() else "normal"
                a_img = to_png(adc[:, :, i])
                paths = {}
                for seq, img in (("dwi", d_img), ("adc", a_img)):
                    sk = f"{s}/ses-1/{seq}/slice{i:04d}"
                    rel = f"SOOP/{label}/{hashlib.md5(sk.encode()).hexdigest()}.png"
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    zf.writestr(rel, buf.getvalue())
                    paths[seq] = rel
                lesion_px = int((np.asarray(Image.fromarray((a_i * 255).astype(np.uint8)).resize(TARGET_SIZE, Image.NEAREST)) > 0).sum())
                rows.append({"source_key": f"{s}/ses-1/dwi/slice{i:04d}", "dataset": DATASET_NAME, "modality": "MRI",
                             "label": label, "out_path": paths["dwi"], "prev_out_path": np.nan, "next_out_path": np.nan,
                             "has_context": False, "dwi_out_path": paths["dwi"], "adc_out_path": paths["adc"],
                             "split": assign[s], "subject": s, "slice": i, "lesion_px": lesion_px if label == "ischemic" else 0})
            if done % 100 == 0:
                log(f"  {done}/{len(eligible)} subjects, {len(rows)} rows")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "soop_rows.csv", index=False)
    report["stats"] = stats
    report["rows_by_split_label"] = df.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict()
    report["subjects_by_split_after_filters"] = df.groupby("split")["subject"].nunique().to_dict()
    report["ischemic_lesion_px_quantiles"] = df.loc[df.label == "ischemic", "lesion_px"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()
    report["zip_MB"] = round(zpath.stat().st_size / 1e6, 1)
    with open(OUT / "soop_prep_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"Rows by split/label: {report['rows_by_split_label']}")
    log(f"Stats: {stats}; zip {report['zip_MB']} MB")
    log("Done.")


if __name__ == "__main__":
    main()
