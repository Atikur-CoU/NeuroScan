"""
NeuroScan Phase Q -- AISD DICOM feasibility audit for R34 (multi-window CT). CPU, no model, no metrics.
TRAINING PATIENTS ONLY (Phase 2 new_train: split-v3 train AISD patients minus the val2 patients; no val2, no test).
Questions (decide whether R34 is feasible / worth running, before any GPU time):
  (1) Can the original AISD DICOMs (GitHub release, dicom-0..3.tar.gz) be matched slice-by-slice to the processed
      image.zip PNGs that our pipeline and the masks use? (series, slice order, orientation, shape; Pearson r)
  (2) Which window produced the PNGs? (least-squares fit grey = a*HU + b on unclipped brain pixels -> WC/WW, residual)
  (3) Lesion vs mirrored contralateral tissue in HU, per mask value 2 (clear) and 3 (blurred), per patient
      (Wilcoxon vs 0), next to the same contrast in PNG grey levels. If AISD lesions are only ~0-1 HU darker
      (as CPAISD), a narrow window has nothing to amplify and R34 should not be run.
  (4) Cost of a full re-render: archive sizes, download/extract/read times, disk.
Nothing here touches labels, splits or any model; outputs are tables and a JSON report.
"""
import json
import re
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path


def ensure_pkg(import_name, pip_name=None):
    try:
        __import__(import_name)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name or import_name], check=True)


for mod in ("pydicom", "gdown", "scipy"):
    ensure_pkg(mod)

import gdown
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from scipy import ndimage, stats

INPUT_ROOT = Path("/kaggle/input")
DS = INPUT_ROOT / "datasets/atik68"
SPLIT_DIR = DS / "neuroscan-phase-p-patient-split-v3"
PHASE2 = DS / "neuroscan-q-phase2-split-data"
IMAGE_ZIP_ID = "157f9aE3ZhRSdIuIbP2PRG8ub9JJWvMGk"   # same files as phase_k / aisd-normals prep
MASK_ZIP_ID = "1d08fFpEvK4D6YTKfRlNuv_OlIxigZxl6"
DICOM_TARS = {"dicom-0": "1WZdjvq0ZFPnNpbhfxadIYPcbWpdjgM-n", "dicom-1": "1CBVHFq8m3-FjG3-aCRfkUMzgHsfheTLd",
              "dicom-2": "1ZBK93GfTwpfJv5DlD2vhZnbO5JXpwrz", "dicom-3": "1mRKPp9hvMCGb5QVUuCG-ELL-zBiHLxun"}
MAX_PATIENTS = 150          # cap on audited training patients (random, seed 2026)
TMP = Path("/tmp/aisd_audit")
OUT = Path("/kaggle/working/output")
OUT.mkdir(parents=True, exist_ok=True)
ORIENTS = {"none": lambda a: a, "fliplr": np.fliplr, "flipud": np.flipud, "rot180": lambda a: a[::-1, ::-1]}
T0 = time.time()
REPORT = {"scope": "AISD training patients only (Phase 2 new_train); no val2, no test"}


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


def disk():
    u = shutil.disk_usage("/tmp")
    return f"/tmp free {u.free / 1e9:.1f} GB"


def download(file_id, dest):
    t0 = time.time()
    try:
        gdown.download(f"https://drive.google.com/uc?id={file_id}", str(dest), quiet=True)
    except Exception as e:  # noqa: BLE001
        log(f"[WARN] download of {dest.name} failed: {e!r}")
        return None
    if not dest.exists():
        log(f"[WARN] download of {dest.name} produced no file")
        return None
    log(f"downloaded {dest.name}: {dest.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.0f}s ({disk()})")
    return dest


def train_patients():
    tr = pd.read_csv(need(SPLIT_DIR / "train.csv"))
    tr = tr[tr["dataset"] == "AISD"]
    pids = set(tr["source_key"].astype(str).str.split("/").str[2])
    v2 = pd.read_csv(need(PHASE2 / "phase2_val2_patients.csv"))
    v2a = set(v2.loc[v2["dataset"] == "AISD", "patient_key"].str.split(":").str[1])
    others = set()
    for s in ("val", "test"):
        d = pd.read_csv(need(SPLIT_DIR / f"{s}.csv"))
        others |= set(d.loc[d["dataset"] == "AISD", "source_key"].astype(str).str.split("/").str[2])
    assert not (pids & others), "AISD patient in train and val/test of split v3"
    keep = sorted(pids - v2a)
    log(f"AISD split-v3 train patients {len(pids)}; minus val2 {len(pids & v2a)} -> Phase 2 training patients {len(keep)}")
    return keep


def largest_component(mask):
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return ndimage.binary_fill_holes(lab == (np.argmax(sizes) + 1))


def mirror_contrast(img, lesion, brain, lo, hi):
    """Mean(lesion) - mean(mirror across the brain's vertical centre line), pixel pairs inside [lo, hi] only."""
    lesion = lesion & brain
    if lesion.sum() < 5:
        return None
    ys, xs = np.nonzero(lesion)
    _, bx = ndimage.center_of_mass(brain)
    xm = np.round(2 * bx - xs).astype(int)
    ok = (xm >= 0) & (xm < img.shape[1])
    ys, xs, xm = ys[ok], xs[ok], xm[ok]
    ok = brain[ys, xm] & ~lesion[ys, xm]
    a, b = img[ys[ok], xs[ok]], img[ys[ok], xm[ok]]
    keep = (a >= lo) & (a <= hi) & (b >= lo) & (b <= hi)
    if keep.sum() < 5:
        return None
    return float(a[keep].mean() - b[keep].mean()), int(keep.sum()), float(b[keep].std())


def to_hu(ds):
    return ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1) or 1) + \
        float(getattr(ds, "RescaleIntercept", 0) or 0)


def resize_to(a, shape):
    if a.shape == shape:
        return a
    return np.asarray(Image.fromarray(a.astype(np.float32)).resize(shape[::-1], Image.BILINEAR))


def corr(hu, png):
    m = (png > 5) & (png < 250)
    if m.sum() < 500:
        return np.nan
    return float(np.corrcoef(np.clip(hu[m], -100, 200), png[m])[0, 1])


def audit_patient(pid, files, pngs, masks):
    """files: DICOM paths of this patient; pngs: {slice_no: path}; masks: {slice_no: path}."""
    rec = {"patient": pid, "n_dicom_files": len(files), "n_png": len(pngs)}
    hdr = []
    for f in files:
        try:
            d = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        except Exception:  # noqa: BLE001
            continue
        if not hasattr(d, "Rows"):
            continue
        ipp = getattr(d, "ImagePositionPatient", None)
        hdr.append({"f": f, "folder": f.name.split("__")[0] if "__" in f.name else "?",
                    "modality": str(getattr(d, "Modality", "?")), "series": str(getattr(d, "SeriesInstanceUID", "?")), "inst": int(getattr(d, "InstanceNumber", 0) or 0),
                    "z": float(ipp[2]) if ipp is not None else np.nan, "rows": int(d.Rows), "cols": int(d.Columns),
                    "desc": str(getattr(d, "SeriesDescription", "")), "kernel": str(getattr(d, "ConvolutionKernel", "")),
                    "thick": float(getattr(d, "SliceThickness", np.nan) or np.nan),
                    "spacing": float(getattr(d, "PixelSpacing", [np.nan])[0]),
                    "slope": float(getattr(d, "RescaleSlope", np.nan) or np.nan),
                    "intercept": float(getattr(d, "RescaleIntercept", np.nan) or np.nan),
                    "wc": str(getattr(d, "WindowCenter", "")), "ww": str(getattr(d, "WindowWidth", ""))})
    if not hdr:
        rec["error"] = "no readable DICOM image"
        return rec, []
    H = pd.DataFrame(hdr)
    rec["folders_modalities"] = H.groupby(["folder", "modality"]).size().to_dict()
    H = H[H["modality"] == "CT"]
    if H.empty:
        rec["error"] = "no CT-modality DICOM"
        return rec, []
    ser = H.groupby("series").size().sort_values(ascending=False)
    rec.update({"n_series": int(len(ser)), "series_sizes": ser.tolist(), "rows_cols": f"{H.rows.iloc[0]}x{H.cols.iloc[0]}",
                "slice_thickness": float(H.thick.median()), "pixel_spacing": float(H.spacing.median()),
                "rescale_slope": float(H.slope.median()), "rescale_intercept": float(H.intercept.median()),
                "tag_window": f"{H.wc.iloc[0]}/{H.ww.iloc[0]}", "series_desc": H.desc.iloc[0], "kernel": H.kernel.iloc[0]})
    nums = sorted(pngs)
    png_arr = {n: np.asarray(Image.open(pngs[n]).convert("L"), np.float32) for n in nums}
    rec["png_shape"] = "x".join(map(str, png_arr[nums[0]].shape))
    mid = nums[len(nums) // 2]
    # choose series / order / orientation on the middle PNG slice, then verify on every slice
    best = None
    for sid in ser.index[:3]:
        S = H[H.series == sid]
        for order in ("z_asc", "z_desc", "inst_asc", "inst_desc"):
            key, asc = ("z" if order.startswith("z") else "inst"), order.endswith("asc")
            if key == "z" and S.z.isna().any():
                continue
            seq = S.sort_values(key, ascending=asc)["f"].tolist()
            if len(seq) == len(nums):
                idx = {n: i for i, n in enumerate(nums)}          # same count: position in sorted order
            elif max(nums) < len(seq):
                idx = {n: n for n in nums}                         # PNG slice number = position in the series
            else:
                continue  # slice-count mismatch that cannot be mapped: reported, not guessed
            d = pydicom.dcmread(str(seq[idx[mid]]), force=True)
            hu = to_hu(d)
            for oname, fn in ORIENTS.items():
                r = corr(resize_to(np.ascontiguousarray(fn(hu)), png_arr[mid].shape), png_arr[mid])
                if best is None or (r == r and r > best[0]):
                    best = (r, sid, order, oname, seq, idx)
    if best is None:
        rec["error"] = f"no series with the PNG slice count ({len(nums)}); series sizes {ser.tolist()}"
        return rec, []
    r_mid, sid, order, oname, seq, idx = best
    rec.update({"chosen_series_size": len(seq), "order": order, "orientation": oname, "r_middle_slice": r_mid,
                "mapping": "by_rank" if len(seq) == len(nums) else "by_slice_number"})
    fn = ORIENTS[oname]
    rows, slices = [], []
    for i, n in enumerate(nums):
        hu = resize_to(np.ascontiguousarray(fn(to_hu(pydicom.dcmread(str(seq[idx[n]]), force=True)))), png_arr[n].shape)
        g = png_arr[n]
        r = corr(hu, g)
        m = (g > 5) & (g < 250) & (hu > -50) & (hu < 150)
        fit = None
        if m.sum() > 2000:
            a, b = np.polyfit(hu[m], g[m], 1)
            resid = float(np.sqrt(np.mean((g[m] - (a * hu[m] + b)) ** 2)))
            fit = (float(a), float(b), resid)
        s = {"patient": pid, "slice": n, "r": r}
        if fit:
            s.update({"grey_per_hu": fit[0], "ww": 255 / fit[0] if fit[0] > 0 else np.nan,
                      "wc": (127.5 - fit[1]) / fit[0] if fit[0] > 0 else np.nan, "fit_rmse_grey": fit[2]})
        if n in masks:
            mk = np.asarray(Image.open(masks[n]))
            mk = mk[..., 0] if mk.ndim == 3 else mk
            if mk.shape != g.shape:
                mk = np.asarray(Image.fromarray(mk.astype(np.uint8)).resize(g.shape[::-1], Image.NEAREST))
            brain_hu = largest_component((hu > -15) & (hu < 100))
            brain_g = largest_component((g > 15) & (g < 245))
            for v in (2, 3):
                les = mk == v
                if les.sum() >= 20:
                    ch = mirror_contrast(hu, les, brain_hu, 0, 80)
                    cg = mirror_contrast(g, les, brain_g, 15, 245)
                    if ch:
                        s[f"dhu_v{v}"], s[f"npairs_v{v}"], s[f"sd_mirror_hu_v{v}"] = ch
                    if cg:
                        s[f"dgrey_v{v}"] = cg[0]
        slices.append(s)
    S = pd.DataFrame(slices)
    rec.update({"r_median_all_slices": float(S["r"].median()), "r_min_all_slices": float(S["r"].min()),
                "fit_wc_median": float(S["wc"].median()) if "wc" in S else np.nan,
                "fit_ww_median": float(S["ww"].median()) if "ww" in S else np.nan,
                "fit_rmse_grey_median": float(S["fit_rmse_grey"].median()) if "fit_rmse_grey" in S else np.nan})
    return rec, slices


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    log(disk())
    pids = train_patients()
    rng = np.random.default_rng(2026)
    sel = set(rng.choice(pids, size=min(MAX_PATIENTS, len(pids)), replace=False).tolist())
    REPORT["patients_training"] = len(pids)
    REPORT["patients_selected"] = len(sel)

    # processed images + masks (as used by our pipeline)
    t0 = time.time()
    pngs, masks = {}, {}
    for name, fid in (("image", IMAGE_ZIP_ID), ("mask", MASK_ZIP_ID)):
        z = download(fid, TMP / f"{name}.zip")
        if z is None:
            print(f"[ERROR] cannot download {name}.zip")
            sys.exit(1)
        with zipfile.ZipFile(z) as zf:
            for info in zf.infolist():
                p = Path(info.filename)
                if p.suffix.lower() == ".png" and p.parent.name in sel and p.stem.isdigit():
                    zf.extract(info, TMP / name)
                    (pngs if name == "image" else masks).setdefault(p.parent.name, {})[int(p.stem)] = TMP / name / info.filename
        z.unlink()
    log(f"processed PNGs for {len(pngs)} selected patients, masks for {len(masks)} ({time.time() - t0:.0f}s)")

    # DICOM archives: stream each tar, extract only the selected training patients, delete the archive
    dcm_files, tar_info = {}, {}
    for name, fid in DICOM_TARS.items():
        t0 = time.time()
        tp = download(fid, TMP / f"{name}.tar.gz")
        for wait in (120, 300):
            if tp is not None:
                break
            log(f"retrying {name} in {wait}s (Google Drive quota)")
            time.sleep(wait)
            tp = download(fid, TMP / f"{name}.tar.gz")
        if tp is None:
            tar_info[name] = {"error": "download failed"}
            continue
        size = tp.stat().st_size
        n_mem, n_kept, first, pid_set, folder_counts = 0, 0, [], set(), {}
        try:
            tf_ctx = tarfile.open(tp, "r|*")
        except tarfile.ReadError as e:
            with open(tp, "rb") as fh:
                magic = fh.read(8)
            tar_info[name] = {"error": f"not a tar archive ({e}); first bytes {magic!r}", "gb": round(size / 1e9, 3)}
            log(f"[WARN] {name}: {tar_info[name]}")
            tp.unlink()
            continue
        with tf_ctx as tf:
            for m in tf:
                if not m.isfile():
                    continue
                n_mem += 1
                if len(first) < 8:
                    first.append(m.name)
                parts = Path(m.name).parts
                hit = next((p for p in parts if p in sel), None)
                cand = next((p for p in parts if re.fullmatch(r"\d{7}", p)), None)
                if cand:
                    pid_set.add(cand)
                if hit:
                    sub = parts[parts.index(hit) + 1] if parts.index(hit) + 1 < len(parts) - 1 else "root"
                    dest = TMP / "dcm" / hit / f"{sub}__{n_kept:06d}_{Path(m.name).name}"
                    folder_counts[sub] = folder_counts.get(sub, 0) + 1
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with tf.extractfile(m) as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    dcm_files.setdefault(hit, []).append(dest)
                    n_kept += 1
        tp.unlink()
        tar_info[name] = {"gb": round(size / 1e9, 2), "members": n_mem, "kept_files": n_kept,
                          "patients_in_archive": len(pid_set), "first_members": first,
                          "subfolders_kept": folder_counts, "seconds": round(time.time() - t0)}
        log(f"{name}: {tar_info[name]} ({disk()})")
    REPORT["archives"] = tar_info
    log(f"DICOM found for {len(dcm_files)}/{len(sel)} selected training patients")

    recs, slices = [], []
    t0 = time.time()
    for i, pid in enumerate(sorted(sel)):
        if pid not in dcm_files or pid not in pngs:
            recs.append({"patient": pid, "error": "no DICOM" if pid not in dcm_files else "no PNG"})
            continue
        try:
            r, s = audit_patient(pid, dcm_files[pid], pngs[pid], masks.get(pid, {}))
        except Exception as e:  # noqa: BLE001
            r, s = {"patient": pid, "error": repr(e)[:200]}, []
        recs.append(r)
        slices += s
        if (i + 1) % 25 == 0:
            log(f"  audited {i + 1}/{len(sel)} ({time.time() - t0:.0f}s)")
    P, S = pd.DataFrame(recs), pd.DataFrame(slices)
    fm = {}
    for d in P.get("folders_modalities", pd.Series(dtype=object)).dropna():
        for k, v in d.items():
            fm[str(k)] = fm.get(str(k), 0) + v
    REPORT["dicom_files_by_folder_modality"] = fm
    log(f"DICOM files by (subfolder, modality): {fm}")
    P.to_csv(OUT / "aisd_dicom_patients.csv", index=False)
    S.to_csv(OUT / "aisd_dicom_slices.csv", index=False)
    REPORT["seconds_per_patient_read_render"] = round((time.time() - t0) / max(len(sel), 1), 2)

    ok = P[P.get("error").isna()] if "error" in P else P
    REPORT["errors"] = P["error"].value_counts().to_dict() if "error" in P else {}
    log(f"errors: {REPORT['errors']}")
    if len(ok):
        REPORT["alignment"] = {
            "patients_aligned": int(len(ok)),
            "r_median_all_slices": {"median": float(ok.r_median_all_slices.median()),
                                    "p5": float(ok.r_median_all_slices.quantile(0.05))},
            "patients_min_slice_r_below_0.95": int((ok.r_min_all_slices < 0.95).sum()),
            "order": ok.order.value_counts().to_dict(), "orientation": ok.orientation.value_counts().to_dict(),
            "n_series": ok.n_series.value_counts().to_dict(), "rows_cols": ok.rows_cols.value_counts().to_dict(),
            "png_shape": ok.png_shape.value_counts().to_dict(),
            "slice_thickness": ok.slice_thickness.value_counts().to_dict(),
            "rescale": {f"{k[0]}/{k[1]}": int(v) for k, v in ok.groupby(["rescale_slope", "rescale_intercept"]).size().items()},
            "tag_window": ok.tag_window.value_counts().head(8).to_dict()}
        REPORT["png_window_fit"] = {
            "wc_median": float(ok.fit_wc_median.median()), "wc_iqr": [float(ok.fit_wc_median.quantile(q)) for q in (.25, .75)],
            "ww_median": float(ok.fit_ww_median.median()), "ww_iqr": [float(ok.fit_ww_median.quantile(q)) for q in (.25, .75)],
            "fit_rmse_grey_median": float(ok.fit_rmse_grey_median.median()),
            "grey_levels_per_hu_median": float(S["grey_per_hu"].median()) if "grey_per_hu" in S else None}
        log(f"ALIGNMENT: {REPORT['alignment']}")
        log(f"PNG WINDOW FIT: {REPORT['png_window_fit']}")
    REPORT["lesion_vs_mirror"] = {}
    for v in (2, 3):
        c = f"dhu_v{v}"
        if c not in S or S[c].notna().sum() == 0:
            continue
        w = S.dropna(subset=[c]).copy()
        w["wt"] = w[f"npairs_v{v}"]
        per = w.groupby("patient").apply(lambda g: pd.Series({
            "dhu": np.average(g[c], weights=g["wt"]),
            "dgrey": float(g[f"dgrey_v{v}"].mean()) if f"dgrey_v{v}" in g else np.nan,
            "sd_mirror": float(g[f"sd_mirror_hu_v{v}"].median()), "slices": len(g)}))
        per["cnr"] = per["dhu"] / per["sd_mirror"]
        wil = stats.wilcoxon(per["dhu"]) if len(per) >= 6 else None
        REPORT["lesion_vs_mirror"][f"value_{v}"] = {
            "patients": int(len(per)), "slices": int(per["slices"].sum()),
            "dHU_median": float(per["dhu"].median()), "dHU_iqr": [float(per["dhu"].quantile(q)) for q in (.25, .75)],
            "wilcoxon_p": float(wil.pvalue) if wil else None,
            "patients_below_minus2HU": int((per["dhu"] < -2).sum()), "patients_above_plus2HU": int((per["dhu"] > 2).sum()),
            "cnr_median": float(per["cnr"].median()),
            "dgrey_median_current_png": float(per["dgrey"].median())}
        per.to_csv(OUT / f"aisd_lesion_mirror_value{v}.csv")
        log(f"LESION vs MIRROR value {v}: {REPORT['lesion_vs_mirror'][f'value_{v}']}")
    REPORT["note"] = ("value 2 = clear acute infarct, 3 = blurred acute infarct (the only AISD values our pipeline labels "
                      "ischemic). dHU < 0 = lesion darker than contralateral tissue. CPAISD reference (CT input audit): "
                      "core -0.78 HU, penumbra +0.08 HU.")
    with open(OUT / "aisd_dicom_audit.json", "w") as f:
        json.dump(REPORT, f, indent=2, default=str)
    shutil.rmtree(TMP, ignore_errors=True)
    log("Done.")


if __name__ == "__main__":
    main()
