"""
NeuroScan paper 2 -- K8 maps vs expert lesion masks (JTCSST second revision, comment 6). CPU only; no training and
no model inference.

Why: the paper compares each compressed copy's Grad-CAM map with FP32's map; the reviewer asks how the maps relate to
expert lesion annotations.

What: every held-out ischemic slice with an expert lesion mask (AISD 569, CPAISD 299, ISLES 2022 635 = 1,503 slices,
all in the 10,602-slice held-out set) is scored for FP32 and the eight compressed copies, from the maps those runs
saved (identity view, ischemic class, 7 x 7 Grad-CAM computed inside the ONNX graph):
  FP32, INT8-static (min-max): neuroscan-p2-compression-gradcam partial/*.npz
  INT8-dynamic as served (one request per slice): neuroscan-p2-int8dyn-served INT8-dyn_served.npz
  percentile, entropy, first layers in FP32: neuroscan-p2-static-calibration; M1-M3: neuroscan-p2-mixed-precision.
Each file is checked to be the right copy (row order and predicted-class counts of the saved run).
Masks and their orientation as in the earlier Grad-CAM analysis of this model (same functions): AISD masks (classes
2 and 3) from the AISD release zip, CPAISD mask.npz next to each slice, ISLES 2022 derivative masks; identity
orientation for all three (checked again here: lesion share inside the head and lesion contrast are logged).
Measures (a) on the raw map (max-normalised, bilinear upsampling to 224 as in torch, restricted to the head = pixels
> 10): pointing hit (peak inside the mask; exact and within 16 px), chance (mask share of the head), energy inside the
mask and its lift over chance; (b) on the displayed map (the server's rule: 70th-percentile suppression, bicubic,
region > 0.6): region size, share of the region inside the mask, share of the mask covered, region touches the mask.
Per copy and dataset: means with patient-cluster bootstrap intervals (2,000, seed 0), paired differences from FP32,
slices whose pointing hit changes; and by whether the copy predicted the slice ischemic.
Check: FP32 means must reproduce the earlier analysis of this model (dataset means of pointing hit, chance, energy,
displayed-region size, precision and recall), logged side by side.
"""
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

T0 = time.time()


def log(m):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {m}", flush=True)


t = time.time()
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nibabel", "gdown"], capture_output=True, text=True)
log(f"pip nibabel gdown: exit {r.returncode} ({time.time() - t:.0f}s) {r.stderr[-300:]}")
if r.returncode:
    sys.exit(1)

import gdown  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

CLS = ("normal", "ischemic", "hemorrhagic")
TARGET = (224, 224)
N_BOOT, SEED = 2000, 0
LESION_DS = ("AISD", "CPAISD_v2", "ISLES2022_dwi_adc")
OUT = Path("/kaggle/working/output")
OUT.mkdir(parents=True, exist_ok=True)
TMP = Path("/tmp/k8")
TMP.mkdir(exist_ok=True)
R = {"note": __doc__.strip().splitlines()[0]}


def save():
    (OUT / "maps_vs_masks_results.json").write_text(json.dumps(R, indent=2, default=float))


def first(*cands, what=""):
    """Mount-path fallback: exists() on each candidate, never a listing."""
    for c in cands:
        if Path(c).exists():
            return Path(c)
    print(f"[ERROR] {what} not found; tried {[str(c) for c in cands]}")
    sys.exit(1)


def ds_root(slug):
    return first(f"/kaggle/input/datasets/atik68/{slug}", f"/kaggle/input/{slug}", what=f"dataset {slug}")


def nb_file(slug, name):
    roots = (f"/kaggle/input/notebooks/atik68/{slug}", f"/kaggle/input/{slug}")
    return first(*(f"{r}{sub}/{name}" for r in roots for sub in ("", "/output")), what=f"{slug}/{name}")


# ------------------------------------------------------------------------------------------------- inputs
t = time.time()
SPLIT_DIR = ds_root("neuroscan-phase-p-patient-split-v3")
PHASE_B = ds_root("neuroscan-phase-b-images")
AISD_K = ds_root("neuroscan-phase-k-aisd-images")
CPAISD_V2_IMG = ds_root("neuroscan-phase-n-cpaisd-v2-images")
CPAISD_RAW = ds_root("neuroscan-cpaisd")
ISLES_RAW = ds_root("neuroscan-isles2022")
CG, SV = "neuroscan-p2-compression-gradcam", "neuroscan-p2-int8dyn-served"
SC, MP = "neuroscan-p2-static-calibration", "neuroscan-p2-mixed-precision"
FILES = {  # copy: (saved file, key in the per-row files, predicted-class counts of that run over the 10,602 rows)
    "FP32": (nb_file(CG, "partial/FP32.npz"), [6613, 2401, 1588]),
    "INT8-dynamic": (nb_file(SV, "INT8-dyn_served.npz"), [6641, 2378, 1583]),
    "INT8-static (min-max)": (nb_file(CG, "partial/INT8-static.npz"), [6511, 2455, 1636]),
    "INT8-static, percentile": (nb_file(SC, "S1-percentile.npz"), [6515, 2436, 1651]),
    "INT8-static, entropy": (nb_file(SC, "S2-entropy.npz"), [6501, 2466, 1635]),
    "INT8-static, first layers in FP32": (nb_file(SC, "S3-minmax-first-fp32.npz"), [6679, 2331, 1592]),
    "M1: lowest 10% in FP32": (nb_file(MP, "M1-sens10.npz"), [6588, 2403, 1611]),
    "M2: lowest 25% in FP32": (nb_file(MP, "M2-sens25.npz"), [6446, 2503, 1653]),
    "M3: M1 and strided convolutions": (nb_file(MP, "M3-sens10-first.npz"), [6740, 2296, 1566]),
}
PER_ROW = pd.read_csv(nb_file(CG, "e2_per_row_probs.csv"))
assert len(PER_ROW) == 10602
PROBS, CAMS = {}, {}
for k, (p, counts) in FILES.items():
    z = np.load(p)
    pr, cm = z["probs"], z["cams"]
    assert pr.shape == (10602, 3) and cm.shape == (10602, 3, 7, 7), (k, pr.shape, cm.shape)
    if "rows" in z.files:
        assert (z["rows"] == np.arange(10602)).all(), f"{k}: row order"
    got = np.bincount(pr.argmax(1), minlength=3).tolist()
    if got != counts:
        print(f"[ERROR] {k}: predicted-class counts {got} != saved run {counts}; wrong file")
        sys.exit(1)
    PROBS[k], CAMS[k] = pr, cm
fp = PER_ROW[[f"p_{c}_FP32" for c in CLS]].to_numpy().argmax(1)
assert (fp == PROBS["FP32"].argmax(1)).all(), "FP32 file does not match the per-row file"
log(f"inputs resolved; 9 saved map files verified ({time.time() - t:.1f}s)")

# ------------------------------------------------------------------------------------------- lesion rows
t = time.time()
test = pd.read_csv(SPLIT_DIR / "test.csv")
les_mask = (PER_ROW["label"] == "ischemic") & PER_ROW["dataset"].isin(LESION_DS)
les = PER_ROW.loc[les_mask, ["source_key", "out_path", "dataset", "label", "cohort", "pk"]].copy()
les["row"] = np.where(les_mask)[0]
les = les.merge(test[["out_path", "dwi_out_path"]], on="out_path", how="left", validate="one_to_one")
R["lesion_rows"] = les.groupby("dataset").size().to_dict()
log(f"lesion rows: {R['lesion_rows']} ({time.time() - t:.1f}s)")

# ------------------------------------------------------- masks (functions of the earlier Grad-CAM analysis)
AISD_MASK_ZIP_ID = "1d08fFpEvK4D6YTKfRlNuv_OlIxigZxl6"
ISLES_KEY_RE = re.compile(r"^(sub-strokecase\d+)/(ses-\d+)/\w+/slice(\d+)$")
SKIP_DESCEND = {"hemorrhagic", "ischemic", "normal", "stroke"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan(root, wanted, max_depth=4, stop_when_found=False):
    """ONE breadth-first pass; folders with > 500 entries (image folders) are never descended into."""
    root, t0 = Path(root), time.time()
    found, level, n = {w: [] for w in wanted}, [root], 0
    for depth in range(max_depth + 1):
        nxt = []
        for d in level:
            n += 1
            try:
                entries = list(os.scandir(d))
            except (FileNotFoundError, NotADirectoryError, PermissionError):
                continue
            if depth > 0 and len(entries) > 500:
                continue
            for e in entries:
                if e.is_dir():
                    if e.name in found:
                        found[e.name].append(Path(e.path))
                    if e.name not in SKIP_DESCEND and not e.name.startswith("stage_"):
                        nxt.append(Path(e.path))
        level = nxt
        if not level or (stop_when_found and all(found.values())):
            break
    log(f"scan {root}: {time.time() - t0:.1f}s, {n} dirs, found {{{', '.join(f'{k}: {[str(p) for p in v]}' for k, v in found.items())}}}")
    return found


def to224(a):
    return np.asarray(Image.fromarray((a > 0).astype(np.uint8) * 255).resize(TARGET, Image.NEAREST)) > 0


def aisd_masks():
    z = TMP / "mask.zip"
    gdown.download(f"https://drive.google.com/uc?id={AISD_MASK_ZIP_ID}", str(z), quiet=True)
    R["aisd_mask_zip_sha256"] = sha256(z)
    out = {}
    with zipfile.ZipFile(z) as zf:
        zf.extractall(TMP / "m")
    for f in (TMP / "m").rglob("*.png"):
        if f.stem.isdigit():
            out[(f.parent.name, int(f.stem))] = f
    return out


def load_nifti(folder, stem):
    """Unchanged from the earlier analysis, including the dataset's non-standard .niigz extension."""
    for ext in (".nii.gz", ".nii", ".niigz"):
        p = folder / f"{stem}{ext}"
        if p.exists():
            if ext == ".niigz":  # gzip-compressed NIfTI with a non-standard extension
                with gzip.open(p, "rb") as f:
                    raw = f.read()
                tmp = tempfile.NamedTemporaryFile(suffix=".nii", delete=False)
                tmp.write(raw)
                tmp.close()
                return nib.load(tmp.name)
            return nib.load(str(p))
    return None


ISLES_DERIV = None


def lesion_mask(r, aisd_m, isles_cache):
    """Binary lesion mask on the 224 x 224 input grid, or None (unchanged from the earlier analysis)."""
    global ISLES_DERIV
    ds, sk = r["dataset"], str(r["source_key"])
    if ds == "AISD":
        f = aisd_m.get((sk.split("/")[2], int(re.sub(r"\D", "", Path(sk).stem) or -1)))
        if f is None:
            return None
        a = np.asarray(Image.open(f))
        a = a[..., 0] if a.ndim == 3 else a
        return to224((a == 2) | (a == 3))
    if ds == "CPAISD_v2":
        cands = [Path(sk), Path(sk.replace("/kaggle/input/datasets/atik68/", "/kaggle/input/")),
                 Path(sk.replace("/kaggle/input/", "/kaggle/input/datasets/atik68/", 1))]
        mp = next((c.parent / "mask.npz" for c in cands if (c.parent / "mask.npz").exists()), None)
        if mp is None:
            return None
        with np.load(mp) as d:
            a = np.asarray(d["mask"] if "mask" in d.files else d[d.files[0]])
        if a.ndim == 3 and a.shape[-1] <= 4 and set(np.unique(a)) <= {0, 1}:
            a = a.argmax(-1)
        return to224(a > 0)
    m = ISLES_KEY_RE.match(sk)
    if not m:
        return None
    pat, s = (m.group(1), m.group(2)), int(m.group(3))
    if pat not in isles_cache:
        if ISLES_DERIV is None:
            ISLES_DERIV = scan(ISLES_RAW, {"derivatives"}, max_depth=3, stop_when_found=True)["derivatives"]
        img = load_nifti(ISLES_DERIV[0] / pat[0] / pat[1], f"{pat[0]}_{pat[1]}_msk") if ISLES_DERIV else None
        isles_cache[pat] = img.get_fdata(dtype=np.float32) if img is not None else None
    a = isles_cache[pat]
    if a is None or not 0 <= s < a.shape[2]:
        return None
    return to224(a[:, :, s] > 0)


def image_path(p):
    name = Path(p).name
    if p.startswith("CPAISD_v2/"):
        return first(CPAISD_V2_IMG / name, what=p)
    if p.startswith("AISD/"):
        return first(AISD_K / "ischemic" / name, what=p)
    return first(PHASE_B / p, what=p)


def gray(r):
    """The grayscale image the mask is drawn on: CT centre slice, MRI DWI."""
    p = r["dwi_out_path"] if r["dataset"] == "ISLES2022_dwi_adc" else r["out_path"]
    return np.asarray(Image.open(image_path(p)).convert("L").resize(TARGET), dtype=np.float32)


t = time.time()
aisd_m = aisd_masks()
log(f"AISD masks: {len(aisd_m)} files ({time.time() - t:.0f}s)")
t = time.time()
cache, M, IMG = {}, [], []
for _, r in les.iterrows():
    M.append(lesion_mask(r, aisd_m, cache))
    IMG.append(gray(r))
les["has_mask"] = [m is not None and m.any() for m in M]
R["mask_coverage"] = {ds: [int(g["has_mask"].sum()), int(len(g))] for ds, g in les.groupby("dataset")}
log(f"masks and images read ({time.time() - t:.0f}s); non-empty masks by dataset: {R['mask_coverage']}")
keep = les["has_mask"].to_numpy()
les = les[keep].reset_index(drop=True)
M = [m for m, k in zip(M, keep) if k]
IMG = [g for g, k in zip(IMG, keep) if k]
HEAD = [g > 10 for g in IMG]

# orientation check (identity, as in the earlier analysis): lesion inside the head, lesion contrast
R["orientation_identity_check"] = {}
for ds, g in les.groupby("dataset"):
    ins = [(M[i] & HEAD[i]).sum() / M[i].sum() for i in g.index]
    con = [(IMG[i][M[i]].mean() - IMG[i][HEAD[i]].mean()) / (IMG[i][HEAD[i]].std() + 1e-6) for i in g.index]
    R["orientation_identity_check"][ds] = {"inside_head": float(np.mean(ins)), "contrast_z": float(np.mean(con))}
log(f"orientation (identity): {R['orientation_identity_check']}")
save()


# ------------------------------------------------------------------------------------------------- scoring
def up_bilinear(c, size=224):
    """torch.nn.functional.interpolate(mode='bilinear', align_corners=False) for one 2-D map."""
    n = c.shape[0]
    src = np.clip((np.arange(size) + 0.5) * n / size - 0.5, 0, None)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w = src - i0
    rows = c[i0] * (1 - w)[:, None] + c[i1] * w[:, None]
    return rows[:, i0] * (1 - w)[None, :] + rows[:, i1] * w[None, :]


def display(cam7):
    """The server's display rule (inference.ThreeClassPipeline._display_cam)."""
    cam = cam7 / (cam7.max() + 1e-8)
    thr = np.percentile(cam, 70)
    cam = np.clip((cam - thr) / (1 - thr + 1e-8), 0, 1)
    return np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize(TARGET, Image.BICUBIC), np.float32) / 255


def score(cam7, m, head):
    cam = cam7 / (cam7.max() + 1e-8)
    hm = up_bilinear(cam) * head
    am = np.unravel_index(np.argmax(hm), hm.shape)
    yy, xx = np.nonzero(m)
    dist = np.sqrt(((yy - am[0]) ** 2 + (xx - am[1]) ** 2).min())
    share = m[head].sum() / max(head.sum(), 1)
    energy = float((hm * m).sum() / (hm.sum() + 1e-8))
    reg = display(cam7) > 0.6
    return {"point_hit": int(bool(m[am])), "point_hit_tol16": int(dist <= 16), "point_chance": float(share),
            "energy_in_mask": energy, "energy_lift": float(energy / share) if share > 0 else np.nan,
            "region_px": int(reg.sum()), "region_in_mask": float((reg & m).sum() / max(reg.sum(), 1)),
            "mask_covered": float((reg & m).sum() / max(m.sum(), 1)), "region_touches_mask": int((reg & m).any())}


t = time.time()
rows = []
for k in FILES:
    cm, pr = CAMS[k], PROBS[k]
    for i, r in les.iterrows():
        rows.append({"copy": k, "dataset": r["dataset"], "pk": r["pk"], "row": int(r["row"]),
                     "predicted_ischemic": int(pr[r["row"]].argmax() == 1), **score(cm[r["row"], 1], M[i], HEAD[i])})
S = pd.DataFrame(rows)
S.to_csv(OUT / "maps_vs_masks_per_slice.csv", index=False)
log(f"scored {len(les)} slices x {len(FILES)} copies ({time.time() - t:.0f}s)")

MEAS = ["point_hit", "point_hit_tol16", "point_chance", "energy_in_mask", "energy_lift", "region_px",
        "region_in_mask", "mask_covered", "region_touches_mask"]


def cluster_ci(v, pk, seed):
    v, pk = np.asarray(v, float), np.asarray(pk)
    ok = ~np.isnan(v)
    v, pk = v[ok], pk[ok]
    u, inv = np.unique(pk, return_inverse=True)
    s, n = np.bincount(inv, v, len(u)), np.bincount(inv, minlength=len(u))
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))
        b.append((w * s).sum() / (w * n).sum())
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


F = S[S["copy"] == "FP32"].reset_index(drop=True)
R["by_dataset"] = {}
for ds in LESION_DS:
    fd = F[F["dataset"] == ds].reset_index(drop=True)
    R["by_dataset"][ds] = {}
    for j, k in enumerate(FILES):
        d = S[(S["copy"] == k) & (S["dataset"] == ds)].reset_index(drop=True)
        assert (d["row"].to_numpy() == fd["row"].to_numpy()).all()
        res = {"slices": int(len(d)), "patients": int(d["pk"].nunique()),
               "means": {m: float(d[m].mean()) for m in MEAS},
               "ci95": {m: cluster_ci(d[m], d["pk"], SEED + j) for m in ("point_hit", "energy_in_mask", "region_in_mask",
                                                                         "mask_covered")},
               "by_own_prediction": {lab: {m: float(g[m].mean()) for m in ("point_hit", "energy_in_mask",
                                                                           "region_in_mask")} | {"slices": int(len(g))}
                                     for lab, g in (("predicted_ischemic", d[d.predicted_ischemic == 1]),
                                                    ("missed", d[d.predicted_ischemic == 0])) if len(g)}}
        if k != "FP32":
            res["vs_fp32"] = {
                "pointing_hit_changed_slices": int((d["point_hit"] != fd["point_hit"]).sum()),
                "pointing_hit_gained": int(((d["point_hit"] == 1) & (fd["point_hit"] == 0)).sum()),
                "pointing_hit_lost": int(((d["point_hit"] == 0) & (fd["point_hit"] == 1)).sum()),
                **{f"d_{m}": {"value": float((d[m] - fd[m]).mean()),
                              "ci95": cluster_ci(d[m] - fd[m], d["pk"], 100 + SEED + j)}
                   for m in ("point_hit", "energy_in_mask", "region_in_mask", "mask_covered")}}
        R["by_dataset"][ds][k] = res
        log(f"{ds} | {k}: hit {res['means']['point_hit']:.3f} (chance {res['means']['point_chance']:.3f}) "
            f"tol16 {res['means']['point_hit_tol16']:.3f} energy {res['means']['energy_in_mask']:.3f} "
            f"region in mask {res['means']['region_in_mask']:.3f} covered {res['means']['mask_covered']:.3f}"
            + (f" | hit changed {res['vs_fp32']['pointing_hit_changed_slices']}" if k != "FP32" else ""))
    save()

# check: FP32 against the earlier Grad-CAM analysis of this model (dataset means)
EARLIER = {"AISD": {"point_hit": 0.1318, "point_hit_tol16": 0.4710, "point_chance": 0.0329, "energy_in_mask": 0.0787,
                    "region_px": 1956.37, "region_in_mask": 0.1176, "mask_covered": 0.3168},
           "CPAISD_v2": {"point_hit": 0.1037, "point_hit_tol16": 0.3746, "point_chance": 0.1685,
                         "energy_in_mask": 0.1496, "region_px": 1916.83, "region_in_mask": 0.1272, "mask_covered": 0.0549},
           "ISLES2022_dwi_adc": {"point_hit": 0.0693, "point_hit_tol16": 0.5228, "point_chance": 0.0425,
                                 "energy_in_mask": 0.0633, "region_px": 2139.04, "region_in_mask": 0.0740,
                                 "mask_covered": 0.5076}}
R["check_fp32_vs_earlier"] = {ds: {m: {"here": R["by_dataset"][ds]["FP32"]["means"][m], "earlier": v}
                                   for m, v in e.items()} for ds, e in EARLIER.items()}
for ds, e in R["check_fp32_vs_earlier"].items():
    log(f"check {ds}: " + "; ".join(f"{m} {v['here']:.4f} vs {v['earlier']:.4f}" for m, v in e.items()))
R["minutes_total"] = round((time.time() - T0) / 60, 1)
save()
log("Done.")
