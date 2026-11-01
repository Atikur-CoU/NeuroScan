"""
NeuroScan paper 2 -- Figure 4 cases (2026-10-31). CPU only; no training; no new classification result.

Saves the model input of five held-out slices so that Figure 4 (served overlays of FP32, INT8-dynamic and INT8-static)
can be drawn from real pixels. The cases were chosen before this run by fixed rules, from the saved per-slice
outputs: FP32 and INT8-static from neuroscan-p2-compression-gradcam (v4; batch-independent), INT8-dynamic from
neuroscan-p2-int8dyn-served (one slice per run, as served; INT8-dynamic depends on the batch):
  A, B, C  CT ischemic (AISD), MRI ischemic (ISLES 2022), CT hemorrhagic: among slices all three copies classify
           correctly, the one whose display-region IoU against FP32 is closest to the median for both INT8 copies
           (sum of absolute distances; ties by path).
  D        the INT8-dynamic change whose FP32 margin is closest to the median of its 32 changes (0.013).
  E        the INT8-static change whose FP32 margin is closest to the median of its 332 changes (0.195).

Held-out rows are rebuilt exactly as in neuroscan-p2-compression-gradcam (saved predictions joined to the archived
manifests, internal test then SOOP), so held_i indexes its saved arrays. Image files are found by direct path (no
folder listing, no zip extraction); each prep-kernel zip has its central directory read once. For each case the three
served-model copies are re-run with the server's 4-view TTA, and the probabilities and identity-view maps must match
the saved ones, else the kernel stops.
"""
import io
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

T0 = time.time()


def log(m):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {m}", flush=True)


t = time.time()
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime==1.20.1"], capture_output=True, text=True)
log(f"pip onnxruntime==1.20.1: exit {r.returncode} ({time.time() - t:.0f}s) {r.stderr[-300:]}")
if r.returncode:
    sys.exit(1)

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

VALID_LABELS = ("normal", "ischemic", "hemorrhagic")
TARGET_SIZE = (224, 224)
MRI_PAIR_DATASETS = ("ISLES2022_dwi_adc", "SOOP_dwi_adc")
EXCLUDED_DATASETS = ("MRI_Images",)
CASES = {"A": (1490, "AISD/ischemic/0226208_016.png"),
         "B": (6884, "ISLES2022/stroke/8a421df60b40b9f8525d3e296a200b55.png"),
         "C": (4276, "Kaggle_CT/stroke/8114b173a9b25ba44bd7e84fd46b69e9.png"),
         "D": (7992, "RSNA_ICH/normal/ID_7ea1db35e.png"),
         "E": (4703, "CPAISD_v2/e71329d9cb35a115e9c01f250f6dd836.png")}
OUT = Path("/kaggle/working/output")
OUT.mkdir(parents=True, exist_ok=True)


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


t = time.time()
SPLIT_DIR = ds_root("neuroscan-phase-p-patient-split-v3")
PHASE_B = ds_root("neuroscan-phase-b-images")
RSNA_J = ds_root("neuroscan-phase-j-rsna-images")
RSNA_L = ds_root("neuroscan-phase-l-rsna-more-images")
AISD_K = ds_root("neuroscan-phase-k-aisd-images")
CPAISD_V2_IMG = ds_root("neuroscan-phase-n-cpaisd-v2-images")
SAVED = ds_root("neuroscan-q-round30as123-preds")
CG = "neuroscan-p2-compression-gradcam"
MODELS = {"FP32": nb_file("neuroscan-q-onnx-export-check", "neuroscan_r30a_s123.onnx"),
          "INT8-dyn": nb_file(CG, "models/neuroscan_r30a_s123_int8_dynamic.onnx"),
          "INT8-static": nb_file(CG, "models/neuroscan_r30a_s123_int8_static.onnx")}
PARTIAL = {"FP32": nb_file(CG, "partial/FP32.npz"), "INT8-static": nb_file(CG, "partial/INT8-static.npz"),
           "INT8-dyn": nb_file("neuroscan-p2-int8dyn-served", "INT8-dyn_served.npz")}
ZIPS = {"AISD/normal/": nb_file("neuroscan-q-aisd-normals-prep", "aisd_normal_images.zip"),
        "RSNA_ICH/normal/": nb_file("neuroscan-q-rsna-normals-prep", "rsna_normal_images.zip"),
        "SOOP/": nb_file("neuroscan-q-soop-prep", "soop_images.zip")}
CSV = {"aisd_normal": nb_file("neuroscan-q-aisd-normals-prep", "aisd_normal_rows.csv"),
       "rsna_normal": nb_file("neuroscan-q-rsna-normals-prep", "rsna_normal_rows.csv"),
       "soop": nb_file("neuroscan-q-soop-prep", "soop_rows.csv")}
log(f"inputs resolved ({time.time() - t:.1f}s)")

# ------------------------------------------------------------------- held-out rows, as in the compression kernel
t = time.time()
split_test = pd.read_csv(SPLIT_DIR / "test.csv")
aisd_n, rsna_n, soop = (pd.read_csv(CSV[k]) for k in ("aisd_normal", "rsna_normal", "soop"))
in_cols = ["prev_out_path", "next_out_path", "has_context", "dwi_out_path", "adc_out_path"]
keys = ["source_key", "out_path"]


def keep(df):
    return df[df["label"].isin(VALID_LABELS) & ~df["dataset"].isin(EXCLUDED_DATASETS)]


test_manifest = pd.concat([keep(split_test), aisd_n[aisd_n["split"] == "test"], rsna_n[rsna_n["split"] == "test"]],
                          ignore_index=True)
soop_ext = soop[soop["split"] == "external"]
held = []
for saved, man in ((pd.read_csv(SAVED / "predictions_test.csv"), test_manifest),
                   (pd.read_csv(SAVED / "predictions_soop_external.csv"), soop_ext)):
    j = saved.merge(man[keys + in_cols], on=keys, how="left", validate="one_to_one")
    held.append(j)
held = pd.concat(held, ignore_index=True)
assert len(held) == 10602, len(held)
for name, (i, path) in CASES.items():
    assert held.at[i, "out_path"] == path, f"case {name}: row {i} is {held.at[i, 'out_path']}"
log(f"held-out rows rebuilt: {len(held)}; all five case rows match ({time.time() - t:.1f}s)")

# --------------------------------------------------------------------------------------- files by direct path
ZF, MEMBERS = {}, {}
for pre, zp in ZIPS.items():
    t = time.time()
    ZF[pre] = zipfile.ZipFile(zp)
    MEMBERS[pre] = set(ZF[pre].namelist())
    log(f"zip {zp.name}: central directory read ({time.time() - t:.1f}s)")
READS = {}


def raw(p):
    """The compression kernel's Files.where/raw, for single files."""
    p = str(p)
    src = p.split("/")[0]
    READS[src] = READS.get(src, 0) + 1
    for pre, zf in ZF.items():
        if p.startswith(pre):
            assert p in MEMBERS[pre], f"{p} not in {pre} zip"
            with zf.open(p) as f:
                return f.read()
    name = Path(p).name
    if p.startswith("CPAISD_v2/"):
        cands = [CPAISD_V2_IMG / name]
    elif p.startswith("RSNA_ICH/"):
        cands = [RSNA_L / "hemorrhagic" / name, RSNA_J / "hemorrhagic" / name]
    elif p.startswith("AISD/"):
        cands = [AISD_K / "ischemic" / name]
    else:
        cands = [PHASE_B / p]
    return first(*cands, what=p).read_bytes()


def gray(p):
    return Image.open(io.BytesIO(raw(p))).convert("L").resize(TARGET_SIZE)


def merged(row):
    """Loader.merged, unchanged: CT previous/current/next or replicated; MRI DWI/ADC/DWI."""
    if row["dataset"] in MRI_PAIR_DATASETS:
        dwi, adc = gray(row["dwi_out_path"]), gray(row["adc_out_path"])
        return Image.merge("RGB", (dwi, adc, dwi))
    s = gray(row["out_path"])
    if bool(row["has_context"]):
        return Image.merge("RGB", (gray(row["prev_out_path"]), s, gray(row["next_out_path"])))
    return Image.merge("RGB", (s, s, s))


# ------------------------------------------------------------------- server preprocessing (inference.py, verbatim)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def tta_views(img):
    yield img
    yield img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    yield img.rotate(8, resample=Image.Resampling.NEAREST, expand=False, fillcolor=(0, 0, 0))
    yield img.rotate(-8, resample=Image.Resampling.NEAREST, expand=False, fillcolor=(0, 0, 0))


def to_array(img):
    a = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return (a - MEAN) / STD


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------------------------------- run and check
t = time.time()
idx = [i for i, _ in CASES.values()]
imgs = {name: merged(held.loc[i]) for name, (i, _) in CASES.items()}
log(f"decoded {len(imgs)} cases ({time.time() - t:.1f}s); file reads by source {READS}")
res = {"cases": {}, "file_reads_by_source": READS, "onnxruntime": ort.__version__}
arrays = {}
for k, path in MODELS.items():
    t = time.time()
    o = ort.SessionOptions()
    o.intra_op_num_threads, o.inter_op_num_threads = 2, 1
    s = ort.InferenceSession(str(path), sess_options=o, providers=["CPUExecutionProvider"])
    saved = np.load(PARTIAL[k])
    for name, (i, _) in CASES.items():
        x = np.stack([to_array(v) for v in tta_views(imgs[name])])
        lg, cm = s.run(["logits", "gradcam"], {"image": x})
        p = softmax(lg.astype(np.float64)).mean(0)
        c = cm[0]
        dp = float(np.abs(p - saved["probs"][i]).max())
        dc = float(np.abs(c - saved["cams"][i]).max() / (np.abs(saved["cams"][i]).max() + 1e-12))
        res["cases"].setdefault(name, {"held_i": i, "out_path": CASES[name][1], "dataset": held.at[i, "dataset"],
                                       "label": held.at[i, "label"]})[k] = {
            "probs": p.tolist(), "pred": VALID_LABELS[int(p.argmax())], "max_prob_diff_vs_saved": dp,
            "max_rel_map_diff_vs_saved": dc}
        arrays[f"{name}_{k}_cams"] = c.astype(np.float32)
        res["cases"][name][k]["pred_same_as_saved"] = bool(p.argmax() == saved["probs"][i].argmax())
        if dp > 1e-3 or dc > 1e-2:
            print(f"[ERROR] case {name} {k}: recomputation differs from the saved run (probs {dp:.2e}, maps {dc:.2e})")
            sys.exit(1)
    log(f"{k}: five cases re-run ({time.time() - t:.1f}s): " + "; ".join(
        f"{n} dp={res['cases'][n][k]['max_prob_diff_vs_saved']:.1e} dmap={res['cases'][n][k]['max_rel_map_diff_vs_saved']:.1e}"
        f" same_pred={res['cases'][n][k]['pred_same_as_saved']}" for n in CASES))

for name, img in imgs.items():
    img.save(OUT / f"case_{name}_input.png")
np.savez_compressed(OUT / "fig4_cases.npz", **arrays)
(OUT / "fig4_cases.json").write_text(json.dumps(res, indent=2))
log("Done.")
