"""
NeuroScan paper 2 -- experiments E2 (model compression) and E3 (closed-form vs autograd Grad-CAM), 2026-10-23. CPU only.
No training; no new classification result. The served model (R30A s123 ConvNeXt-Tiny, ONNX FP32, SHA-256 4937ab38...) is
compared with compressed copies of itself. All settings below were fixed before the run (paper/second_paper/
PLAN_E2_E3_E4.md, author decision D6 = A); held-out rows are run once and every variant is reported.

Held-out rows: the pooled internal test (internal test + CPAISD, 8,930 rows) and the SOOP hold-out (1,672 rows), i.e.
exactly the rows of the saved final-evaluation predictions. Their input columns come from the archived manifests; image
files are found by direct path (no folder listing, no zip extraction). Merge rules = the final model's Loader.merged
(CT previous/current/next or replicated; MRI DWI/ADC/DWI); the RSNA rows use the Round 24 images (rsna_dicom=False).

Variants (fixed):
  FP32        the served file
  FP16        onnxconverter_common float16, keep_io_types=True; if any output is non-finite on the first 64 rows,
              FP16-LN (LayerNormalization/ReduceMean/Pow/Sqrt kept in FP32) is built and reported as well
  INT8-dyn    quant_pre_process, then quantize_dynamic: QInt8 weights, op types MatMul
  INT8-static quant_pre_process, then quantize_static: QDQ, per-channel, QInt8 weights / QUInt8 activations, MinMax
              calibration on 512 TRAINING rows (equal per source x label, seed 123, identity view), op types MatMul+Conv
  Both INT8 variants exclude every node downstream of the stage-4 activations (pooling -> LayerNorm -> linear and the
  in-graph Grad-CAM), so the closed-form Grad-CAM weights stay exact and map changes come from the backbone only.

E2: size and SHA-256; server-condition cost of each variant in a fresh process with the server's pinned packages and
its inference.py verbatim (1 thread, arena off): load time, RSS, peak RSS, latency of 200 requests (100 CT with
context, 60 MRI DWI+ADC, 40 CT single slice) end-to-end and model-only; agreement with FP32 on all held-out rows
under the deployed rule (4-view TTA, plain argmax): argmax agreement with a patient-cluster bootstrap CI, flip table,
probability differences, alert-tier changes, by cohort and source; secondary: paired accuracy deltas.
Sanity: FP32 ONNX must reproduce the saved probabilities (max |diff| <= 1e-3; an argmax difference is allowed only
where the saved top-2 margin is < 1e-4), else the kernel stops.
Time guard (fixed): if a variant's projected agreement time exceeds 150 min after 512 rows, it runs on a fixed
stratified subsample of 3,000 held-out rows instead, and the result says so.

E3: 600 held-out rows (equal per source x label, seed 123), identity view, all three classes: closed-form maps (ONNX
FP32 and the same formula in PyTorch) vs autograd Grad-CAM (hook on ConvNeXt stage 4); time and peak RSS of forward
only / closed form / autograd (predicted class; all classes) in PyTorch and logits-only vs logits+Grad-CAM in ONNX
Runtime, each in a fresh 1-thread process on 100 rows. Maps under compression on all held-out rows: Pearson r of the
7 x 7 map for FP32's predicted class, same peak cell, IoU of the deployed display region (server rule, > 0.6), and
outline appearing/disappearing under each variant's own prediction.
"""
import hashlib
import io
import json
import os
import re
import resource
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

T0 = time.time()


def rss_gb():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return float("nan")


def log(m):
    print(f"[{(time.time() - T0) / 60:6.1f} min | RSS {rss_gb():5.1f} GB] {m}", flush=True)


def pip(args, target=None):
    t = time.time()
    cmd = [sys.executable, "-m", "pip", "install", "-q", *args] + (["--target", target] if target else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    log(f"pip {' '.join(args)}{' -> ' + target if target else ''}: exit {r.returncode} ({time.time() - t:.0f}s) {r.stderr[-300:]}")
    if r.returncode:
        sys.exit(1)


SERVER_PINS = ["onnxruntime==1.20.1", "numpy==2.1.3", "pillow==11.0.0"]
# Kaggle's image runs Python 3.13 (2026-10-23): onnx 1.17.0 has no cp313 wheel, 1.18.0 does. The server pins
# (onnxruntime 1.20.1, numpy 2.1.3, pillow 11.0.0) all have cp313 wheels; the server itself runs Python 3.11.
pip(["onnxruntime==1.20.1", "onnx==1.18.0"])
pip(["--no-deps", "onnxconverter-common==1.16.0"])   # --no-deps: keep Kaggle's protobuf; float16 needs onnx only
pip(SERVER_PINS, target="/tmp/srvpkgs")

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_dynamic, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process
from torchvision import models

SEED = 123
VALID_LABELS = ("normal", "ischemic", "hemorrhagic")
LABEL_TO_IDX = {k: i for i, k in enumerate(VALID_LABELS)}
TARGET_SIZE = (224, 224)
MRI_PAIR_DATASETS = ("ISLES2022_dwi_adc", "SOOP_dwi_adc")
EXCLUDED_DATASETS = ("MRI_Images",)
NEW_PREFIXES = ("AISD/normal/", "RSNA_ICH/normal/")       # zip members of the prep kernels (RSNA extra is train-only)
FP32_SHA = "4937ab381ee3e7c29d5ab18ea761a8c19ddffb6488210fb6030511a8041502eb"
WEIGHTS_SHA_PREFIX = "1e2db8a7431f"
N_CALIB, N_E3, N_E3_TIMING, N_LAT = 512, 600, 100, (100, 60, 40)
CALIB_CHUNK = 8   # calibration images per MinMax range update (memory only; same ranges)
N_BOOT = 2000
AGREE_THREADS, AGREE_ROWS_PER_RUN = 4, 16
TIME_GUARD_MIN, GUARD_SUBSAMPLE = 150, 3000
FP16_LN_BLOCK = ["LayerNormalization", "ReduceMean", "Pow", "Sqrt"]
KEYS = ["accuracy", "balanced_accuracy", "macro_f1_present_classes", "normal_recall", "ischemic_recall", "ischemic_precision",
        "hemorrhagic_recall", "hemorrhagic_precision", "ischemic_auroc", "hemorrhagic_auroc", "macro_ovr_auroc"]

OUT = Path("/kaggle/working/output")
MODELS_OUT = OUT / "models"
WORK = Path("/tmp/p2")
for d in (OUT, MODELS_OUT, WORK):
    d.mkdir(parents=True, exist_ok=True)
R = {"note": "paper 2, E2 + E3; settings fixed in advance (PLAN_E2_E3_E4.md); FP32 = the served ONNX file",
     "server_packages": SERVER_PINS, "python": sys.version.split()[0], "server_python_on_render": "3.11",
     "main_packages": {"onnxruntime": ort.__version__, "onnx": onnx.__version__, "torch": torch.__version__,
                       "numpy": np.__version__}}
MAX_IR = 10   # highest ONNX IR version onnxruntime 1.20.1 loads; onnx 1.18's make_model defaults to 11


def fix_ir(path):
    """Keeps a saved model loadable by the server's onnxruntime (IR <= 10); every op used is opset 17."""
    m = onnx.load(str(path))
    if m.ir_version > MAX_IR:
        m.ir_version = MAX_IR
        onnx.save(m, str(path))


def save_results():
    with open(OUT / "e2e3_results.json", "w") as f:
        json.dump(jsonable(R), f, indent=2)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, float) and o != o:
        return None
    return o


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


# ---------------------------------------------------------------------------------------------------------- inputs
t = time.time()
SPLIT_DIR = ds_root("neuroscan-phase-p-patient-split-v3")
PHASE_B = ds_root("neuroscan-phase-b-images")
RSNA_J = ds_root("neuroscan-phase-j-rsna-images")
RSNA_L = ds_root("neuroscan-phase-l-rsna-more-images")
AISD_K = ds_root("neuroscan-phase-k-aisd-images")
CPAISD_V2_IMG = ds_root("neuroscan-phase-n-cpaisd-v2-images")
SAVED = ds_root("neuroscan-q-round30as123-preds")
PHASE2 = ds_root("neuroscan-q-phase2-split-data")
WEIGHTS = first(ds_root("neuroscan-q-r30a-s123-weights") / "r30a_s123_weights.pt", what="weights")
ONNX_FP32 = nb_file("neuroscan-q-onnx-export-check", "neuroscan_r30a_s123.onnx")
ZIPS = {"AISD/normal/": nb_file("neuroscan-q-aisd-normals-prep", "aisd_normal_images.zip"),
        "RSNA_ICH/normal/": nb_file("neuroscan-q-rsna-normals-prep", "rsna_normal_images.zip"),
        "SOOP/": nb_file("neuroscan-q-soop-prep", "soop_images.zip")}
CSV = {"aisd_normal": nb_file("neuroscan-q-aisd-normals-prep", "aisd_normal_rows.csv"),
       "rsna_normal": nb_file("neuroscan-q-rsna-normals-prep", "rsna_normal_rows.csv"),
       "soop": nb_file("neuroscan-q-soop-prep", "soop_rows.csv")}
log(f"inputs resolved ({time.time() - t:.1f}s): ONNX {ONNX_FP32}")
R["fp32_sha256"] = sha256(ONNX_FP32)
assert R["fp32_sha256"] == FP32_SHA, f"served ONNX checksum mismatch: {R['fp32_sha256']}"

# The server's inference.py, verbatim (inserted by build_kernel.py; its SHA-256 is checked here).
INFERENCE_PY = "__INFERENCE_PY__"
INFERENCE_PY_SHA = "__INFERENCE_PY_SHA__"
assert hashlib.sha256(INFERENCE_PY.encode("utf-8")).hexdigest() == INFERENCE_PY_SHA
R["inference_py_sha256"] = INFERENCE_PY_SHA
SHA_LINE = f'EXPECTED_SHA256 = "{FP32_SHA}"'
assert INFERENCE_PY.count(SHA_LINE) == 1


def server_dir(name, model_path):
    """A copy of the server with only the model file (symlink) and its expected SHA-256 swapped."""
    d = WORK / f"srv_{name}"
    (d / "models").mkdir(parents=True, exist_ok=True)
    (d / "inference.py").write_text(INFERENCE_PY.replace(SHA_LINE, f'EXPECTED_SHA256 = "{sha256(model_path)}"'))
    link = d / "models" / "neuroscan_r30a_s123.onnx"
    if link.exists() or link.is_symlink():
        link.unlink()
    os.symlink(model_path, link)
    return d


sys.path.insert(0, str(server_dir("fp32", ONNX_FP32)))
import inference as srv  # noqa: E402  (server preprocessing, display rule and tier colours)

# ------------------------------------------------------------------------------------------------------------ rows
t = time.time()
split_test = pd.read_csv(SPLIT_DIR / "test.csv")
split_train = pd.read_csv(SPLIT_DIR / "train.csv")
aisd_n, rsna_n, soop = (pd.read_csv(CSV[k]) for k in ("aisd_normal", "rsna_normal", "soop"))
in_cols = ["prev_out_path", "next_out_path", "has_context", "dwi_out_path", "adc_out_path"]
keys = ["source_key", "out_path"]


def keep(df):
    return df[df["label"].isin(VALID_LABELS) & ~df["dataset"].isin(EXCLUDED_DATASETS)]


test_manifest = pd.concat([keep(split_test), aisd_n[aisd_n["split"] == "test"], rsna_n[rsna_n["split"] == "test"]],
                          ignore_index=True)
soop_ext = soop[soop["split"] == "external"]
saved_t = pd.read_csv(SAVED / "predictions_test.csv")
saved_s = pd.read_csv(SAVED / "predictions_soop_external.csv")
P_SAVED = [f"p_{l}_round30as123" for l in VALID_LABELS]
held = []
for saved, man, name in ((saved_t, test_manifest, "test"), (saved_s, soop_ext, "soop")):
    j = saved.merge(man[keys + ["dataset", "label"] + in_cols].rename(columns={"dataset": "_ds", "label": "_lb"}),
                    on=keys, how="left", validate="one_to_one")
    assert j["_ds"].notna().all(), f"{name}: {int(j['_ds'].isna().sum())} saved rows missing from the manifest"
    assert (j["_ds"] == j["dataset"]).all() and (j["_lb"] == j["label"]).all(), f"{name}: dataset/label mismatch"
    held.append(j.drop(columns=["_ds", "_lb"]))
held = pd.concat(held, ignore_index=True)
held["cohort"] = np.where(held["dataset"] == "SOOP_dwi_adc", "SOOP",
                          np.where(held["dataset"] == "CPAISD_v2", "CPAISD", "internal_test"))
assert (held["cohort"] != "SOOP").sum() == 8930 and (held["cohort"] == "SOOP").sum() == 1672, held["cohort"].value_counts()

RSNA_MAP = dict(zip(*(lambda m: (m["stem"].astype(str), m["PatientID"]))(pd.read_csv(PHASE2 / "rsna_patient_map.csv"))))
CPAISD_RE = re.compile(r"/([\d.]{10,})/(\d+)/raw\.dcm$")
MRI_RE = re.compile(r"^(sub-[\w]+)/(ses-\w+)/")


def patient_key(ds, sk, op, subject):
    """As in the paper-1 analysis kernels (patient-cluster bootstrap)."""
    sk, op = str(sk), str(op)
    if ds == "AISD":
        return f"AISD:{sk.split('/')[2]}"
    if ds == "CPAISD_v2":
        m = CPAISD_RE.search(sk.replace("\\", "/"))
        return f"CPAISD:{m.group(1)}" if m else f"CPAISD_row:{op}"
    if ds == "ISLES2022_dwi_adc":
        m = MRI_RE.match(sk)
        return f"ISLES:{m.group(1)}/{m.group(2)}" if m else f"ISLES_row:{sk}"
    if ds == "SOOP_dwi_adc":
        if isinstance(subject, str) and subject:
            return f"SOOP:{subject}"
        m = MRI_RE.match(sk)
        return f"SOOP:{m.group(1)}" if m else f"SOOP_row:{sk}"
    if ds == "RSNA_ICH":
        pid = RSNA_MAP.get(Path(op).stem)
        return f"RSNA:{pid}" if isinstance(pid, str) and pid else f"RSNA_row:{Path(op).stem}"
    if ds == "SICH_CT":
        return f"SICH:{sk.replace(chr(92), '/').split('/')[-2]}"
    return f"{ds}_row:{op}"


held["pk"] = [patient_key(d, s, o, u) for d, s, o, u in zip(held["dataset"], held["source_key"], held["out_path"], held["subject"])]


def equal_strata(df, n, seed):
    """n rows, as equal as possible per dataset x label (all of a small stratum), fixed seed."""
    g = df.groupby(["dataset", "label"])
    sizes, left, quota = g.size().sort_values(), n, {}
    for i, (k, s) in enumerate(sizes.items()):
        quota[k] = min(int(s), left // (len(sizes) - i))
        left -= quota[k]
    return pd.concat([grp.sample(n=quota[k], random_state=seed) for k, grp in g]).reset_index(drop=True)


calib_pool = pd.concat([keep(split_train), aisd_n[aisd_n["split"] == "train"], rsna_n[rsna_n["split"] == "train"],
                        soop[soop["split"] == "train"]], ignore_index=True)
calib = equal_strata(calib_pool, N_CALIB, SEED)
e3_rows = equal_strata(held.reset_index().rename(columns={"index": "held_i"}), N_E3, SEED)
guard_rows = np.sort(equal_strata(held.reset_index().rename(columns={"index": "held_i"}), GUARD_SUBSAMPLE, SEED)["held_i"].to_numpy())
R["rows"] = {"held_out_by_cohort": held["cohort"].value_counts().to_dict(),
             "held_out_by_dataset": held["dataset"].value_counts().to_dict(), "patients": int(held["pk"].nunique()),
             "calibration_by_stratum": {f"{a}|{b}": int(v) for (a, b), v in calib.groupby(["dataset", "label"]).size().items()},
             "e3_by_stratum": {f"{a}|{b}": int(v) for (a, b), v in e3_rows.groupby(["dataset", "label"]).size().items()}}
log(f"rows ({time.time() - t:.1f}s): {R['rows']['held_out_by_cohort']}, patients {R['rows']['patients']}, "
    f"calibration {len(calib)}, E3 {len(e3_rows)}")


# ------------------------------------------------------------------------------------------------------- files
class Files:
    """Direct-path resolver (the final model's Loader._resolve with rsna_dicom=False, without its folder indexes):
    prep-kernel zips are read member by member (one central-directory read each, never extracted); RSNA/AISD lesion
    images live in fixed folders (RSNA: phase-l before phase-j, as the old index overrode); CPAISD_v2 at the dataset
    root; everything else under the phase-b images root."""

    def __init__(self):
        self.zips, self.members = {}, {}
        for pre, zp in ZIPS.items():
            t0 = time.time()
            zf = zipfile.ZipFile(zp)
            self.zips[pre], self.members[pre] = zf, set(zf.namelist())
            log(f"zip {zp.name}: {len(self.members[pre])} members in central directory ({time.time() - t0:.1f}s)")
        self.reads = {}

    def where(self, p):
        p = str(p)
        for pre in self.zips:
            if p.startswith(pre):
                return ("zip", pre, p) if p in self.members[pre] else None
        name = Path(p).name
        if p.startswith("CPAISD_v2/"):
            cands = [CPAISD_V2_IMG / name]
        elif p.startswith("RSNA_ICH/"):
            cands = [RSNA_L / "hemorrhagic" / name, RSNA_J / "hemorrhagic" / name]
        elif p.startswith("AISD/"):
            cands = [AISD_K / "ischemic" / name]
        else:
            cands = [PHASE_B / p]
        hit = next((c for c in cands if c.exists()), None)
        return ("fs", None, hit) if hit else None

    def raw(self, p):
        kind, pre, ref = self.loc[str(p)]
        src = pre if kind == "zip" else str(p).split("/")[0]
        with LOCK:   # one lock for the read counter and the shared ZipFile handles
            self.reads[src] = self.reads.get(src, 0) + 1
            if kind == "zip":
                with self.zips[pre].open(ref) as f:
                    return f.read()
        return Path(ref).read_bytes()

    def resolve_all(self, paths):
        t0, self.loc, missing = time.time(), {}, []
        for p in sorted(set(paths)):
            w = self.where(p)
            if w is None:
                missing.append(p)
            else:
                self.loc[p] = w
        log(f"resolved {len(self.loc)} files by direct path ({time.time() - t0:.1f}s); missing {len(missing)}")
        if missing:
            print(f"[ERROR] {len(missing)} image files not found, e.g. {missing[:5]}")
            sys.exit(1)


import threading  # noqa: E402

LOCK = threading.Lock()
FILES = Files()
raw_bytes = FILES.raw


def gray(p):
    """Loader._slice: 8-bit grayscale, resized to 224 x 224."""
    return Image.open(io.BytesIO(raw_bytes(p))).convert("L").resize(TARGET_SIZE)


def merged(row):
    """Loader.merged, unchanged."""
    if row["dataset"] in MRI_PAIR_DATASETS:
        dwi, adc = gray(row["dwi_out_path"]), gray(row["adc_out_path"])
        return Image.merge("RGB", (dwi, adc, dwi))
    s = gray(row["out_path"])
    if bool(row["has_context"]):
        return Image.merge("RGB", (gray(row["prev_out_path"]), s, gray(row["next_out_path"])))
    return Image.merge("RGB", (s, s, s))


def needed_paths(df):
    out = []
    for _, r in df.iterrows():
        if r["dataset"] in MRI_PAIR_DATASETS:
            out += [r["dwi_out_path"], r["adc_out_path"]]
        else:
            out.append(r["out_path"])
            if bool(r["has_context"]):
                out += [r["prev_out_path"], r["next_out_path"]]
    return out


FILES.resolve_all(needed_paths(held) + needed_paths(calib))


def decode(df, tag):
    t0 = time.time()
    arr = np.empty((len(df), 224, 224, 3), dtype=np.uint8)
    rows = [r for _, r in df.iterrows()]

    def one(i):
        arr[i] = np.asarray(merged(rows[i]))

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, range(len(rows))))
    log(f"decoded {tag}: {len(df)} rows ({time.time() - t0:.0f}s, {arr.nbytes / 1e9:.2f} GB)")
    return arr


X_HELD = decode(held, "held-out")
X_CAL = decode(calib, "calibration")
R["file_reads_by_source"] = dict(FILES.reads)
log(f"file reads by source: {FILES.reads}")


def views_batch(arrs):
    """The server's 4-view TTA and normalisation, exactly (inference._tta_views, inference._to_array)."""
    return np.stack([srv._to_array(v) for a in arrs for v in srv._tta_views(Image.fromarray(a))])


# ------------------------------------------------------------------------------------------------------ variants
def session(path, threads):
    o = ort.SessionOptions()
    o.intra_op_num_threads, o.inter_op_num_threads = threads, 1
    return ort.InferenceSession(str(path), sess_options=o, providers=["CPUExecutionProvider"])


def head_nodes(model):
    """Every node downstream of ConvNeXt stage 4's output (the Einsum's activation input)."""
    ein = [n for n in model.graph.node if n.op_type == "Einsum"]
    assert len(ein) == 1, f"expected one Einsum (in-graph Grad-CAM), found {len(ein)}"
    a = ein[0].input[1]
    consumers = {}
    for n in model.graph.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    seen, todo = {}, [a]
    while todo:
        for n in consumers.get(todo.pop(), []):
            if n.name not in seen:
                seen[n.name] = n
                todo += list(n.output)
    assert all(seen), "unnamed head nodes"
    return sorted(seen), sorted({n.op_type for n in seen.values()})


class Calib(CalibrationDataReader):
    """Supports CalibStridedMinMax (len + set_range): ORT calibrates one stride at a time and merges the ranges."""

    def __init__(self, arrs):
        self.data = [{"image": srv._to_array(Image.fromarray(a))[None]} for a in arrs]
        self.set_range(0, len(self.data))

    def __len__(self):
        return len(self.data)

    def set_range(self, start_index, end_index):
        self.it = iter(self.data[start_index:end_index])

    def get_next(self):
        return next(self.it, None)


def op_counts(path):
    c = {}
    for n in onnx.load(str(path)).graph.node:
        c[n.op_type] = c.get(n.op_type, 0) + 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


VARIANTS = {"FP32": ONNX_FP32}
t = time.time()
pre = WORK / "fp32_pre.onnx"
quant_pre_process(str(ONNX_FP32), str(pre))
fix_ir(pre)
pm = onnx.load(str(pre))
excl, excl_ops = head_nodes(pm)
R["quant_excluded_head"] = {"nodes": len(excl), "op_types": excl_ops}
log(f"quant_pre_process done ({time.time() - t:.0f}s); head nodes excluded from INT8: {len(excl)} ({excl_ops})")

from onnxconverter_common import float16  # noqa: E402

t = time.time()
f16 = MODELS_OUT / "neuroscan_r30a_s123_fp16.onnx"
onnx.save(float16.convert_float_to_float16(onnx.load(str(ONNX_FP32)), keep_io_types=True), str(f16))
fix_ir(f16)
VARIANTS["FP16"] = f16
log(f"FP16 built ({time.time() - t:.0f}s)")

t = time.time()
dyn = MODELS_OUT / "neuroscan_r30a_s123_int8_dynamic.onnx"
quantize_dynamic(str(pre), str(dyn), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"], nodes_to_exclude=excl)
fix_ir(dyn)
VARIANTS["INT8-dyn"] = dyn
log(f"INT8 dynamic built ({time.time() - t:.0f}s)")

t = time.time()
sta = MODELS_OUT / "neuroscan_r30a_s123_int8_static.onnx"
quantize_static(str(pre), str(sta), Calib(X_CAL), quant_format=QuantFormat.QDQ, per_channel=True,
                weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8, op_types_to_quantize=["MatMul", "Conv"],
                nodes_to_exclude=excl, calibrate_method=CalibrationMethod.MinMax,
                # By default MinMax keeps every calibration image's full activation outputs until the end (tens of
                # GB for ConvNeXt-Tiny; v2 was killed with no log). CalibStridedMinMax computes and merges the ranges
                # per stride (min of mins, max of maxes): identical ranges. Not CalibMaxIntermediateOutputs: in ORT
                # 1.20.1 it clears each chunk without computing it (v3 failed: "No data is collected").
                extra_options={"CalibStridedMinMax": CALIB_CHUNK})
assert len(X_CAL) % CALIB_CHUNK == 0
fix_ir(sta)
VARIANTS["INT8-static"] = sta
log(f"INT8 static built with {len(X_CAL)} calibration rows ({time.time() - t:.0f}s)")


def finite_check(path, n=64):
    try:
        s = session(path, AGREE_THREADS)
    except Exception as e:  # e.g. no FP16 kernel on this CPU execution provider
        return f"session creation failed: {type(e).__name__}: {str(e)[:400]}"
    lg, cm = s.run(["logits", "gradcam"], {"image": views_batch(X_HELD[:n // 4])})
    return None if np.isfinite(lg).all() and np.isfinite(cm).all() else "non-finite outputs"


R["variant_errors"] = {}
err = finite_check(f16)
if err:
    log(f"FP16: {err}")
    R["variant_errors"]["FP16"] = err
    f16ln = MODELS_OUT / "neuroscan_r30a_s123_fp16_ln.onnx"
    onnx.save(float16.convert_float_to_float16(onnx.load(str(ONNX_FP32)), keep_io_types=True,
                                               op_block_list=list(float16.DEFAULT_OP_BLOCK_LIST) + FP16_LN_BLOCK), str(f16ln))
    fix_ir(f16ln)
    VARIANTS["FP16-LN"] = f16ln
    err2 = finite_check(f16ln)
    if err2:
        R["variant_errors"]["FP16-LN"] = err2
    if err.startswith("session"):
        VARIANTS.pop("FP16")
    if err2 and err2.startswith("session"):
        VARIANTS.pop("FP16-LN")
for k in ("INT8-dyn", "INT8-static"):
    e = finite_check(VARIANTS[k])
    if e:
        R["variant_errors"][k] = e
        if e.startswith("session"):
            VARIANTS.pop(k)

R["variants"] = {}
BUILT = {"FP32": ONNX_FP32, "FP16": f16, "FP16-LN": MODELS_OUT / "neuroscan_r30a_s123_fp16_ln.onnx",
         "INT8-dyn": dyn, "INT8-static": sta}
for k, p in BUILT.items():
    if not Path(p).exists():
        continue
    onnx.checker.check_model(str(p))
    R["variants"][k] = {"file": Path(p).name, "mb": round(Path(p).stat().st_size / 1e6, 2), "sha256": sha256(p),
                        "op_counts": op_counts(p), "runs_here": k in VARIANTS}
    log(f"{k}: {R['variants'][k]['mb']} MB, sha256 {R['variants'][k]['sha256'][:12]}, ops {list(R['variants'][k]['op_counts'].items())[:8]}")
save_results()


# ------------------------------------------------------------------------------------------- E2 agreement runs
def run_all(path, idx, tag, guard):
    """4-view TTA probabilities (plain softmax mean) and identity-view Grad-CAM maps (all classes) for rows idx.
    guard: return None after 512 rows if the projected time exceeds TIME_GUARD_MIN (never for FP32)."""
    s = session(path, AGREE_THREADS)
    probs, cams, t0 = np.zeros((len(idx), 3)), np.zeros((len(idx), 3, 7, 7), dtype=np.float32), time.time()
    for st in range(0, len(idx), AGREE_ROWS_PER_RUN):
        b = idx[st:st + AGREE_ROWS_PER_RUN]
        lg, cm = s.run(["logits", "gradcam"], {"image": views_batch(X_HELD[b])})
        probs[st:st + len(b)] = srv._softmax(lg.astype(np.float64)).reshape(len(b), 4, 3).mean(1)
        cams[st:st + len(b)] = cm.reshape(len(b), 4, 3, 7, 7)[:, 0]
        done, el = st + len(b), time.time() - t0
        projected = el / done * len(idx) / 60
        if done in (512, len(idx)) or done % 2048 == 0:
            log(f"  [{tag}] {done}/{len(idx)} ({el:.0f}s; projected {projected:.0f} min)")
        if guard and done == 512 and projected > TIME_GUARD_MIN:
            return None
    return probs, cams


ALL = np.arange(len(held))
PRED, CAMS, ROWSET = {}, {}, {}
(OUT / "partial").mkdir(exist_ok=True)
for k, p in VARIANTS.items():
    t = time.time()
    r, ROWSET[k] = run_all(p, ALL, k, guard=k != "FP32"), ALL
    if r is None:
        log(f"{k}: projected time over {TIME_GUARD_MIN} min -> fixed subsample of {GUARD_SUBSAMPLE} rows")
        R["variant_errors"][f"{k}_time_guard"] = f"run on the fixed {GUARD_SUBSAMPLE}-row subsample"
        r, ROWSET[k] = run_all(p, guard_rows, k, guard=False), guard_rows
    PRED[k], CAMS[k] = r
    np.savez_compressed(OUT / "partial" / f"{k}.npz", probs=PRED[k], cams=CAMS[k], rows=ROWSET[k])
    R.setdefault("agreement_run_minutes", {})[k] = round((time.time() - t) / 60, 1)
    if k == "FP32":
        sv = held[P_SAVED].to_numpy(float)
        d = np.abs(PRED[k] - sv)
        srt = np.sort(sv, axis=1)
        mism = PRED[k].argmax(1) != sv.argmax(1)
        tie = (srt[:, 2] - srt[:, 1]) < 1e-4
        R["fp32_vs_saved"] = {"max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
                              "argmax_mismatches": int(mism.sum()), "mismatches_at_ties": int((mism & tie).sum())}
        log(f"FP32 ONNX vs saved final-evaluation probabilities: {R['fp32_vs_saved']}")
        save_results()
        if d.max() > 1e-3 or (mism & ~tie).any():
            print("[ERROR] FP32 ONNX does not reproduce the saved probabilities; inputs are not the evaluated ones")
            sys.exit(1)
    save_results()


# ------------------------------------------------------------------------------------------- E2 statistics
def decide(p):
    return p.argmax(1)


def tier(p, mri):
    """Alert tier exactly as served (inference.ThreeClassPipeline._risk_tier_color; MRI hemorrhage = not validated)."""
    i = int(p.argmax())
    if mri and i == 2:
        return "not_validated"
    c = srv.ThreeClassPipeline._risk_tier_color(VALID_LABELS[i], *map(float, p))
    return {(192, 57, 43): "red", (230, 126, 34): "orange", (127, 140, 141): "grey", (41, 128, 185): "blue"}[c]


def auroc(yb, s):
    n1 = int(yb.sum())
    n0 = len(yb) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return float((r[yb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def slice_metrics(y, pred, p):
    r = {"n": int(len(y)), "accuracy": float((pred == y).mean())}
    present = [c for c in range(3) if (y == c).any()]
    cm = np.bincount(y * 3 + pred, minlength=9).reshape(3, 3)
    r["confusion_matrix_rows_true_cols_pred"] = cm.tolist()
    recs, f1s, aucs = [], [], []
    for c in range(3):
        tp, fn, fp = cm[c, c], cm[c].sum() - cm[c, c], cm[:, c].sum() - cm[c, c]
        if tp + fn:
            r[f"{VALID_LABELS[c]}_recall"] = tp / (tp + fn)
        if tp + fp:
            r[f"{VALID_LABELS[c]}_precision"] = tp / (tp + fp)
        if c in present:
            recs.append(tp / (tp + fn))
            f1s.append(2 * tp / (2 * tp + fp + fn))
            a = auroc((y == c).astype(int), p[:, c])
            if a == a:
                aucs.append(a)
            if c > 0:
                r[f"{VALID_LABELS[c]}_auroc"] = a
    r["balanced_accuracy"] = float(np.mean(recs))
    r["macro_f1_present_classes"] = float(np.mean(f1s))
    if len(aucs) >= 2:
        r["macro_ovr_auroc"] = float(np.mean(aucs))
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r.items()}


def clusters(df):
    d0 = df.reset_index(drop=True)
    return [[g.index.to_numpy()[ix] for ix in g.groupby("pk").indices.values()] for _, g in d0.groupby("dataset")]


def boot_idx(groups, rng):
    return np.concatenate([np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))]) for cl in groups])


def compare(df, pa, pb, seed):
    """a = FP32, b = variant, deployed rule; patient-cluster bootstrap (2,000) for agreement and metric deltas."""
    y = df["label"].map(LABEL_TO_IDX).to_numpy()
    da, db = decide(pa), decide(pb)
    mri = df["dataset"].isin(MRI_PAIR_DATASETS).to_numpy()
    ta = np.array([tier(p, m) for p, m in zip(pa, mri)])
    tb = np.array([tier(p, m) for p, m in zip(pb, mri)])
    agree, tagree = (da == db).astype(float), (ta == tb).astype(float)
    ma, mb = slice_metrics(y, da, pa), slice_metrics(y, db, pb)
    groups, rng = clusters(df), np.random.default_rng(seed)
    bs = {"agreement": [], "tier_agreement": [], **{k: [] for k in KEYS}}
    for _ in range(N_BOOT):
        ix = boot_idx(groups, rng)
        bs["agreement"].append(agree[ix].mean())
        bs["tier_agreement"].append(tagree[ix].mean())
        a, b = slice_metrics(y[ix], da[ix], pa[ix]), slice_metrics(y[ix], db[ix], pb[ix])
        for k in KEYS:
            if k in a and k in b and a[k] == a[k] and b[k] == b[k]:
                bs[k].append(b[k] - a[k])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    dp = np.abs(pa - pb)
    out = {"n": int(len(df)), "patients": int(df["pk"].nunique()),
           "argmax_agreement": float(agree.mean()), "argmax_agreement_ci95": ci(bs["agreement"]),
           "flips": int((da != db).sum()), "flip_table_rows_fp32_cols_variant": np.bincount(da * 3 + db, minlength=9).reshape(3, 3).tolist(),
           "tier_agreement": float(tagree.mean()), "tier_agreement_ci95": ci(bs["tier_agreement"]),
           "tier_changes": pd.Series([f"{a}->{b}" for a, b in zip(ta, tb) if a != b]).value_counts().to_dict(),
           "prob_abs_diff_max": float(dp.max()), "prob_abs_diff_mean": float(dp.mean()),
           "prob_abs_diff_p99": float(np.percentile(dp.max(1), 99)), "secondary_metric_deltas_variant_minus_fp32": {}}
    for k in KEYS:
        if k in ma and k in mb and len(bs[k]) > 50:
            out["secondary_metric_deltas_variant_minus_fp32"][k] = {"fp32": ma[k], "variant": mb[k], "diff": mb[k] - ma[k],
                                                                    "ci95": ci(bs[k])}
    out["by_source"] = {ds: {"n": int(len(g)), "argmax_agreement": float(agree[g.index.to_numpy()].mean()),
                             "flips": int((da != db)[g.index.to_numpy()].sum())}
                        for ds, g in df.reset_index(drop=True).groupby("dataset")}
    return out


COHORTS = {"pooled_internal_test": held["cohort"] != "SOOP", "internal_test": held["cohort"] == "internal_test",
           "CPAISD": held["cohort"] == "CPAISD", "SOOP_holdout": held["cohort"] == "SOOP"}
R["E2_agreement"] = {}
seed = 900
for k in VARIANTS:
    if k == "FP32":
        continue
    rows = ROWSET[k]
    pos = {r: i for i, r in enumerate(rows)}
    R["E2_agreement"][k] = {"rows_used": "all held-out rows" if len(rows) == len(held) else f"fixed {len(rows)}-row subsample"}
    for cname, mask in COHORTS.items():
        sel = np.array([r for r in rows if mask.iloc[r]])
        if len(sel) == 0:
            continue
        seed += 1
        res = compare(held.iloc[sel].reset_index(drop=True), PRED["FP32"][sel], PRED[k][[pos[r] for r in sel]], seed)
        R["E2_agreement"][k][cname] = res
        log(f"E2 {k} {cname}: n={res['n']} agreement {res['argmax_agreement']:.4f} {res['argmax_agreement_ci95']} "
            f"flips {res['flips']} tier {res['tier_agreement']:.4f} max|dp| {res['prob_abs_diff_max']:.4f}")
    save_results()

per_row = held[["source_key", "out_path", "dataset", "label", "cohort", "pk"]].copy()
for k in VARIANTS:
    full = np.full((len(held), 3), np.nan)
    full[ROWSET[k]] = PRED[k]
    for c, lab in enumerate(VALID_LABELS):
        per_row[f"p_{lab}_{k}"] = full[:, c].round(6)
per_row.to_csv(OUT / "e2_per_row_probs.csv", index=False)


# ------------------------------------------------------------------------------- E2 server-condition cost probes
LAT = WORK / "lat"
LAT.mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)
ct_ctx = held[~held["dataset"].isin(MRI_PAIR_DATASETS) & held["has_context"].astype(bool)].index.to_numpy()
mri = held[held["dataset"].isin(MRI_PAIR_DATASETS)].index.to_numpy()
ct_any = held[~held["dataset"].isin(MRI_PAIR_DATASETS)].index.to_numpy()
requests = []
for kind, pool, n in (("ct_context", ct_ctx, N_LAT[0]), ("mri_dwi_adc", mri, N_LAT[1]), ("ct_single", ct_any, N_LAT[2])):
    for i in rng.choice(pool, n, replace=False):
        r = held.loc[i]
        roles = ({"main": r["dwi_out_path"], "adc": r["adc_out_path"]} if kind == "mri_dwi_adc" else
                 {"main": r["out_path"], "prev": r["prev_out_path"], "next": r["next_out_path"]} if kind == "ct_context" else
                 {"main": r["out_path"]})
        req = {"kind": kind, "modality": "MRI" if kind == "mri_dwi_adc" else "CT"}
        for role, p in roles.items():
            f = LAT / f"{len(requests):03d}_{role}{Path(p).suffix or '.png'}"
            f.write_bytes(raw_bytes(p))
            req[role] = str(f)
        requests.append(req)
order = np.random.default_rng(SEED + 1).permutation(len(requests))
requests = [requests[i] for i in order]
json.dump(requests, open(WORK / "requests.json", "w"))
log(f"latency set: {len(requests)} requests (raw stored files, as uploaded)")

PROBE = r'''
import json, resource, sys, time
def rss():
    for l in open("/proc/self/status"):
        if l.startswith("VmRSS"):
            return int(l.split()[1]) // 1024
out = {"rss_start_mb": rss()}
sys.path.insert(0, sys.argv[1])
t = time.time()
import numpy as np, onnxruntime
from PIL import Image
import inference as inf
p = inf.get_pipeline()
out["load_s"], out["rss_after_load_mb"] = round(time.time() - t, 3), rss()
out["peak_after_load_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
out["versions"] = {"onnxruntime": onnxruntime.__version__, "numpy": np.__version__}
reqs = json.load(open(sys.argv[2]))
e2e, kinds, xs = [], [], []
for q in reqs:
    t = time.time()
    ims = {k: Image.open(q[k]) for k in ("main", "prev", "next", "adc") if k in q}
    for im in ims.values():
        im.load()
    r = p.predict(q["modality"], ims["main"], ims.get("prev"), ims.get("next"), ims.get("adc"))
    e2e.append(time.time() - t)
    kinds.append(q["kind"])
    if len(xs) < 50:
        merged, _, _ = inf.build_input(q["modality"], ims["main"], ims.get("prev"), ims.get("next"), ims.get("adc"))
        xs.append(np.stack([inf._to_array(v) for v in inf._tta_views(merged)]))
mo = []
for x in xs:
    t = time.time()
    p.session.run(["logits", "gradcam"], {"image": x})
    mo.append(time.time() - t)
out["first_request_s"] = round(e2e[0], 3)
out["e2e_s"] = e2e
out["kinds"] = kinds
out["model_only_s"] = mo
out["rss_after_requests_mb"] = rss()
out["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
print("RESULT " + json.dumps(out), flush=True)
'''
(WORK / "probe.py").write_text(PROBE)
ENV1 = dict(os.environ, PYTHONPATH="/tmp/srvpkgs", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")


def run_probe(script, *args):
    p = subprocess.run([sys.executable, str(script), *map(str, args)], env=ENV1, capture_output=True, text=True)
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
    if line is None:
        return {"error": p.stderr[-1500:]}
    return json.loads(line[7:])


def summ(v):
    v = np.asarray(v, float)
    return {"n": int(len(v)), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": float(v.max()),
            "mean": float(v.mean())}


R["E2_server_cost"] = {}
for k, pth in VARIANTS.items():
    t = time.time()
    res = run_probe(WORK / "probe.py", server_dir(k.lower().replace("-", "_"), pth), WORK / "requests.json")
    if "error" not in res:
        e2e, kinds = np.array(res.pop("e2e_s")[3:]), np.array(res.pop("kinds")[3:])   # first 3 = warm-up
        res["e2e_after_warmup"] = {"all": summ(e2e), **{kd: summ(e2e[kinds == kd]) for kd in sorted(set(kinds))}}
        res["model_only_after_warmup"] = summ(res.pop("model_only_s")[3:])
    R["E2_server_cost"][k] = res
    log(f"E2 cost {k} ({time.time() - t:.0f}s): " + json.dumps({x: res.get(x) for x in ("load_s", "rss_after_load_mb", "peak_rss_mb", "first_request_s", "error")})
        + (f" e2e median {res['e2e_after_warmup']['all']['median']:.3f}s p95 {res['e2e_after_warmup']['all']['p95']:.3f}s; "
           f"model-only median {res['model_only_after_warmup']['median']:.3f}s" if "error" not in res else ""))
    save_results()

# --------------------------------------------------------------------------------------------------------- E3
t = time.time()
ck = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
R["weights_sha256"] = sha256(WEIGHTS)
assert R["weights_sha256"].startswith(WEIGHTS_SHA_PREFIX), R["weights_sha256"]
model = models.convnext_tiny(weights=None)
model.classifier[2] = nn.Linear(768, 3)
model.load_state_dict(ck["model_state"])
model.eval()


class CamHead(nn.Module):
    """The exported head (kernel neuroscan-q-onnx-export-check), unchanged: logits + closed-form stage-4 Grad-CAM."""

    def __init__(self, m):
        super().__init__()
        self.features, self.ln, self.fc = m.features, m.classifier[0], m.classifier[2]

    def forward(self, x):
        a = self.features(x)
        pooled = a.mean(dim=(2, 3))
        mu = pooled.mean(dim=1, keepdim=True)
        var = pooled.var(dim=1, unbiased=False, keepdim=True)
        s = torch.sqrt(var + self.ln.eps)
        xhat = (pooled - mu) / s
        logits = self.fc(xhat * self.ln.weight + self.ln.bias)
        v = self.fc.weight * self.ln.weight
        vb = v.unsqueeze(0)
        xh = xhat.unsqueeze(1)
        g = (vb - vb.mean(dim=2, keepdim=True) - xh * (vb * xh).mean(dim=2, keepdim=True)) / s.unsqueeze(1)
        w = g / (a.shape[2] * a.shape[3])
        return logits, torch.relu(torch.einsum("bck,bkij->bcij", w, a))


head = CamHead(model).eval()
acts = {}


def _cap(m, i, o):
    acts["a"] = o
    if o.requires_grad:
        o.register_hook(lambda gr: acts.__setitem__("g", gr))


hook = model.features[7].register_forward_hook(_cap)


def autograd_cams(x):
    """Standard Grad-CAM (Selvaraju et al.): channel weights = spatial mean of d logit_c / d A, one backward per class."""
    xx = torch.from_numpy(x).requires_grad_(True)
    lg = model(xx)
    out = []
    for c in range(3):
        model.zero_grad(set_to_none=True)
        lg[:, c].sum().backward(retain_graph=c < 2)
        w = acts["g"].mean(dim=(2, 3), keepdim=True)
        out.append(torch.relu((w * acts["a"]).sum(1)).detach().numpy())
    return np.stack(out, 1)


torch.set_num_threads(AGREE_THREADS)
ei = e3_rows["held_i"].to_numpy()
X3 = np.stack([srv._to_array(Image.fromarray(a)) for a in X_HELD[ei]])   # identity view
AG, CF = [], []
for st in range(0, len(ei), 20):
    xb = X3[st:st + 20]
    AG.append(autograd_cams(xb))
    with torch.no_grad():
        CF.append(head(torch.from_numpy(xb))[1].numpy())
AG, CF = np.concatenate(AG), np.concatenate(CF)
hook.remove()
OX = CAMS["FP32"][ei]


def norm(m):
    return m / (m.reshape(len(m), 3, -1).max(-1)[..., None, None] + 1e-8)


def cmp_maps(a, b):
    raw = np.abs(a - b).reshape(len(a), -1).max(1)
    rel = raw / (np.abs(a).reshape(len(a), -1).max(1) + 1e-12)
    nm = np.abs(norm(a) - norm(b)).reshape(len(a), -1).max(1)
    return {"raw_maxabs": float(raw.max()), "relative_maxabs": float(rel.max()), "normalised_maxabs": float(nm.max()),
            "normalised_maxabs_median": float(np.median(nm)), "share_rows_normalised_below_1e-5": float((nm < 1e-5).mean())}


R["E3_agreement"] = {"rows": int(len(ei)), "onnx_closed_form_vs_autograd": cmp_maps(AG, OX),
                     "torch_closed_form_vs_autograd": cmp_maps(AG, CF), "onnx_vs_torch_closed_form": cmp_maps(CF, OX)}
log(f"E3 agreement ({time.time() - t:.0f}s): {json.dumps(R['E3_agreement'])}")
pd.DataFrame({"held_i": ei, "dataset": e3_rows["dataset"], "label": e3_rows["label"],
              "normalised_maxabs_onnx_vs_autograd": np.abs(norm(AG) - norm(OX)).reshape(len(ei), -1).max(1),
              "normalised_maxabs_torch_cf_vs_autograd": np.abs(norm(AG) - norm(CF)).reshape(len(ei), -1).max(1)}
             ).to_csv(OUT / "e3_autograd_rows.csv", index=False)
save_results()

# E3 cost: fresh 1-thread processes, 100 rows, batch 1
np.save(WORK / "e3_x.npy", X3[:N_E3_TIMING])
torch.save(ck["model_state"], WORK / "state.pt")
logits_only = WORK / "fp32_logits_only.onnx"
onnx.utils.extract_model(str(ONNX_FP32), str(logits_only), ["image"], ["logits"])
fix_ir(logits_only)
TPROBE = r'''
import json, resource, sys, time
def rss():
    for l in open("/proc/self/status"):
        if l.startswith("VmRSS"):
            return int(l.split()[1]) // 1024
mode, work = sys.argv[1], sys.argv[2]
import numpy as np
x = np.load(f"{work}/e3_x.npy")
out = {"rss_after_inputs_mb": rss()}
ts = []
if mode.startswith("onnx"):
    import onnxruntime as ort
    o = ort.SessionOptions(); o.intra_op_num_threads = 1; o.inter_op_num_threads = 1; o.enable_cpu_mem_arena = False
    s = ort.InferenceSession(sys.argv[3], sess_options=o, providers=["CPUExecutionProvider"])
    names = ["logits"] if mode == "onnx_logits_only" else ["logits", "gradcam"]
    out["rss_after_load_mb"] = rss()
    for i in range(len(x)):
        t = time.time(); s.run(names, {"image": x[i:i + 1]}); ts.append(time.time() - t)
else:
    import torch, torch.nn as nn
    from torchvision import models
    torch.set_num_threads(1)
    exec(open(f"{work}/camhead.py").read())
    m = models.convnext_tiny(weights=None); m.classifier[2] = nn.Linear(768, 3)
    m.load_state_dict(torch.load(f"{work}/state.pt", map_location="cpu")); m.eval()
    h = CamHead(m).eval()
    acts = {}
    def cap(mod, i, o):
        acts["a"] = o
        if o.requires_grad:
            o.register_hook(lambda g: acts.__setitem__("g", g))
    m.features[7].register_forward_hook(cap)
    out["rss_after_load_mb"] = rss()
    for i in range(len(x)):
        xb = torch.from_numpy(x[i:i + 1])
        t = time.time()
        if mode == "torch_forward":
            with torch.no_grad(): m(xb)
        elif mode == "torch_closed_form":
            with torch.no_grad(): h(xb)
        else:
            lg = m(xb.requires_grad_(True))
            classes = [int(lg.argmax())] if mode == "torch_autograd_pred" else [0, 1, 2]
            for j, c in enumerate(classes):
                m.zero_grad(set_to_none=True)
                lg[0, c].backward(retain_graph=j < len(classes) - 1)
                w = acts["g"].mean(dim=(2, 3), keepdim=True)
                torch.relu((w * acts["a"]).sum(1))
        ts.append(time.time() - t)
out["seconds"] = ts[3:]
out["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
print("RESULT " + json.dumps(out), flush=True)
'''
CAMHEAD_SRC = '''
class CamHead(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.features, self.ln, self.fc = m.features, m.classifier[0], m.classifier[2]

    def forward(self, x):
        a = self.features(x)
        pooled = a.mean(dim=(2, 3))
        mu = pooled.mean(dim=1, keepdim=True)
        var = pooled.var(dim=1, unbiased=False, keepdim=True)
        s = torch.sqrt(var + self.ln.eps)
        xhat = (pooled - mu) / s
        logits = self.fc(xhat * self.ln.weight + self.ln.bias)
        v = self.fc.weight * self.ln.weight
        vb = v.unsqueeze(0)
        xh = xhat.unsqueeze(1)
        g = (vb - vb.mean(dim=2, keepdim=True) - xh * (vb * xh).mean(dim=2, keepdim=True)) / s.unsqueeze(1)
        w = g / (a.shape[2] * a.shape[3])
        return logits, torch.relu(torch.einsum("bck,bkij->bcij", w, a))
'''
(WORK / "camhead.py").write_text(CAMHEAD_SRC)
(WORK / "tprobe.py").write_text(TPROBE)
ENV_TORCH = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")   # Kaggle packages (torch is not on the server)
R["E3_cost"] = {}
for mode, extra, env in (("torch_forward", [], ENV_TORCH), ("torch_closed_form", [], ENV_TORCH),
                         ("torch_autograd_pred", [], ENV_TORCH), ("torch_autograd_all3", [], ENV_TORCH),
                         ("onnx_logits_only", [logits_only], ENV1), ("onnx_logits_gradcam", [ONNX_FP32], ENV1)):
    t = time.time()
    p = subprocess.run([sys.executable, str(WORK / "tprobe.py"), mode, str(WORK), *map(str, extra)], env=env,
                       capture_output=True, text=True)
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
    res = json.loads(line[7:]) if line else {"error": p.stderr[-1500:]}
    if "seconds" in res:
        res["seconds"] = summ(res["seconds"])
    R["E3_cost"][mode] = res
    log(f"E3 cost {mode} ({time.time() - t:.0f}s): {json.dumps(res)[:400]}")
    save_results()

# E3: maps under compression (all rows each variant ran on), FP32's predicted class; outline under own prediction


def display_region(cam7):
    return np.asarray(srv.ThreeClassPipeline._display_cam(cam7), dtype=np.float32) / 255.0 > 0.6


R["E3_maps_under_compression"] = {}
map_rows = []
for k in VARIANTS:
    if k == "FP32":
        continue
    rows = ROWSET[k]
    pa, pb = PRED["FP32"][rows], PRED[k]
    ca, cb = CAMS["FP32"][rows], CAMS[k]
    cls_a, cls_b = pa.argmax(1), pb.argmax(1)
    mri = held["dataset"].isin(MRI_PAIR_DATASETS).to_numpy()[rows]
    rec = []
    for i in range(len(rows)):
        a, b = ca[i, cls_a[i]].ravel(), cb[i, cls_a[i]].ravel()
        r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
        ra, rb = display_region(ca[i, cls_a[i]]), display_region(cb[i, cls_a[i]])
        un = (ra | rb).sum()
        out_a = VALID_LABELS[cls_a[i]] in ("ischemic", "hemorrhagic") and not (mri[i] and cls_a[i] == 2)
        out_b = VALID_LABELS[cls_b[i]] in ("ischemic", "hemorrhagic") and not (mri[i] and cls_b[i] == 2)
        rec.append({"held_i": int(rows[i]), "variant": k, "pearson_r": r, "same_peak_cell": int(a.argmax() == b.argmax()),
                    "display_iou": float((ra & rb).sum() / un) if un else np.nan,
                    "outline_fp32": int(out_a), "outline_variant": int(out_b)})
    d = pd.DataFrame(rec)
    d["cohort"] = held["cohort"].to_numpy()[d["held_i"]]
    map_rows.append(d)
    q = lambda s: {"median": float(s.median()), "q1": float(s.quantile(.25)), "q3": float(s.quantile(.75)),
                   "mean": float(s.mean()), "n": int(s.notna().sum())}
    R["E3_maps_under_compression"][k] = {
        c: {"pearson_r": q(g["pearson_r"]), "same_peak_cell": float(g["same_peak_cell"].mean()),
            "display_iou": q(g["display_iou"]),
            "outline_appears": int(((g["outline_fp32"] == 0) & (g["outline_variant"] == 1)).sum()),
            "outline_disappears": int(((g["outline_fp32"] == 1) & (g["outline_variant"] == 0)).sum())}
        for c, g in [("all", d), *d.groupby("cohort")]}
    log(f"E3 maps {k}: {json.dumps(R['E3_maps_under_compression'][k]['all'])}")
if map_rows:
    pd.concat(map_rows).to_csv(OUT / "e3_maps_summary.csv", index=False)
save_results()
log("Done.")
