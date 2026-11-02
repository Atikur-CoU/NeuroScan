"""
NeuroScan paper 2 -- static INT8 with other calibration methods (2026-11-02). CPU only; no training; no new
classification result. Plan: paper/second_paper/PLAN_static_calibration.md (settings fixed before this run).

Why: the paper's INT8-static copy (min-max calibration) changed 332 of 10,602 held-out predictions, against 32 for
INT8-dynamic. This kernel tests whether the usual alternatives to min-max fix static quantization for the served
ConvNeXt-Tiny. The held-out slices are only scored, never used to choose anything.

Copies (identical to the paper's INT8-static except for the stated change): quant_pre_process, then quantize_static
with QDQ, per-channel QInt8 weights, QUInt8 activations, MatMul + Conv, the 26 head / Grad-CAM nodes left in FP32, the
same 512 training slices (equal per dataset x label, seed 123, identity view), CalibStridedMinMax = 8 (ORT calls
collect_data per chunk of 8; min-max merges ranges, the histogram methods merge histograms), ONNX Runtime 1.20.1.
  S0  min-max (control): rebuilt to check that this pipeline reproduces the paper's INT8-static.
  S1  percentile: CalibrationMethod.Percentile with ORT's defaults (99.999th percentile, 2,048 bins, symmetric).
  S2  entropy: CalibrationMethod.Entropy with ORT's defaults (128 bins, 128 quantized bins).
  S3  min-max with the first layers in FP32: every Conv with a stride other than 1 left unquantized, which in
      ConvNeXt-Tiny is the stem (4x4/4) and the three downsampling convolutions (2x2/2); exactly 4 are required.
Each copy is built in a child process; its memory is watched and it is stopped above 24 GB (reported as failed).

Control: S0 must match the saved INT8-static outputs of neuroscan-p2-compression-gradcam on the first 256 held-out
slices (max probability difference <= 1e-4), else the kernel stops: the comparison would not be like for like.
Dry run: S1-S3 on 64 held-out slices, per request (one slice, 4 views) and batched (16 slices): checks finiteness,
batch independence (<= 1e-4) and time. The full run starts only if the projected time for S1-S3 is under 360 min.
Full run: all 10,602 held-out slices, batches of 16 slices (4 threads), as the compression kernel ran INT8-static;
a copy that is not batch independent is run per request instead.

Scoring (the compression kernel's code, unchanged): agreement with the saved FP32 outputs under the deployed rule
(plain argmax of the 4-view mean) with the patient-cluster bootstrap (2,000), alert tiers, paired metric deltas, by
cohort and source; maps under compression (Pearson r, same peak cell, display-region IoU, outline changes); changed
slices by transition and truth with their FP32 margins; size and single-thread model-call time (50 requests of
4 views after 3 warm-up) for FP32, the paper's INT8-static and S1-S3. Files by direct path (no listing, no
extraction); per-step time and memory logged.
"""
import hashlib
import io
import json
import subprocess
import sys
import threading
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


t = time.time()
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime==1.20.1", "onnx==1.18.0"],
                   capture_output=True, text=True)
log(f"pip onnxruntime==1.20.1 onnx==1.18.0: exit {r.returncode} ({time.time() - t:.0f}s) {r.stderr[-300:]}")
if r.returncode:
    sys.exit(1)

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import pandas as pd  # noqa: E402
from onnxruntime.quantization.shape_inference import quant_pre_process  # noqa: E402
from PIL import Image  # noqa: E402

SEED = 123
VALID_LABELS = ("normal", "ischemic", "hemorrhagic")
LABEL_TO_IDX = {k: i for i, k in enumerate(VALID_LABELS)}
TARGET_SIZE = (224, 224)
MRI_PAIR_DATASETS = ("ISLES2022_dwi_adc", "SOOP_dwi_adc")
EXCLUDED_DATASETS = ("MRI_Images",)
FP32_SHA = "4937ab381ee3e7c29d5ab18ea761a8c19ddffb6488210fb6030511a8041502eb"
PAPER_STATIC_SHA = "35d8a4007b813f8fb6eca6c10078823fc5b69a654314c36d76c3412a38b02a46"
N_CALIB, CALIB_CHUNK = 512, 8
N_BOOT = 2000
N_CONTROL, N_DRY, N_LAT = 256, 64, 50
THREADS, ROWS_PER_RUN, WORKERS = 4, 16, 4
MEM_LIMIT_KB = 24_000_000
BUILD_TIMEOUT_S = 3600
PROJECTED_LIMIT_MIN = 360
MAX_IR = 10
KEYS = ["accuracy", "balanced_accuracy", "macro_f1_present_classes", "normal_recall", "ischemic_recall", "ischemic_precision",
        "hemorrhagic_recall", "hemorrhagic_precision", "ischemic_auroc", "hemorrhagic_auroc", "macro_ovr_auroc"]
COPIES = {"S0-minmax": ("MinMax", False), "S1-percentile": ("Percentile", False), "S2-entropy": ("Entropy", False),
          "S3-minmax-first-fp32": ("MinMax", True)}

OUT = Path("/kaggle/working/output")
MODELS_OUT = OUT / "models"
WORK = Path("/tmp/p2s")
for d in (OUT, MODELS_OUT, WORK):
    d.mkdir(parents=True, exist_ok=True)
R = {"note": __doc__.strip().splitlines()[0], "python": sys.version.split()[0],
     "packages": {"onnxruntime": ort.__version__, "onnx": onnx.__version__, "numpy": np.__version__},
     "settings": {"calibration_rows": N_CALIB, "calib_chunk": CALIB_CHUNK, "copies": {k: {"method": m, "first_layers_fp32": f}
                                                                                       for k, (m, f) in COPIES.items()},
                  "ort_defaults": {"Percentile": "num_bins 2048, percentile 99.999, symmetric True",
                                   "Entropy": "num_bins 128, num_quantized_bins 128, symmetric False"},
                  "memory_limit_gb": MEM_LIMIT_KB / 1e6, "projected_limit_min": PROJECTED_LIMIT_MIN}}


def save():
    (OUT / "static_calibration_results.json").write_text(json.dumps(jsonable(R), indent=2))


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


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fix_ir(path):
    """Keeps a saved model loadable by the server's onnxruntime (IR <= 10)."""
    m = onnx.load(str(path))
    if m.ir_version > MAX_IR:
        m.ir_version = MAX_IR
        onnx.save(m, str(path))


# ---------------------------------------------------------------------------------------------------------- inputs
t = time.time()
SPLIT_DIR = ds_root("neuroscan-phase-p-patient-split-v3")
PHASE_B = ds_root("neuroscan-phase-b-images")
RSNA_J = ds_root("neuroscan-phase-j-rsna-images")
RSNA_L = ds_root("neuroscan-phase-l-rsna-more-images")
AISD_K = ds_root("neuroscan-phase-k-aisd-images")
CPAISD_V2_IMG = ds_root("neuroscan-phase-n-cpaisd-v2-images")
SAVED = ds_root("neuroscan-q-round30as123-preds")
CG = "neuroscan-p2-compression-gradcam"
ONNX_FP32 = nb_file("neuroscan-q-onnx-export-check", "neuroscan_r30a_s123.onnx")
PAPER_STATIC = nb_file(CG, "models/neuroscan_r30a_s123_int8_static.onnx")
PARTIAL = {k: np.load(nb_file(CG, f"partial/{k}.npz")) for k in ("FP32", "INT8-static")}
PER_ROW = pd.read_csv(nb_file(CG, "e2_per_row_probs.csv"))
ZIPS = {"AISD/normal/": nb_file("neuroscan-q-aisd-normals-prep", "aisd_normal_images.zip"),
        "RSNA_ICH/normal/": nb_file("neuroscan-q-rsna-normals-prep", "rsna_normal_images.zip"),
        "SOOP/": nb_file("neuroscan-q-soop-prep", "soop_images.zip")}
CSV = {"aisd_normal": nb_file("neuroscan-q-aisd-normals-prep", "aisd_normal_rows.csv"),
       "rsna_normal": nb_file("neuroscan-q-rsna-normals-prep", "rsna_normal_rows.csv"),
       "soop": nb_file("neuroscan-q-soop-prep", "soop_rows.csv")}
R["fp32_sha256"], R["paper_static_sha256"] = sha256(ONNX_FP32), sha256(PAPER_STATIC)
assert R["fp32_sha256"] == FP32_SHA, R["fp32_sha256"]
assert R["paper_static_sha256"] == PAPER_STATIC_SHA, R["paper_static_sha256"]
log(f"inputs resolved and checked ({time.time() - t:.1f}s)")

# The server's inference.py, verbatim (inserted by build_kernel.py): preprocessing, TTA views, tiers, display rule.
INFERENCE_PY = "__INFERENCE_PY__"
INFERENCE_PY_SHA = "__INFERENCE_PY_SHA__"
assert hashlib.sha256(INFERENCE_PY.encode("utf-8")).hexdigest() == INFERENCE_PY_SHA
R["inference_py_sha256"] = INFERENCE_PY_SHA
SRV = Path("/tmp/srv")
SRV.mkdir(exist_ok=True)
(SRV / "inference.py").write_text(INFERENCE_PY)
sys.path.insert(0, str(SRV))
import inference as srv  # noqa: E402

# --------------------------------------------- graph checks first (need only the model file; fail before decoding)
t = time.time()
PRE_MODEL = WORK / "fp32_pre.onnx"
quant_pre_process(str(ONNX_FP32), str(PRE_MODEL))
fix_ir(PRE_MODEL)
pm = onnx.load(str(PRE_MODEL))


def head_nodes(model):
    """Every node downstream of ConvNeXt stage 4's output (the Einsum's activation input): compression kernel."""
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
    return sorted(seen)


def strided_convs(model):
    """Conv nodes with any stride other than 1: ConvNeXt-Tiny's stem and its three downsampling convolutions."""
    out = []
    for n in model.graph.node:
        if n.op_type == "Conv":
            st = next((list(a.ints) for a in n.attribute if a.name == "strides"), [1, 1])
            ks = next((list(a.ints) for a in n.attribute if a.name == "kernel_shape"), None)
            if any(s != 1 for s in st):
                out.append((n.name, st, ks))
    return out


HEAD = head_nodes(pm)
assert len(HEAD) == 26, len(HEAD)
FIRST = strided_convs(pm)
R["excluded"] = {"head_nodes": len(HEAD), "first_layers": [{"node": n, "strides": s, "kernel": k} for n, s, k in FIRST]}
assert len(FIRST) == 4, f"expected the stem and 3 downsampling convolutions, found {FIRST}"
log(f"quant_pre_process ({time.time() - t:.0f}s); head nodes {len(HEAD)}; strided convs {FIRST}")
del pm

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
held = []
for saved, man in ((pd.read_csv(SAVED / "predictions_test.csv"), test_manifest),
                   (pd.read_csv(SAVED / "predictions_soop_external.csv"), soop[soop["split"] == "external"])):
    held.append(saved[keys].merge(man[keys + in_cols + ["dataset"]], on=keys, how="left", validate="one_to_one"))
held = pd.concat(held, ignore_index=True)
assert len(held) == len(PER_ROW) == 10602
assert (held["out_path"].to_numpy() == PER_ROW["out_path"].to_numpy()).all(), "row order differs from the saved run"
for c in ("label", "cohort", "pk"):
    held[c] = PER_ROW[c].to_numpy()


def equal_strata(df, n, seed):
    """n rows, as equal as possible per dataset x label (all of a small stratum), fixed seed (compression kernel)."""
    g = df.groupby(["dataset", "label"])
    sizes, left, quota = g.size().sort_values(), n, {}
    for i, (k, s) in enumerate(sizes.items()):
        quota[k] = min(int(s), left // (len(sizes) - i))
        left -= quota[k]
    return pd.concat([grp.sample(n=quota[k], random_state=seed) for k, grp in g]).reset_index(drop=True)


calib_pool = pd.concat([keep(split_train), aisd_n[aisd_n["split"] == "train"], rsna_n[rsna_n["split"] == "train"],
                        soop[soop["split"] == "train"]], ignore_index=True)
calib = equal_strata(calib_pool, N_CALIB, SEED)
assert len(calib) == N_CALIB
R["rows"] = {"held_out_by_cohort": held["cohort"].value_counts().to_dict(),
             "calibration_by_stratum": {f"{a}|{b}": int(v) for (a, b), v in calib.groupby(["dataset", "label"]).size().items()}}
log(f"rows ({time.time() - t:.1f}s): held-out {len(held)} in the saved order, calibration {len(calib)}")

# --------------------------------------------------------------------------------------- files by direct path
ZF, MEMBERS, LOCK, READS = {}, {}, threading.Lock(), {}
for prefix, zp in ZIPS.items():
    t = time.time()
    ZF[prefix] = zipfile.ZipFile(zp)
    MEMBERS[prefix] = set(ZF[prefix].namelist())
    log(f"zip {zp.name}: central directory read, {len(MEMBERS[prefix])} members ({time.time() - t:.1f}s)")


def raw(p):
    p = str(p)
    with LOCK:
        READS[p.split("/")[0]] = READS.get(p.split("/")[0], 0) + 1
        for prefix, zf in ZF.items():
            if p.startswith(prefix):
                assert p in MEMBERS[prefix], f"{p} not in {prefix} zip"
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
    """Loader.merged, unchanged."""
    if row["dataset"] in MRI_PAIR_DATASETS:
        dwi, adc = gray(row["dwi_out_path"]), gray(row["adc_out_path"])
        return Image.merge("RGB", (dwi, adc, dwi))
    s = gray(row["out_path"])
    if bool(row["has_context"]):
        return Image.merge("RGB", (gray(row["prev_out_path"]), s, gray(row["next_out_path"])))
    return Image.merge("RGB", (s, s, s))


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
R["file_reads_by_source"] = dict(READS)
log(f"file reads by source: {READS}")
# Calibration inputs exactly as the compression kernel's Calib reader built them (identity view, server normalisation).
CAL = np.stack([srv._to_array(Image.fromarray(a))[None] for a in X_CAL]).astype(np.float32)
np.save(WORK / "cal.npy", CAL)
del CAL
save()

# ---------------------------------------------------------------------------------------------------- build copies
BUILDER = r'''
import json, sys
import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
pre, out, method, excl_path, cal_path, chunk = sys.argv[1:7]
X = np.load(cal_path)
excl = json.load(open(excl_path))
class Calib(CalibrationDataReader):
    def __init__(self, x):
        self.x = x
        self.set_range(0, len(x))
    def __len__(self):
        return len(self.x)
    def set_range(self, start_index, end_index):
        self.it = iter(range(start_index, end_index))
    def get_next(self):
        i = next(self.it, None)
        return None if i is None else {"image": self.x[i]}
quantize_static(pre, out, Calib(X), quant_format=QuantFormat.QDQ, per_channel=True, weight_type=QuantType.QInt8,
                activation_type=QuantType.QUInt8, op_types_to_quantize=["MatMul", "Conv"], nodes_to_exclude=excl,
                calibrate_method=getattr(CalibrationMethod, method), extra_options={"CalibStridedMinMax": int(chunk)})
m = onnx.load(out)
if m.ir_version > 10:
    m.ir_version = 10
    onnx.save(m, out)
print("BUILT", flush=True)
'''
(WORK / "builder.py").write_text(BUILDER)


def proc_rss_kb(pid):
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS"):
                return int(line.split()[1])
    except OSError:
        pass
    return 0


def build(name, method, first_fp32):
    out = MODELS_OUT / f"neuroscan_r30a_s123_int8_static_{name.split('-', 1)[1].replace('-', '_')}.onnx"
    excl = HEAD + ([n for n, _, _ in FIRST] if first_fp32 else [])
    (WORK / f"excl_{name}.json").write_text(json.dumps(excl))
    t0, peak = time.time(), 0
    p = subprocess.Popen([sys.executable, str(WORK / "builder.py"), str(PRE_MODEL), str(out), method,
                          str(WORK / f"excl_{name}.json"), str(WORK / "cal.npy"), str(CALIB_CHUNK)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines, reader = [], threading.Thread(target=lambda: lines.extend(p.stdout), daemon=True)
    reader.start()
    err = None
    while p.poll() is None:
        peak = max(peak, proc_rss_kb(p.pid))
        if peak > MEM_LIMIT_KB:
            p.kill()
            err = f"stopped: child memory above {MEM_LIMIT_KB / 1e6:.0f} GB"
        elif time.time() - t0 > BUILD_TIMEOUT_S:
            p.kill()
            err = f"stopped: build over {BUILD_TIMEOUT_S / 60:.0f} min"
        time.sleep(1)
    reader.join(timeout=10)
    if err is None and (p.returncode != 0 or not any(l.startswith("BUILT") for l in lines)):
        err = f"exit {p.returncode}: {''.join(lines)[-1500:]}"
    info = {"method": method, "first_layers_fp32": first_fp32, "excluded_nodes": len(excl),
            "build_s": round(time.time() - t0, 1), "child_peak_rss_gb": round(peak / 1e6, 2), "error": err}
    if err is None:
        info.update(file=out.name, mb=round(out.stat().st_size / 1e6, 2), sha256=sha256(out))
    log(f"built {name}: {json.dumps(info)}")
    return (out if err is None else None), info


R["copies"], BUILT = {}, {}
for name, (method, ff) in COPIES.items():
    path, R["copies"][name] = build(name, method, ff)
    if path is not None:
        BUILT[name] = path
    save()
R["copies"]["S0-minmax"]["same_file_as_paper"] = R["copies"]["S0-minmax"].get("sha256") == PAPER_STATIC_SHA


# ------------------------------------------------------------------------------------------------------- inference
def views_batch(arrs):
    """The server's 4-view TTA and normalisation, exactly (inference._tta_views, inference._to_array)."""
    return np.stack([srv._to_array(v) for a in arrs for v in srv._tta_views(Image.fromarray(a))])


def session(path, threads):
    o = ort.SessionOptions()
    o.intra_op_num_threads, o.inter_op_num_threads = threads, 1
    return ort.InferenceSession(str(path), sess_options=o, providers=["CPUExecutionProvider"])


def batched(path, idx, tag):
    """Compression kernel's run_all: 16 slices (64 views) per run, 4 threads."""
    s = session(path, THREADS)
    probs, cams, t0 = np.zeros((len(idx), 3)), np.zeros((len(idx), 3, 7, 7), dtype=np.float32), time.time()
    for st in range(0, len(idx), ROWS_PER_RUN):
        b = idx[st:st + ROWS_PER_RUN]
        lg, cm = s.run(["logits", "gradcam"], {"image": views_batch(X_HELD[b])})
        probs[st:st + len(b)] = srv._softmax(lg.astype(np.float64)).reshape(len(b), 4, 3).mean(1)
        cams[st:st + len(b)] = cm.reshape(len(b), 4, 3, 7, 7)[:, 0]
        done, el = st + len(b), time.time() - t0
        if len(idx) > 1000 and (done % 2048 == 0 or done == len(idx)):
            log(f"  [{tag}] {done}/{len(idx)} ({el:.0f}s; projected {el / done * len(idx) / 60:.0f} min)")
    return probs, cams, time.time() - t0


def per_request(path, idx, tag):
    """One run per slice, its 4 views as the batch (the server's request), 4 workers with 1 thread each."""
    probs, cams = np.zeros((len(idx), 3)), np.zeros((len(idx), 3, 7, 7), dtype=np.float32)
    local, t0, done = threading.local(), time.time(), [0]

    def one(j):
        if not hasattr(local, "s"):
            local.s = session(path, 1)
        lg, cm = local.s.run(["logits", "gradcam"], {"image": views_batch(X_HELD[idx[j]:idx[j] + 1])})
        probs[j] = srv._softmax(lg.astype(np.float64)).mean(0)
        cams[j] = cm[0]
        with LOCK:
            done[0] += 1
            if len(idx) > 1000 and (done[0] % 1000 == 0 or done[0] == len(idx)):
                el = time.time() - t0
                log(f"  [{tag}] {done[0]}/{len(idx)} ({el:.0f}s; projected {el / done[0] * len(idx) / 60:.0f} min)")

    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(one, range(len(idx))))
    return probs, cams, time.time() - t0


# Control: S0 must reproduce the paper's INT8-static outputs.
if "S0-minmax" not in BUILT:
    print("[ERROR] the min-max control could not be built; nothing else is comparable")
    sys.exit(1)
ctl = np.arange(N_CONTROL)
p0, c0, _ = batched(BUILT["S0-minmax"], ctl, "S0 control")
dp = float(np.abs(p0 - PARTIAL["INT8-static"]["probs"][ctl]).max())
dc = float(np.abs(c0 - PARTIAL["INT8-static"]["cams"][ctl]).max() / np.abs(PARTIAL["INT8-static"]["cams"][ctl]).max())
R["control"] = {"rows": N_CONTROL, "max_prob_diff_vs_paper_static": dp, "max_rel_map_diff": dc,
                "same_file_as_paper": R["copies"]["S0-minmax"]["same_file_as_paper"]}
log(f"control S0 vs the paper's INT8-static on {N_CONTROL} rows: max prob diff {dp:.2e}, map {dc:.2e}, "
    f"same file {R['control']['same_file_as_paper']}")
save()
if dp > 1e-4:
    print("[ERROR] the rebuilt min-max copy does not reproduce the paper's INT8-static; stopping")
    sys.exit(1)

# Dry run: finiteness, batch independence and time for S1-S3.
NEW = [k for k in BUILT if k != "S0-minmax"]
dry, R["dry_run"] = np.arange(N_DRY), {}
projected = 0.0
for k in NEW:
    try:
        pb, cb, tb = batched(BUILT[k], dry, f"{k} dry batched")
        pr, cr, _ = per_request(BUILT[k], dry, f"{k} dry per request")
    except Exception as e:  # noqa: BLE001
        R["copies"][k]["error"] = f"inference failed: {type(e).__name__}: {str(e)[:400]}"
        log(f"{k}: {R['copies'][k]['error']}")
        continue
    finite = bool(np.isfinite(pb).all() and np.isfinite(cb).all())
    d = float(np.abs(pb - pr).max())
    proj = tb / N_DRY * len(held) / 60
    R["dry_run"][k] = {"finite": finite, "batched_vs_per_request_max_prob_diff": d, "batch_independent": d <= 1e-4,
                       "batched_s_per_slice": tb / N_DRY, "projected_full_min": proj}
    log(f"dry run {k}: {json.dumps(R['dry_run'][k])}")
    if finite:
        projected += proj
save()
R["dry_run_projected_total_min"] = projected
log(f"dry run: projected full-run time for {NEW}: {projected:.0f} min (limit {PROJECTED_LIMIT_MIN})")
if projected > PROJECTED_LIMIT_MIN:
    print(f"[ERROR] projected time {projected:.0f} min is over {PROJECTED_LIMIT_MIN} min; stopping after the dry run")
    save()
    sys.exit(1)

# Full run.
ALL = np.arange(len(held))
PRED, CAMS = {}, {}
for k in NEW:
    if not R["dry_run"].get(k, {}).get("finite"):
        continue
    t = time.time()
    if R["dry_run"][k]["batch_independent"]:
        PRED[k], CAMS[k], _ = batched(BUILT[k], ALL, k)
        mode = "batched (16 slices)"
    else:
        PRED[k], CAMS[k], _ = per_request(BUILT[k], ALL, k)
        mode = "per request"
    R["copies"][k]["run"] = {"mode": mode, "minutes": round((time.time() - t) / 60, 1)}
    np.savez_compressed(OUT / f"{k}.npz", probs=PRED[k], cams=CAMS[k], rows=ALL)
    log(f"{k}: full run {mode} done ({R['copies'][k]['run']['minutes']} min)")
    save()

PF, CF = PARTIAL["FP32"]["probs"], PARTIAL["FP32"]["cams"]
assert (PARTIAL["FP32"]["rows"] == ALL).all()
per_row = PER_ROW[["out_path", "dataset", "label", "cohort", "pk"]].copy()
for k in ["FP32", "INT8-static"] + list(PRED):
    P = PARTIAL[k]["probs"] if k in PARTIAL else PRED[k]
    for c, lab in enumerate(VALID_LABELS):
        per_row[f"p_{lab}_{k}"] = P[:, c].round(6)
per_row.to_csv(OUT / "per_row_probs.csv", index=False)


# ----------------------------------------------------------------- scoring (compression kernel's compare, unchanged)
def decide(p):
    return p.argmax(1)


def tier(p, mri):
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


def display_region(cam7):
    return np.asarray(srv.ThreeClassPipeline._display_cam(cam7), dtype=np.float32) / 255.0 > 0.6


def maps(PS, CS):
    cls_a, cls_b = PF.argmax(1), PS.argmax(1)
    mri = held["dataset"].isin(MRI_PAIR_DATASETS).to_numpy()
    rec = []
    for i in range(len(ALL)):
        a, b = CF[i, cls_a[i]].ravel(), CS[i, cls_a[i]].ravel()
        r_ = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
        ra, rb = display_region(CF[i, cls_a[i]]), display_region(CS[i, cls_a[i]])
        un = (ra | rb).sum()
        out_a = VALID_LABELS[cls_a[i]] in ("ischemic", "hemorrhagic") and not (mri[i] and cls_a[i] == 2)
        out_b = VALID_LABELS[cls_b[i]] in ("ischemic", "hemorrhagic") and not (mri[i] and cls_b[i] == 2)
        rec.append({"pearson_r": r_, "same_peak_cell": int(a.argmax() == b.argmax()),
                    "display_iou": float((ra & rb).sum() / un) if un else np.nan,
                    "outline_fp32": int(out_a), "outline_variant": int(out_b)})
    d = pd.DataFrame(rec)
    d["cohort"] = held["cohort"].to_numpy()
    q = lambda s: {"median": float(s.median()), "q1": float(s.quantile(.25)), "q3": float(s.quantile(.75)),
                   "mean": float(s.mean()), "n": int(s.notna().sum())}
    return {c: {"pearson_r": q(g["pearson_r"]), "same_peak_cell": float(g["same_peak_cell"].mean()),
                "display_iou": q(g["display_iou"]),
                "outline_appears": int(((g["outline_fp32"] == 0) & (g["outline_variant"] == 1)).sum()),
                "outline_disappears": int(((g["outline_fp32"] == 1) & (g["outline_variant"] == 0)).sum())}
            for c, g in [("all", d), *d.groupby("cohort")]}


def transitions(PS):
    """Changed slices over all held-out rows: transition, truth and FP32 margin (as Table 5 of the paper)."""
    y = held["label"].map(LABEL_TO_IDX).to_numpy()
    a, b = PF.argmax(1), PS.argmax(1)
    ch = a != b
    srt = np.sort(PF, 1)
    margin = srt[:, 2] - srt[:, 1]
    kind = np.where(a == y, "correct_to_wrong", np.where(b == y, "wrong_to_correct", "wrong_to_wrong"))
    return {"changed": int(ch.sum()), "by_kind": pd.Series(kind[ch]).value_counts().to_dict(),
            "correct_to_wrong_by_truth": pd.Series(held["label"].to_numpy()[ch & (a == y)]).value_counts().to_dict(),
            "fp32_margin_changed": {"median": float(np.median(margin[ch])) if ch.any() else None,
                                    "max": float(margin[ch].max()) if ch.any() else None},
            "near_tie_share": float((margin[ch] < 0.1).mean()) if ch.any() else None,
            "fp32_margin_all_median": float(np.median(margin))}


COHORTS = {"pooled_internal_test": held["cohort"] != "SOOP", "internal_test": held["cohort"] == "internal_test",
           "CPAISD": held["cohort"] == "CPAISD", "SOOP_holdout": held["cohort"] == "SOOP"}
R["E2_agreement"], R["E3_maps"], R["transitions"] = {}, {}, {}
for vi, k in enumerate(PRED):
    R["E2_agreement"][k] = {}
    for ci_, (cname, mask) in enumerate(COHORTS.items()):
        sel = np.where(mask.to_numpy())[0]
        res = compare(held.iloc[sel].reset_index(drop=True), PF[sel], PRED[k][sel], 911 + 10 * vi + ci_)
        R["E2_agreement"][k][cname] = res
        d = res["secondary_metric_deltas_variant_minus_fp32"].get("accuracy", {})
        log(f"E2 {k} {cname}: agreement {res['argmax_agreement']:.4f} {res['argmax_agreement_ci95']} flips {res['flips']} "
            f"tier {res['tier_agreement']:.4f} accuracy change {d.get('diff', float('nan')):+.4f} {d.get('ci95')}")
    R["E3_maps"][k] = maps(PRED[k], CAMS[k])
    R["transitions"][k] = transitions(PRED[k])
    log(f"E3 {k}: {json.dumps(R['E3_maps'][k]['all'])}; transitions {json.dumps(R['transitions'][k])}")
    save()

# --------------------------------------------------------------------------------- size and model-call time
lat = np.arange(N_LAT + 3)
R["cost"] = {}
for k, pth in [("FP32", ONNX_FP32), ("INT8-static (paper)", PAPER_STATIC)] + [(k, BUILT[k]) for k in PRED]:
    s = session(pth, 1)
    ts = []
    for i in lat:
        x = views_batch(X_HELD[i:i + 1])
        t0 = time.time()
        s.run(["logits", "gradcam"], {"image": x})
        ts.append(time.time() - t0)
    ts = np.array(ts[3:])
    R["cost"][k] = {"mb": round(Path(pth).stat().st_size / 1e6, 2), "model_call_s_median": float(np.median(ts)),
                    "model_call_s_p95": float(np.percentile(ts, 95)), "n": int(len(ts))}
    log(f"cost {k}: {json.dumps(R['cost'][k])}")
    save()
log("Done.")
