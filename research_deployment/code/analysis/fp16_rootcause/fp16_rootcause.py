"""
NeuroScan paper 2 -- K3, JTCSST revision (R2-6): why did the two FP16 copies not load, and can a 16-bit copy be served?
CPU only; no stroke data. Input: the served FP32 ONNX file (output of kernel neuroscan-q-onnx-export-check; SHA-256
checked). Test inputs: 32 deterministic head-like phantoms (the export-check generator), identity view.

1. Root cause. Rebuild the paper's two copies exactly (onnxconverter-common 1.16.0, keep_io_types=True; plain, and with
   LayerNormalization/ReduceMean/Pow/Sqrt blocked), load them in ONNX Runtime 1.20.1 (the server's version), and trace
   each node named in the load error back through the FP32 graph: op type, attributes, inputs and their producers, so
   the failing node can be attributed to a part of the model (backbone, LayerNorm, classifier head, in-graph Grad-CAM).
2. Variants (each: does it load; finite outputs; agreement with FP32 on the phantoms; Grad-CAM difference; size;
   one-thread median time):
     A  plain conversion (paper)                    B  LayerNorm kept in FP32 (paper)
     C  plain, logits-only subgraph (Grad-CAM part cut away with onnx.utils.extract_model)
     D  plain + iterative repair: each node named in a load error is added to node_block_list and the model rebuilt
     E  as D, starting from B
     F  ONNX Runtime's own converter (onnxruntime.transformers.float16, keep_io_types=True)
     G  as F, LayerNorm kept in FP32
     H  graph upgraded to opset 20 (onnx.version_converter), then the plain conversion
3. Runtime version: A and B are also loaded in the newest onnxruntime from PyPI (separate process), to see whether the
   failure belongs to the converter or to the runtime version.
Output: JSON + CSV only; the FP16 files stay in /tmp.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.chdir("/tmp")
T0 = time.time()


def log(m):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {m}", flush=True)


def pip(args, target=None):
    t = time.time()
    cmd = [sys.executable, "-m", "pip", "install", "-q", *args] + (["--target", target] if target else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    log(f"pip {' '.join(args)}{' -> ' + target if target else ''}: exit {r.returncode} ({time.time() - t:.0f}s) {r.stderr[-300:]}")


pip(["onnxruntime==1.20.1", "onnx==1.18.0"])
pip(["--no-deps", "onnxconverter-common==1.16.0"])
pip(["onnxruntime"], target="/tmp/ort_new")
import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnx.utils  # noqa: E402
import onnxruntime as ort  # noqa: E402
import pandas as pd  # noqa: E402
from onnx import helper, version_converter  # noqa: E402
from onnxconverter_common import float16 as occ16  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

OUT = Path("/kaggle/working/output")
OUT.mkdir(parents=True, exist_ok=True)
WORK = Path("/tmp/k3")
WORK.mkdir(exist_ok=True)
EXPECTED = "4937ab381ee3e7c29d5ab18ea761a8c19ddffb6488210fb6030511a8041502eb"
LN_BLOCK = ["LayerNormalization", "ReduceMean", "Pow", "Sqrt"]
MAX_IR, N_IMG, N_TIME = 10, 32, 20
R = {"note": "K3, revision; served FP32 file, synthetic phantoms; no stroke data",
     "versions": {"onnxruntime": ort.__version__, "onnx": onnx.__version__}, "variants": {}}
ROWS = []


def save():
    (OUT / "fp16_rootcause.json").write_text(json.dumps(R, indent=2, default=str))
    pd.DataFrame(ROWS).to_csv(OUT / "fp16_rootcause.csv", index=False)


def first(*c):
    for p in c:
        if Path(p).exists():
            return Path(p)
    sys.exit(f"[ERROR] not found: {c}")


SRC = first(*(f"{r}{s}/neuroscan_r30a_s123.onnx" for r in ("/kaggle/input/notebooks/atik68/neuroscan-q-onnx-export-check",
                                                            "/kaggle/input/neuroscan-q-onnx-export-check")
              for s in ("/output", "")))
h = hashlib.sha256(SRC.read_bytes()).hexdigest()
assert h == EXPECTED, f"served model checksum mismatch: {h}"
FP32 = onnx.load(str(SRC))
R["source"] = {"path": str(SRC), "sha256": h, "opset": [(o.domain, o.version) for o in FP32.opset_import],
               "nodes": len(FP32.graph.node),
               "op_counts": pd.Series([n.op_type for n in FP32.graph.node]).value_counts().to_dict()}
log(f"served model {SRC} ok; {len(FP32.graph.node)} nodes; opset {R['source']['opset']}")


def phantom(seed):
    rng = np.random.default_rng(seed)
    img = Image.new("L", (512, 512), 0)
    d = ImageDraw.Draw(img)
    d.ellipse([60, 40, 452, 472], fill=int(rng.integers(150, 200)))
    d.ellipse([80, 60, 432, 452], fill=int(rng.integers(90, 130)))
    cx, cy, r = rng.integers(160, 352), rng.integers(140, 372), rng.integers(20, 60)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=int(rng.integers(40, 220)))
    a = np.asarray(img, dtype=np.float32) + rng.normal(0, 8, (512, 512))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))


MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
X = np.stack([((np.asarray(Image.merge("RGB", (g, g, g)), np.float32).transpose(2, 0, 1) / 255.0) - MEAN) / STD
              for g in (phantom(k).resize((224, 224)) for k in range(N_IMG))]).astype(np.float32)


def session(path):
    o = ort.SessionOptions()
    o.intra_op_num_threads = o.inter_op_num_threads = 1
    o.enable_cpu_mem_arena = False
    return ort.InferenceSession(str(path), sess_options=o, providers=["CPUExecutionProvider"])


def fix_ir(path):
    m = onnx.load(str(path))
    if m.ir_version > MAX_IR:
        m.ir_version = MAX_IR
        onnx.save(m, str(path))


ref = session(SRC)
OUTS = [o.name for o in ref.get_outputs()]
RL, RC = ref.run(OUTS, {"image": X})
RL_ONLY = None


def softmax(z):
    z = z.astype(np.float64)
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def norm(m):
    return m / (m.reshape(len(m), m.shape[1], -1).max(-1)[..., None, None] + 1e-8)


# ------------------------------------------------------------------ root-cause tracing in the FP32 graph
PRODUCER = {o: n for n in FP32.graph.node for o in n.output}
CONSUMERS = {}
for n in FP32.graph.node:
    for i in n.input:
        CONSUMERS.setdefault(i, []).append(n.name)
CAM_NODES = set()
ein = [n for n in FP32.graph.node if n.op_type == "Einsum"]
if ein:  # nodes that feed only the Grad-CAM output: ancestors of the Einsum's weight input not on the logits path
    def ancestors(name):
        seen, todo = set(), [name]
        while todo:
            t = todo.pop()
            p = PRODUCER.get(t)
            if p is not None and p.name not in seen:
                seen.add(p.name)
                todo += list(p.input)
        return seen
    logits_anc = ancestors("logits")
    CAM_NODES = (ancestors("gradcam") - logits_anc)
R["source"]["gradcam_only_nodes"] = len(CAM_NODES)


def describe(node_name, depth=3):
    """The node in the FP32 graph, its attributes, and up to `depth` levels of producers of each input."""
    n = next((x for x in FP32.graph.node if x.name == node_name), None)
    if n is None:
        return {"name": node_name, "found_in_fp32_graph": False}
    d = {"name": n.name, "op": n.op_type, "attrs": {a.name: str(helper.get_attribute_value(a))[:80] for a in n.attribute},
         "inputs": list(n.input), "outputs": list(n.output), "consumers": CONSUMERS.get(n.output[0], [])[:6],
         "part": "Grad-CAM only" if n.name in CAM_NODES else "logits path (shared with Grad-CAM)"}
    if depth:
        d["producers"] = [describe(PRODUCER[i].name, depth - 1) if i in PRODUCER else {"input": i}
                          for i in n.input]
    return d


def error_nodes(msg):
    return re.findall(r"node \(([^)]+)\)", msg)


# ------------------------------------------------------------------ variants
def build_occ(model, ln=False, nodes=()):
    ops = list(occ16.DEFAULT_OP_BLOCK_LIST) + (LN_BLOCK if ln else [])
    return occ16.convert_float_to_float16(model, keep_io_types=True, op_block_list=ops, node_block_list=list(nodes) or None)


def build_ortconv(model, ln=False):
    from onnxruntime.transformers.float16 import DEFAULT_OP_BLOCK_LIST, convert_float_to_float16
    ops = list(DEFAULT_OP_BLOCK_LIST) + (LN_BLOCK if ln else [])
    return convert_float_to_float16(model, keep_io_types=True, op_block_list=ops)


def evaluate(name, model, extra=None, logits_only=False):
    r = R["variants"][name] = dict(extra or {})
    p = WORK / f"{name}.onnx"
    onnx.save(model, str(p))
    fix_ir(p)
    r["mb"] = round(p.stat().st_size / 1e6, 1)
    r["fp16_initializers"] = int(sum(t.data_type == onnx.TensorProto.FLOAT16 for t in model.graph.initializer))
    try:
        s = session(p)
    except Exception as e:
        r["loads"], r["load_error"] = False, str(e)[:700]
        r["failing_nodes"] = [describe(n) for n in error_nodes(str(e))]
        log(f"{name}: does NOT load: {str(e)[:300]}")
        ROWS.append({"variant": name, "loads": False, "error": str(e)[:200]})
        save()
        return r, p
    outs = [o.name for o in s.get_outputs()]
    res = s.run(outs, {"image": X})
    lg = res[0]
    r["loads"] = True
    r["finite"] = bool(all(np.isfinite(a).all() for a in res))
    base = RL
    r["logits_maxabs_vs_fp32"] = float(np.abs(lg.astype(np.float64) - base).max())
    r["prob_maxabs_vs_fp32"] = float(np.abs(softmax(lg) - softmax(base)).max())
    r["top1_changes_vs_fp32"] = int((lg.argmax(1) != base.argmax(1)).sum())
    if not logits_only and len(res) > 1:
        r["gradcam_normalised_maxabs_vs_fp32"] = float(np.abs(norm(res[1].astype(np.float64)) - norm(RC)).max())
    xs = X[:1]
    for _ in range(3):
        s.run(outs, {"image": xs})
    ts = []
    for _ in range(N_TIME):
        t = time.perf_counter()
        s.run(outs, {"image": xs})
        ts.append(time.perf_counter() - t)
    r["median_s_1thread"] = float(np.median(ts))
    log(f"{name}: loads; finite {r['finite']}; max|dp| {r['prob_maxabs_vs_fp32']:.2e}; top-1 changes "
        f"{r['top1_changes_vs_fp32']}; {r['mb']} MB; {r['median_s_1thread']:.3f} s")
    ROWS.append({"variant": name, "loads": True, **{k: v for k, v in r.items() if not isinstance(v, (dict, list))}})
    save()
    return r, p


def repair(name, ln):
    """Add every node named in a load error to node_block_list and rebuild, up to 15 rounds."""
    blocked, hist = [], []
    for _ in range(15):
        m = build_occ(onnx.load(str(SRC)), ln=ln, nodes=blocked)
        p = WORK / f"{name}_try.onnx"
        onnx.save(m, str(p))
        fix_ir(p)
        try:
            session(p)
            break
        except Exception as e:
            nn = [x for x in error_nodes(str(e)) if x not in blocked]
            hist.append({"error": str(e)[:300], "new_nodes": nn})
            if not nn:
                break
            blocked += nn
    r, _ = evaluate(name, m, {"blocked_nodes": blocked, "rounds": hist,
                              "blocked_described": [describe(b, 1) for b in blocked]})
    return r


TIMES = {}
fp32_t = []
for _ in range(3):
    ref.run(OUTS, {"image": X[:1]})
for _ in range(N_TIME):
    t = time.perf_counter()
    ref.run(OUTS, {"image": X[:1]})
    fp32_t.append(time.perf_counter() - t)
R["fp32_median_s_1thread"] = float(np.median(fp32_t))
log(f"FP32 reference: {R['fp32_median_s_1thread']:.3f} s per image, one thread")

STEPS = [
    ("A_plain", lambda: evaluate("A_plain", build_occ(onnx.load(str(SRC))))),
    ("B_layernorm_fp32", lambda: evaluate("B_layernorm_fp32", build_occ(onnx.load(str(SRC)), ln=True))),
    ("C_plain_logits_only", lambda: evaluate(
        "C_plain_logits_only",
        build_occ(onnx.utils.Extractor(onnx.load(str(SRC))).extract_model(["image"], ["logits"])),
        logits_only=True)),
    ("D_plain_repaired", lambda: repair("D_plain_repaired", False)),
    ("E_layernorm_fp32_repaired", lambda: repair("E_layernorm_fp32_repaired", True)),
    ("F_ort_converter", lambda: evaluate("F_ort_converter", build_ortconv(onnx.load(str(SRC))))),
    ("G_ort_converter_layernorm_fp32", lambda: evaluate("G_ort_converter_layernorm_fp32",
                                                        build_ortconv(onnx.load(str(SRC)), ln=True))),
    ("H_opset20_plain", lambda: evaluate("H_opset20_plain",
                                         build_occ(version_converter.convert_version(onnx.load(str(SRC)), 20)))),
]
for name, fn in STEPS:
    t = time.time()
    try:
        fn()
    except Exception as e:
        R["variants"].setdefault(name, {})["build_error"] = f"{type(e).__name__}: {str(e)[:500]}"
        R["variants"][name]["traceback"] = traceback.format_exc()[-1200:]
        log(f"{name}: build FAILED {type(e).__name__}: {str(e)[:300]}")
        ROWS.append({"variant": name, "loads": None, "error": f"build: {str(e)[:200]}"})
    R["variants"].setdefault(name, {})["seconds"] = round(time.time() - t, 1)
    save()

# ------------------------------------------------------------------ newest onnxruntime (separate process)
probe = WORK / "probe.py"
probe.write_text(
    "import json, sys\nsys.path.insert(0, '/tmp/ort_new')\nimport onnxruntime as o\nout = {'version': o.__version__}\n"
    "for p in sys.argv[1:]:\n"
    "    try:\n        o.InferenceSession(p, providers=['CPUExecutionProvider']); out[p] = 'loads'\n"
    "    except Exception as e:\n        out[p] = str(e)[:400]\nprint(json.dumps(out))\n")
files = [str(WORK / f"{v}.onnx") for v in ("A_plain", "B_layernorm_fp32") if (WORK / f"{v}.onnx").exists()]
r = subprocess.run([sys.executable, "-I", str(probe), *files], capture_output=True, text=True, cwd="/tmp")
try:
    R["newest_onnxruntime"] = json.loads(r.stdout.strip().splitlines()[-1])
except Exception:
    R["newest_onnxruntime"] = {"probe_failed": r.stderr[-800:]}
log(f"newest onnxruntime: {json.dumps(R['newest_onnxruntime'])[:600]}")
save()
log("Done.")
