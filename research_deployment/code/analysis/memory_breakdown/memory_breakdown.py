"""
NeuroScan paper 2 -- K5, JTCSST revision (R2-12): where does the service's memory go?  CPU only; no stroke data.

The earlier kernel neuroscan-p2-peak-memory measured the whole pipeline (FP32 peak about 298 MB, INT8-dynamic about
195 MB; full service 328 and 228 MB). This run splits that into parts, each read as VmRSS / VmHWM from /proc in a fresh
process with the server's pinned packages (requirements.txt of render_inference_server), in the order the service
builds up:
  start (bare Python) -> numpy + Pillow -> onnxruntime -> server framework (fastapi, uvicorn, python-multipart)
  -> model session (server options: 1 thread, memory arena off) -> first request -> 20 more requests.
Configurations (2 repeats each): model FP32 / INT8-dynamic x request of 4 views (as served) / 1 view x arena off (as
served) / on; plus FP32, 4 views, arena off with graph optimisation disabled (does load-time optimisation cause the
load peak?). Requests are deterministic random tensors of the served input shape; content does not affect memory.
Also from the ONNX files: initializer bytes by data type, and every intermediate tensor's size for one view (ONNX shape
inference), so the measured request memory can be compared with the tensors the graph must hold.
Derived parts per configuration: interpreter, numeric libraries, ONNX Runtime, server framework, model session (RSS
after load minus before), load transient (HWM minus RSS after load), request working memory (HWM after requests
minus RSS after load) and retained growth (RSS after requests minus after load).
Output: JSON + CSV only.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.chdir("/tmp")
T0 = time.time()
WORK = Path("/tmp/k5")
OUT = Path("/kaggle/working/output")
for d in (WORK, OUT):
    d.mkdir(parents=True, exist_ok=True)
REPEATS, N_REQ = 2, 20
REQS = ["fastapi==0.115.0", "uvicorn[standard]==0.30.6", "python-multipart==0.0.9", "pillow==11.0.0", "numpy==2.1.3",
        "onnxruntime==1.20.1"]
SHA = {"FP32": "4937ab381ee3e7c29d5ab18ea761a8c19ddffb6488210fb6030511a8041502eb",
       "INT8-dyn": "ad2f6e0ee22b28ec98295b742075a2168e480801073a8df3f1222df37c74d99f"}
R = {"note": "K5, revision; memory breakdown, VmRSS/VmHWM per stage in fresh processes", "python": sys.version.split()[0],
     "server_requirements": REQS}


def log(m):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {m}", flush=True)


def save():
    (OUT / "memory_breakdown.json").write_text(json.dumps(R, indent=2, default=str))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def nb_file(slug, rel):
    for r in (f"/kaggle/input/notebooks/atik68/{slug}", f"/kaggle/input/{slug}"):
        for sub in ("/output", ""):
            p = Path(f"{r}{sub}/{rel}")
            if p.exists():
                return p
    sys.exit(f"[ERROR] {slug}/{rel} not found")


t = time.time()
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--target", "/tmp/srvpkgs", *REQS],
                   capture_output=True, text=True)
log(f"pip server requirements -> /tmp/srvpkgs: exit {r.returncode} ({time.time() - t:.0f}s) {r.stderr[-300:]}")
if r.returncode:
    sys.exit(1)
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx==1.18.0"], capture_output=True, text=True)
log(f"pip onnx (parent, for shape inference): exit {r.returncode}")

MODELS = {"FP32": nb_file("neuroscan-q-onnx-export-check", "neuroscan_r30a_s123.onnx"),
          "INT8-dyn": nb_file("neuroscan-p2-compression-gradcam", "models/neuroscan_r30a_s123_int8_dynamic.onnx")}
R["models"] = {}
for k, p in MODELS.items():
    d = sha256(p)
    assert d == SHA[k], f"{k}: checksum {d}"
    R["models"][k] = {"file": p.name, "mb": round(p.stat().st_size / 1e6, 2), "sha256": d}
log(f"models verified: { {k: v['mb'] for k, v in R['models'].items()} }")

# ------------------------------------------------------------------ static sizes from the ONNX files
import numpy as np  # noqa: E402
import onnx  # noqa: E402
from onnx import shape_inference  # noqa: E402

ELEM = {1: 4, 10: 2, 11: 8, 6: 4, 7: 8, 2: 1, 3: 1, 9: 1, 12: 4, 13: 8}
for k, p in MODELS.items():
    m = onnx.load(str(p))
    by_type = {}
    for t_ in m.graph.initializer:
        n = int(np.prod(t_.dims)) * ELEM.get(t_.data_type, 4)
        name = onnx.TensorProto.DataType.Name(t_.data_type)
        by_type[name] = by_type.get(name, 0) + n
    inp = m.graph.input[0].type.tensor_type.shape.dim
    inp[0].ClearField("dim_param")
    inp[0].dim_value = 1
    try:
        inf = shape_inference.infer_shapes(m, data_prop=True)
        sizes = []
        for vi in inf.graph.value_info:
            tt = vi.type.tensor_type
            dims = [d_.dim_value for d_ in tt.shape.dim]
            if dims and all(d_ > 0 for d_ in dims):
                sizes.append(int(np.prod(dims)) * ELEM.get(tt.elem_type, 4))
        sizes.sort(reverse=True)
        acts = {"tensors_with_static_shape": len(sizes), "largest_mb": round(sizes[0] / 1e6, 2),
                "top10_mb": [round(s / 1e6, 2) for s in sizes[:10]], "sum_all_mb": round(sum(sizes) / 1e6, 1),
                "input_one_view_mb": round(3 * 224 * 224 * 4 / 1e6, 2), "input_four_views_mb": round(4 * 3 * 224 * 224 * 4 / 1e6, 2)}
    except Exception as e:
        acts = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    R["models"][k].update({"initializer_mb_by_type": {a: round(b / 1e6, 1) for a, b in by_type.items()},
                           "initializers": len(m.graph.initializer), "nodes": len(m.graph.node),
                           "intermediate_tensors_one_view": acts})
    log(f"{k}: initializers {R['models'][k]['initializer_mb_by_type']}; tensors {acts}")
    del m
save()

# ------------------------------------------------------------------ staged probe
PROBE = r'''
import json, sys, time
def st():
    d = {}
    for l in open("/proc/self/status"):
        if l.startswith(("VmRSS", "VmHWM")):
            d[l.split(":")[0]] = int(l.split()[1]) // 1024
    return {"rss": d["VmRSS"], "hwm": d["VmHWM"]}
model, views, arena, opt = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "on", sys.argv[4]
out = {"start": st()}
import numpy as np
from PIL import Image
out["numpy_pillow"] = st()
import onnxruntime as ort
out["onnxruntime"] = st()
import fastapi, uvicorn, multipart  # the server framework, as app.py imports it
out["server_framework"] = st()
o = ort.SessionOptions()
o.intra_op_num_threads = 1
o.inter_op_num_threads = 1
o.enable_cpu_mem_arena = arena
if opt == "disabled":
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
s = ort.InferenceSession(model, sess_options=o, providers=["CPUExecutionProvider"])
out["model_session"] = st()
x = np.random.default_rng(0).standard_normal((views, 3, 224, 224)).astype(np.float32)
ts = []
for i in range(1 + N_REQ):
    t = time.perf_counter()
    s.run(["logits", "gradcam"], {"image": x})
    ts.append(time.perf_counter() - t)
    if i == 0:
        out["first_request"] = st()
out["after_requests"] = st()
out["seconds_median"] = sorted(ts)[len(ts) // 2]
print("RESULT " + json.dumps(out), flush=True)
'''.replace("N_REQ", str(N_REQ))
(WORK / "probe.py").write_text(PROBE)
ENV = dict(os.environ, PYTHONPATH="/tmp/srvpkgs", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")


def probe(model, views, arena, opt):
    p = subprocess.run([sys.executable, str(WORK / "probe.py"), str(MODELS[model]), str(views), arena, opt],
                       env=ENV, capture_output=True, text=True, cwd="/tmp")
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
    return json.loads(line[7:]) if line else {"error": p.stderr[-1500:]}


def parts(r):
    s = r
    return {"interpreter_rss": s["start"]["rss"],
            "numpy_pillow": s["numpy_pillow"]["rss"] - s["start"]["rss"],
            "onnxruntime_import": s["onnxruntime"]["rss"] - s["numpy_pillow"]["rss"],
            "server_framework_import": s["server_framework"]["rss"] - s["onnxruntime"]["rss"],
            "model_session_retained": s["model_session"]["rss"] - s["server_framework"]["rss"],
            "load_transient": s["model_session"]["hwm"] - s["model_session"]["rss"],
            "request_working": s["after_requests"]["hwm"] - s["model_session"]["rss"],
            "retained_growth_after_requests": s["after_requests"]["rss"] - s["model_session"]["rss"],
            "peak_hwm": s["after_requests"]["hwm"], "rss_after_requests": s["after_requests"]["rss"],
            "seconds_median": round(s["seconds_median"], 3)}


CONFIGS = [(m, v, a, "default") for m in MODELS for v in (4, 1) for a in ("off", "on")]
CONFIGS.append(("FP32", 4, "off", "disabled"))
R["configs"], rows = {}, []
for m, v, a, opt in CONFIGS:
    key = f"{m}|views={v}|arena={a}|opt={opt}"
    R["configs"][key] = []
    for rep in range(REPEATS):
        t = time.time()
        res = probe(m, v, a, opt)
        R["configs"][key].append(res)
        if "error" in res:
            log(f"{key} #{rep + 1}: ERROR {res['error'][-400:]}")
            continue
        pt = parts(res)
        rows.append({"model": m, "views": v, "arena": a, "graph_opt": opt, "repeat": rep + 1, **pt})
        log(f"{key} #{rep + 1} ({time.time() - t:.0f}s): {json.dumps(pt)}")
    save()

import pandas as pd  # noqa: E402

df = pd.DataFrame(rows)
df.to_csv(OUT / "memory_breakdown.csv", index=False)
if len(df):
    R["summary_mean_mb"] = (df.drop(columns="repeat").groupby(["model", "views", "arena", "graph_opt"]).mean()
                            .round(1).reset_index().to_dict("records"))
save()
log("Done.")
