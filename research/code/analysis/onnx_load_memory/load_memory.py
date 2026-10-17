"""
NeuroScan -- ONNX Runtime load-memory check (2026-10-07). CPU, no data. The export check showed 205-231 MB steady
memory but a 1,654 MB peak while the session was created. Each configuration below runs in a fresh process with the
server's pinned packages (onnxruntime 1.20.1, numpy 2.1.3, pillow 11.0.0): peak RSS during load, RSS after 3
four-view requests, seconds per request, and logits compared with the default configuration.
Configurations: default (all graph optimizations); optimizations disabled; basic only; all + prepacking disabled;
an offline-optimized model (saved once in a separate process) loaded with optimizations disabled.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

SRC = next(p for p in (Path("/kaggle/input/notebooks/atik68/neuroscan-q-onnx-export-check/output/neuroscan_r30a_s123.onnx"),
                       Path("/kaggle/input/neuroscan-q-onnx-export-check/output/neuroscan_r30a_s123.onnx"))
           if p.exists()) if any(Path(p).exists() for p in (
    "/kaggle/input/notebooks/atik68/neuroscan-q-onnx-export-check/output/neuroscan_r30a_s123.onnx",
    "/kaggle/input/neuroscan-q-onnx-export-check/output/neuroscan_r30a_s123.onnx")) else next(Path("/kaggle/input").rglob("neuroscan_r30a_s123.onnx"))
print("model:", SRC, flush=True)
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--target", "/tmp/srvpkgs", "onnxruntime==1.20.1",
                    "numpy==2.1.3", "pillow==11.0.0"], capture_output=True, text=True)
print("pip exit", r.returncode, flush=True)
OPT = "/tmp/opt_model.onnx"
PROBE = r'''
import json, resource, sys, time
import numpy as np, onnxruntime as ort
cfg, src, opt = sys.argv[1], sys.argv[2], sys.argv[3]
def rss():
    for l in open("/proc/self/status"):
        if l.startswith("VmRSS"):
            return int(l.split()[1]) // 1024
o = ort.SessionOptions(); o.intra_op_num_threads = 1; o.inter_op_num_threads = 1; o.enable_cpu_mem_arena = False
path = src
if cfg == "make_offline":
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    o.optimized_model_filepath = opt
    ort.InferenceSession(src, sess_options=o, providers=["CPUExecutionProvider"]); print("RESULT {}"); sys.exit()
if cfg == "disabled":
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
elif cfg == "basic":
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
elif cfg == "no_prepack":
    o.add_session_config_entry("session.disable_prepacking", "1")
elif cfg == "offline_optimized":
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    path = opt
t = time.time()
s = ort.InferenceSession(path, sess_options=o, providers=["CPUExecutionProvider"])
out = {"load_s": round(time.time() - t, 2), "rss_after_load_mb": rss(), "peak_after_load_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024}
x = np.random.default_rng(0).normal(size=(4, 3, 224, 224)).astype(np.float32)
ts = []
for _ in range(3):
    t = time.time(); lg, cm = s.run(["logits", "gradcam"], {"image": x}); ts.append(time.time() - t)
out.update({"rss_after_requests_mb": rss(), "peak_total_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
            "seconds_per_request": round(float(np.median(ts)), 3), "logits": lg.tolist(), "cam_sum": float(cm.sum())})
print("RESULT " + json.dumps(out), flush=True)
'''
Path("/tmp/probe.py").write_text(PROBE)
env = dict(os.environ, PYTHONPATH="/tmp/srvpkgs", OMP_NUM_THREADS="1")
res = {}
for cfg in ("default", "disabled", "basic", "no_prepack", "make_offline", "offline_optimized"):
    p = subprocess.run([sys.executable, "/tmp/probe.py", cfg, str(SRC), OPT], env=env, capture_output=True, text=True)
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
    if line is None:
        print(cfg, "FAILED", p.stderr[-800:], flush=True)
        continue
    res[cfg] = json.loads(line[7:])
import numpy as np
ref = np.array(res["default"]["logits"])
for cfg, v in res.items():
    if "logits" in v:
        v["logits_maxabs_vs_default"] = float(np.abs(np.array(v.pop("logits")) - ref).max())
    print(cfg, json.dumps(v), flush=True)
if Path(OPT).exists():
    print("offline-optimized model MB", round(Path(OPT).stat().st_size / 1e6, 1))
Path("/kaggle/working/output").mkdir(parents=True, exist_ok=True)
json.dump(res, open("/kaggle/working/output/onnx_load_memory.json", "w"), indent=2)
print("Done.")
