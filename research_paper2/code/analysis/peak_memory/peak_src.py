"""
NeuroScan paper 2 -- E2 follow-up: peak memory of the served model and its INT8 copies, measured correctly (2026-10-24).
CPU only; no training; no held-out evaluation data (the 28-case web-app test pack only).

Why: kernel neuroscan-p2-compression-gradcam read peak memory as ru_maxrss of a child process. On Linux ru_maxrss
survives execve, so every child reported its large parent's peak (4,398 MB for every variant). Here the peak is the
child's own VmHWM from /proc (a new address space starts at exec), the parent stays small (standard library only), and
a control run shows the artefact: one probe is started from a parent holding ~2 GB, read both ways.

Measurements, each in a fresh process with the server's pinned packages (requirements.txt of render_inference_server):
  A. pipeline: the server's inference.py verbatim (1 thread, arena off) -- VmHWM/VmRSS after imports, after model
     load, after the first request and after 80 requests (the 28 cases on their evaluated pathway + 12 single CT
     slices, cycled); 3 repeats per variant.
  B. full service: the server's app.py under uvicorn on localhost (throw-away local key), 120 POST /api/predict
     requests sent from this parent; the server's VmHWM/VmRSS read from /proc/<pid>/status; 1 run per variant.
Variants: FP32 (served, SHA-256 4937ab38...), INT8-dynamic, INT8-static (files from neuroscan-p2-compression-gradcam,
SHA-256 checked). FP16 is not measured: ONNX Runtime 1.20.1 refuses both FP16 files.
"""
import csv
import hashlib
import io
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
import zipfile
from pathlib import Path

T0 = time.time()
WORK = Path("/tmp/pm")
OUT = Path("/kaggle/working/output")
for d in (WORK, OUT):
    d.mkdir(parents=True, exist_ok=True)
REPEATS, N_PIPE, N_APP = 3, 80, 120
LOCAL_KEY = "peak-memory-local-test"   # throw-away key for the local uvicorn process only
SHA = {"FP32": "4937ab381ee3e7c29d5ab18ea761a8c19ddffb6488210fb6030511a8041502eb",
       "INT8-dyn": "ad2f6e0ee22b28ec98295b742075a2168e480801073a8df3f1222df37c74d99f",
       "INT8-static": "35d8a4007b813f8fb6eca6c10078823fc5b69a654314c36d76c3412a38b02a46"}
R = {"note": "peak memory re-measured with VmHWM; see docstring", "python": sys.version.split()[0]}


def log(m):
    print(f"[{(time.time() - T0) / 60:5.1f} min | parent {status_mb('VmRSS')} MB] {m}", flush=True)


def status_mb(field, pid="self"):
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith(field):
                return int(line.split()[1]) // 1024
    except OSError:
        return None


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def first(*cands, what=""):
    for c in cands:
        if Path(c).exists():
            return Path(c)
    print(f"[ERROR] {what} not found; tried {[str(c) for c in cands]}")
    sys.exit(1)


def nb_file(slug, rel):
    roots = (f"/kaggle/input/notebooks/atik68/{slug}", f"/kaggle/input/{slug}")
    return first(*(f"{r}{sub}/{rel}" for r in roots for sub in ("/output", "")), what=f"{slug}/{rel}")


# ------------------------------------------------------------------------------------------------- packages
REQS = "__REQUIREMENTS_TXT__"
t = time.time()
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--target", "/tmp/srvpkgs",
                    *[l for l in REQS.splitlines() if l.strip()]], capture_output=True, text=True)
log(f"pip server requirements -> /tmp/srvpkgs: exit {r.returncode} ({time.time() - t:.0f}s) {r.stderr[-300:]}")
if r.returncode:
    sys.exit(1)
R["server_requirements"] = REQS.splitlines()

# ------------------------------------------------------------------------------------------------- models
MODELS = {"FP32": nb_file("neuroscan-q-onnx-export-check", "neuroscan_r30a_s123.onnx"),
          "INT8-dyn": nb_file("neuroscan-p2-compression-gradcam", "models/neuroscan_r30a_s123_int8_dynamic.onnx"),
          "INT8-static": nb_file("neuroscan-p2-compression-gradcam", "models/neuroscan_r30a_s123_int8_static.onnx")}
R["models"] = {}
for k, p in MODELS.items():
    d = sha256(p)
    assert d == SHA[k], f"{k}: checksum {d}"
    R["models"][k] = {"file": p.name, "mb": round(p.stat().st_size / 1e6, 2), "sha256": d}
log(f"models verified: { {k: v['mb'] for k, v in R['models'].items()} }")

# The server's committed inference.py and app.py, verbatim (inserted by build_kernel.py; SHA-256 checked).
INFERENCE_PY, INFERENCE_SHA = "__INFERENCE_PY__", "__INFERENCE_PY_SHA__"
APP_PY, APP_SHA = "__APP_PY__", "__APP_PY_SHA__"
assert hashlib.sha256(INFERENCE_PY.encode()).hexdigest() == INFERENCE_SHA
assert hashlib.sha256(APP_PY.encode()).hexdigest() == APP_SHA
R["inference_py_sha256"], R["app_py_sha256"] = INFERENCE_SHA, APP_SHA
SHA_LINE = f'EXPECTED_SHA256 = "{SHA["FP32"]}"'
assert INFERENCE_PY.count(SHA_LINE) == 1


def server_dir(name):
    d = WORK / f"srv_{name}"
    (d / "models").mkdir(parents=True, exist_ok=True)
    (d / "inference.py").write_text(INFERENCE_PY.replace(SHA_LINE, f'EXPECTED_SHA256 = "{SHA[name]}"'))
    (d / "app.py").write_text(APP_PY)
    link = d / "models" / "neuroscan_r30a_s123.onnx"
    if not link.exists():
        os.symlink(MODELS[name], link)
    return d


# ------------------------------------------------------------------------------------------------- requests
PACK = nb_file("neuroscan-q-test-pack", "neuroscan_test_pack.zip")
REQ = WORK / "req"
REQ.mkdir(exist_ok=True)
with zipfile.ZipFile(PACK) as zf:
    cases = list(csv.DictReader(io.StringIO(zf.read("answer_sheet.csv").decode("utf-8"))))
    requests_ = []
    for c in cases:
        name = c["case"]
        if c["modality"] == "MRI":
            roles = {"file": f"{name}/1_DWI_main_image.png", "adc_file": f"{name}/2_ADC.png"}
        else:
            roles = {"file": f"{name}/2_main_slice.png", "prev_file": f"{name}/1_previous_slice.png",
                     "next_file": f"{name}/3_next_slice.png"}
        files = {}
        for role, member in roles.items():
            f = REQ / member.replace("/", "__")
            f.write_bytes(zf.read(member))
            files[role] = str(f)
        requests_.append({"modality": c["modality"], "files": files})
        if c["modality"] == "CT":
            requests_.append({"modality": "CT", "files": {"file": files["file"]}})   # single-slice pathway
json.dump(requests_, open(WORK / "requests.json", "w"))
R["requests"] = {"cases": len(cases), "request_kinds": len(requests_), "pack_sha256": sha256(PACK)}
log(f"test pack: {len(cases)} cases -> {len(requests_)} request kinds")

ENV = dict(os.environ, PYTHONPATH="/tmp/srvpkgs", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

PROBE = r'''
import json, resource, sys, time
def st(field):
    for l in open("/proc/self/status"):
        if l.startswith(field):
            return int(l.split()[1]) // 1024
out = {"start": {"hwm": st("VmHWM"), "rss": st("VmRSS")}}
sys.path.insert(0, sys.argv[1])
import numpy as np, onnxruntime
from PIL import Image
import inference as inf
out["after_imports"] = {"hwm": st("VmHWM"), "rss": st("VmRSS")}
p = inf.get_pipeline()
out["after_load"] = {"hwm": st("VmHWM"), "rss": st("VmRSS")}
reqs = json.load(open(sys.argv[2]))
n = int(sys.argv[3])
ts = []
for i in range(n):
    q = reqs[i % len(reqs)]
    ims = {k: Image.open(v) for k, v in q["files"].items()}
    for im in ims.values():
        im.load()
    t = time.time()
    p.predict(q["modality"], ims["file"], ims.get("prev_file"), ims.get("next_file"), ims.get("adc_file"))
    ts.append(time.time() - t)
    if i == 0:
        out["after_first_request"] = {"hwm": st("VmHWM"), "rss": st("VmRSS")}
out["after_requests"] = {"hwm": st("VmHWM"), "rss": st("VmRSS")}
out["ru_maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
out["seconds_median"] = sorted(ts)[len(ts) // 2]
print("RESULT " + json.dumps(out), flush=True)
'''
(WORK / "probe.py").write_text(PROBE)


def probe(name):
    p = subprocess.run([sys.executable, str(WORK / "probe.py"), str(server_dir(name)), str(WORK / "requests.json"),
                        str(N_PIPE)], env=ENV, capture_output=True, text=True)
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
    return json.loads(line[7:]) if line else {"error": p.stderr[-1500:]}


# ------------------------------------------------------------------------------------- A. pipeline probes
R["A_pipeline"] = {}
for name in MODELS:
    R["A_pipeline"][name] = []
    for rep in range(REPEATS):
        t = time.time()
        res = probe(name)
        R["A_pipeline"][name].append(res)
        log(f"A {name} #{rep + 1} ({time.time() - t:.0f}s): " + (res.get("error", "")[-300:] or
            f"peak (VmHWM) {res['after_requests']['hwm']} MB; RSS after load {res['after_load']['rss']} MB, after "
            f"requests {res['after_requests']['rss']} MB; ru_maxrss {res['ru_maxrss_mb']} MB; median {res['seconds_median']:.3f}s"))
    json.dump(R, open(OUT / "peak_memory.json", "w"), indent=2)

# control: the same FP32 probe started from a parent holding ~2 GB -- ru_maxrss inherits it, VmHWM does not
ballast = bytearray(b"\x01") * (2 * 1024 ** 3)
log(f"control: parent now holds {len(ballast) / 1e9:.1f} GB")
ctrl = probe("FP32")
R["control_parent_2GB"] = {"parent_vmrss_mb": status_mb("VmRSS"), "child": ctrl}
log(f"control child: VmHWM {ctrl.get('after_requests', {}).get('hwm')} MB vs ru_maxrss {ctrl.get('ru_maxrss_mb')} MB")
del ballast
json.dump(R, open(OUT / "peak_memory.json", "w"), indent=2)


# -------------------------------------------------------------------------------------- B. full service
def multipart(fields, files):
    b = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    for k, path in files.items():
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"; filename="{k}.png"\r\n'
                     f'Content-Type: image/png\r\n\r\n'.encode() + Path(path).read_bytes() + b"\r\n")
    parts.append(f"--{b}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={b}"


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


R["B_service"] = {}
for name in MODELS:
    port, t = free_port(), time.time()
    env = dict(ENV, NEUROSCAN_API_KEY=LOCAL_KEY, PYTHONPATH=f"{server_dir(name)}:/tmp/srvpkgs")
    srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(port)],
                           cwd=server_dir(name), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    res, ok, lat = {"pid_start_hwm_mb": status_mb("VmHWM", srv.pid)}, 0, []
    try:
        url = f"http://127.0.0.1:{port}"
        for _ in range(120):
            try:
                urllib.request.urlopen(urllib.request.Request(f"{url}/api/health", headers={"X-API-Key": LOCAL_KEY}), timeout=2)
                break
            except Exception:
                time.sleep(0.5)
        res["ready_s"] = round(time.time() - t, 2)
        res["after_startup"] = {"hwm": status_mb("VmHWM", srv.pid), "rss": status_mb("VmRSS", srv.pid)}
        for i in range(N_APP):
            q = requests_[i % len(requests_)]
            body, ctype = multipart({"modality": q["modality"]}, q["files"])
            t1 = time.time()
            rq = urllib.request.Request(f"{url}/api/predict", data=body, headers={"X-API-Key": LOCAL_KEY, "Content-Type": ctype})
            with urllib.request.urlopen(rq, timeout=120) as resp:
                ok += int(resp.status == 200 and "prediction" in json.loads(resp.read()))
            lat.append(time.time() - t1)
        res["after_requests"] = {"hwm": status_mb("VmHWM", srv.pid), "rss": status_mb("VmRSS", srv.pid)}
        res["requests_ok"], res["requests"] = ok, N_APP
        res["latency_median_s"] = sorted(lat)[len(lat) // 2]
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
    finally:
        srv.terminate()
        try:
            res["server_log_tail"] = srv.communicate(timeout=20)[0][-800:]
        except subprocess.TimeoutExpired:
            srv.kill()
    R["B_service"][name] = res
    log(f"B {name}: {json.dumps({k: res.get(k) for k in ('ready_s', 'after_startup', 'after_requests', 'requests_ok', 'latency_median_s', 'error')})}")
    json.dump(R, open(OUT / "peak_memory.json", "w"), indent=2)
log("Done.")
