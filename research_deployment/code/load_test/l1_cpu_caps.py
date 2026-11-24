"""
NeuroScan paper 2 -- L1 (JTCSST second revision, comment 5): the inference service under other compute
configurations, emulated on the authors' PC.

Why: the live services run on one configuration (Render free tier: 0.1 CPU, 512 MB). The reviewer asks how the
deployment behaves on other compute budgets.

What: the committed server (app.py + inference.py, from the hash-checked requirements.lock environment of the rebuild
check) runs on 127.0.0.1 inside a Windows job object that hard-caps its CPU (0.1, 0.25, 0.5, 1 and 2 cores of this
machine) and its memory (512 MB, as on the free tier; the peak the job used is read from the job object). For each
model variant (fp32, int8_dynamic) and cap:
  1. start-up: time until /api/health answers (model load under the cap); the variant and weights SHA-256 it reports;
  2. sequential: the 28 test-pack cases, 3 rounds, one request at a time (after 2 warm-up requests): client latency,
     server model time, prediction and probabilities against the answer sheet of saved offline predictions;
  3. concurrent: the 28 cases sent by 4 clients at once (7 each): latency under queueing, and whether every
     concurrent answer equals the same case's sequential answer (fidelity under concurrent requests);
  4. the job's peak memory, and whether the memory limit was ever hit (a failed request or a dead server).
The API key is a random value made here, passed only in the server's environment, never printed or saved.
This emulates CPU share and memory on one PC (Intel i3-8130U, 2 cores / 4 threads); it is not Render's hardware,
CPU model or network. The cap is applied to the server process right after it starts (before the model loads).

Usage (Windows only):
  python -I l1_cpu_caps.py --repo <render_inference_server> --python <venv python.exe> --pack <test pack zip>
         --models <folder with both .onnx files> --out <folder> [--caps 0.1 0.25 0.5 1 2] [--rounds 3]
  python -I l1_cpu_caps.py --selftest      (checks the CPU cap on a busy loop; no server, no model)
"""
import argparse
import csv
import ctypes
import hashlib
import io
import json
import os
import platform
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path

LABELS = ("normal", "ischemic", "hemorrhagic")
PORT = 8766
NCPU = os.cpu_count()
MODEL_FILES = {"fp32": "neuroscan_r30a_s123.onnx", "int8_dynamic": "neuroscan_r30a_s123_int8_dynamic.onnx"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------ Windows job object
k32 = ctypes.WinDLL("kernel32", use_last_error=True)
JobObjectExtendedLimitInformation, JobObjectCpuRateControlInformation = 9, 15
JOB_OBJECT_LIMIT_JOB_MEMORY, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x200, 0x2000
CPU_RATE_ENABLE, CPU_RATE_HARD_CAP = 0x1, 0x4


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                                  "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class BASIC_LIMIT(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]


class EXTENDED_LIMIT(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", BASIC_LIMIT), ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


class CPU_RATE(ctypes.Structure):
    _fields_ = [("ControlFlags", wintypes.DWORD), ("CpuRate", wintypes.DWORD)]


k32.CreateJobObjectW.restype = wintypes.HANDLE
k32.OpenProcess.restype = wintypes.HANDLE


def _ok(r, what):
    if not r:
        raise ctypes.WinError(ctypes.get_last_error(), what)


def make_job(cores, mem_mb):
    """A job object with a hard CPU cap of `cores` of this machine and a job memory limit (None = no limit)."""
    job = k32.CreateJobObjectW(None, None)
    _ok(job, "CreateJobObject")
    ext = EXTENDED_LIMIT()
    ext.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if mem_mb:
        ext.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_JOB_MEMORY
        ext.JobMemoryLimit = int(mem_mb * 1024 * 1024)
    _ok(k32.SetInformationJobObject(job, JobObjectExtendedLimitInformation, ctypes.byref(ext), ctypes.sizeof(ext)),
        "SetInformationJobObject(memory)")
    if cores is not None:
        rate = max(1, min(10000, round(cores / NCPU * 10000)))   # units of 1/100 % of all logical CPUs
        cr = CPU_RATE(CPU_RATE_ENABLE | CPU_RATE_HARD_CAP, rate)
        _ok(k32.SetInformationJobObject(job, JobObjectCpuRateControlInformation, ctypes.byref(cr), ctypes.sizeof(cr)),
            "SetInformationJobObject(cpu rate)")
    return job


def assign(job, pid):
    h = k32.OpenProcess(0x1F0FFF, False, pid)
    _ok(h, "OpenProcess")
    try:
        _ok(k32.AssignProcessToJobObject(job, h), "AssignProcessToJobObject")
    finally:
        k32.CloseHandle(h)


def peak_job_mb(job):
    ext = EXTENDED_LIMIT()
    _ok(k32.QueryInformationJobObject(job, JobObjectExtendedLimitInformation, ctypes.byref(ext), ctypes.sizeof(ext),
                                      None), "QueryInformationJobObject")
    return round(ext.PeakJobMemoryUsed / 2 ** 20, 1)


def selftest():
    """Busy loop for 6 s of wall time under each cap; CPU seconds used / wall seconds should be close to the cap."""
    code = "import time\nt=time.time()\nwhile time.time()-t<6: pass\nimport os;print(sum(os.times()[:2]))"
    res = {}
    for cores in (0.1, 0.25, 0.5, 1.0):
        job = make_job(cores, None)
        p = subprocess.Popen([sys.executable, "-I", "-c", code], stdout=subprocess.PIPE, text=True)
        assign(job, p.pid)
        cpu = float(p.communicate()[0].strip())
        res[cores] = round(cpu / 6, 3)
        k32.CloseHandle(job)
        log(f"cap {cores} cores: busy loop used {res[cores]} CPU-seconds per second")
    return res


# -------------------------------------------------------------------------------------------- HTTP (stdlib only)
def http(method, url, key, fields=None, files=None, timeout=600):
    """Same multipart client as the rebuild check (e5_rebuild_check.py)."""
    headers = {"X-API-Key": key}
    body = None
    if method == "POST":
        boundary = uuid.uuid4().hex
        buf = io.BytesIO()
        for k, v in (fields or {}).items():
            buf.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
        for k, data in (files or {}).items():
            buf.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{k}.png\"\r\n"
                      f"Content-Type: image/png\r\n\r\n".encode())
            buf.write(data)
            buf.write(b"\r\n")
        buf.write(f"--{boundary}--\r\n".encode())
        body = buf.getvalue()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, None
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        return f"error: {type(e).__name__}", None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pack(path):
    """Same reader as the rebuild check."""
    zf = zipfile.ZipFile(path)
    rows = list(csv.DictReader(io.TextIOWrapper(zf.open("answer_sheet.csv"), encoding="utf-8")))
    cases = []
    for r in rows:
        c = r["case"]
        if r["modality"] == "MRI":
            files = {"file": f"{c}/1_DWI_main_image.png", "adc_file": f"{c}/2_ADC.png"}
        else:
            files = {"file": f"{c}/2_main_slice.png", "prev_file": f"{c}/1_previous_slice.png",
                     "next_file": f"{c}/3_next_slice.png"}
        cases.append({"case": c, "modality": r["modality"], "true_label": r["true_label"],
                      "offline_prediction": r["offline_prediction_deployed_rule"],
                      "offline_probs": [float(r[f"offline_prob_{l}"]) for l in LABELS],
                      "bytes": {k: zf.read(v) for k, v in files.items()}})
    return cases, sha256(path)


# ------------------------------------------------------------------------------------------------------- run
def one(base, key, c, variant, cores, phase, rnd):
    t0 = time.perf_counter()
    code, body = http("POST", f"{base}/api/predict", key, {"modality": c["modality"]}, c["bytes"])
    rec = {"variant": variant, "cores": cores, "phase": phase, "round": rnd, "case": c["case"], "http": code,
           "client_ms": round(1000 * (time.perf_counter() - t0), 1)}
    if code == 200 and body:
        probs = [float(body[f"prob_{l}"]) for l in LABELS]
        rec.update({"prediction": body["prediction"], "same_as_answer_sheet": body["prediction"] == c["offline_prediction"],
                    **{f"prob_{l}": round(p, 6) for l, p in zip(LABELS, probs)},
                    "max_abs_prob_diff_vs_sheet": round(max(abs(x - y) for x, y in zip(probs, c["offline_probs"])), 6),
                    "server_model_ms": body.get("timing_ms", {}).get("model"),
                    "server_total_ms": body.get("timing_ms", {}).get("total"),
                    "heatmap": bool(body.get("heatmap_png_base64"))})
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--repo"), ap.add_argument("--python"), ap.add_argument("--pack"), ap.add_argument("--models")
    ap.add_argument("--out")
    ap.add_argument("--caps", nargs="+", type=float, default=[0.1, 0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--mem-mb", type=float, default=512)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--clients", type=int, default=4)
    a = ap.parse_args()
    if a.selftest:
        print(json.dumps(selftest()))
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(a.repo)
    info = {"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "host": {"os": platform.platform(), "cpu": platform.processor(), "logical_cpus": NCPU,
                     "python_server": subprocess.run([a.python, "-c", "import sys;print(sys.version)"],
                                                     capture_output=True, text=True).stdout.strip()},
            "script_sha256": sha256(__file__), "caps_cores": a.caps, "memory_limit_mb": a.mem_mb,
            "rounds": a.rounds, "clients": a.clients,
            "commit": subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                                     text=True).stdout.strip(),
            "server_files_sha256": {f: sha256(repo / f) for f in ("app.py", "inference.py", "requirements.lock")}}
    info["selftest_cpu_per_wall_second"] = selftest()
    # a private copy of the server with the model files where inference.py expects them (models/)
    srv = out / "server_copy"
    (srv / "models").mkdir(parents=True, exist_ok=True)
    for f in ("app.py", "inference.py"):
        shutil.copy2(repo / f, srv / f)
    for v, f in MODEL_FILES.items():
        dst = srv / "models" / f
        if not dst.exists():
            shutil.copy2(Path(a.models) / f, dst)
    cases, info["pack_sha256"] = load_pack(a.pack)
    key = secrets.token_urlsafe(24)  # test value for this local run only; never printed or saved
    base = f"http://127.0.0.1:{PORT}"
    rows, info["runs"] = [], {}
    for variant in MODEL_FILES:
        for cores in a.caps:
            tag = f"{variant} @ {cores} cores"
            job = make_job(cores, a.mem_mb)
            env = dict(os.environ, NEUROSCAN_API_KEY=key, NEUROSCAN_MODEL_VARIANT=variant, PYTHONUNBUFFERED="1")
            slog = open(out / f"server_{variant}_{cores}.log", "w", encoding="utf-8")
            t_start = time.perf_counter()
            proc = subprocess.Popen([a.python, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(PORT)],
                                    cwd=srv, env=env, stdout=slog, stderr=subprocess.STDOUT)
            assign(job, proc.pid)
            run = {"cores": cores, "variant": variant}
            try:
                health = None
                while time.perf_counter() - t_start < 900 and proc.poll() is None:
                    code, health = http("GET", f"{base}/api/health", key, timeout=5)
                    if code == 200:
                        break
                    health = None
                    time.sleep(0.5)
                run["ready_seconds"] = round(time.perf_counter() - t_start, 1)
                if not health:
                    run["failed"] = f"server not ready (exit {proc.poll()}); see server log"
                    log(f"{tag}: {run['failed']}")
                    continue
                ck = health["checkpoint"]
                run["served_variant_and_sha_ok"] = (ck["model_variant"] == variant and
                                                    ck["weights_sha256"] == sha256(srv / "models" / MODEL_FILES[variant]))
                log(f"{tag}: ready in {run['ready_seconds']} s; variant/sha ok {run['served_variant_and_sha_ok']}")
                for c in cases[:2]:
                    one(base, key, c, variant, cores, "warmup", 0)
                seq = {}
                for rnd in range(1, a.rounds + 1):
                    for c in cases:
                        r = one(base, key, c, variant, cores, "sequential", rnd)
                        rows.append(r)
                        seq.setdefault(c["case"], r)
                    log(f"{tag}: sequential round {rnd} done")
                # concurrent: `clients` threads, each sending its share of the cases at the same time
                share = [cases[i::a.clients] for i in range(a.clients)]
                got, t0 = [], time.perf_counter()

                def client(lst):
                    for c in lst:
                        got.append(one(base, key, c, variant, cores, "concurrent", 1))

                th = [threading.Thread(target=client, args=(s,)) for s in share]
                for t in th:
                    t.start()
                for t in th:
                    t.join()
                run["concurrent_wall_seconds"] = round(time.perf_counter() - t0, 1)
                for r in got:
                    s = seq.get(r["case"], {})
                    r["same_as_sequential"] = (r.get("prediction") == s.get("prediction") and
                                               all(r.get(f"prob_{l}") == s.get(f"prob_{l}") for l in LABELS))
                rows.extend(got)
                ok = [r for r in rows if r["variant"] == variant and r["cores"] == cores and r["http"] == 200]
                sq = sorted(r["client_ms"] for r in ok if r["phase"] == "sequential")
                cc = sorted(r["client_ms"] for r in ok if r["phase"] == "concurrent")
                md = sorted(r["server_model_ms"] for r in ok if r["phase"] == "sequential")
                q = lambda v, p: v[min(len(v) - 1, int(p * len(v)))] if v else None
                run.update({
                    "requests": a.rounds * len(cases) + len(cases),
                    "answered": len(ok),
                    "failed_requests": a.rounds * len(cases) + len(cases) - len(ok),
                    "sequential_client_ms": {"median": q(sq, .5), "p95": q(sq, .95), "max": sq[-1] if sq else None},
                    "sequential_server_model_ms_median": q(md, .5),
                    "throughput_per_minute_sequential": round(60000 / q(sq, .5), 1) if sq else None,
                    "concurrent_client_ms": {"median": q(cc, .5), "max": cc[-1] if cc else None},
                    "same_prediction_as_answer_sheet": sum(r["same_as_answer_sheet"] for r in ok),
                    "max_abs_prob_diff_vs_sheet": max((r["max_abs_prob_diff_vs_sheet"] for r in ok), default=None),
                    "concurrent_same_as_sequential": sum(bool(r.get("same_as_sequential")) for r in got),
                    "concurrent_answered": sum(r["http"] == 200 for r in got),
                    "all_heatmaps": all(r["heatmap"] for r in ok),
                    "server_alive_at_end": proc.poll() is None})
                log(f"{tag}: {json.dumps({k: run[k] for k in ('sequential_client_ms', 'sequential_server_model_ms_median', 'failed_requests', 'same_prediction_as_answer_sheet', 'concurrent_same_as_sequential')})}")
            finally:
                try:
                    run["peak_job_memory_mb"] = peak_job_mb(job)
                except OSError:
                    pass
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                k32.CloseHandle(job)
                slog.close()
                info["runs"][f"{variant}|{cores}"] = run
                (out / "l1_results.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
                time.sleep(2)
    with open(out / "l1_requests.csv", "w", newline="", encoding="utf-8") as f:
        cols = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    info["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (out / "l1_results.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    log(f"done; outputs in {out}")


if __name__ == "__main__":
    main()
