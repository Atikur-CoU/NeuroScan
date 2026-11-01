"""
NeuroScan paper 2 -- experiment E5: can the inference service be rebuilt from the repository alone, and does the
rebuilt copy give the evaluated answers?

Steps (each one timed and logged):
  1. clone     fresh `git clone` of the committed repository into an empty folder (no working-tree files carried over);
               record the commit and check the tree is clean.
  2. sums      every file listed in research/SHA256SUMS.txt against its SHA-256 in the clone.
  3. venv      new virtual environment; install requirements.lock with hash checking, as the Dockerfile does
               (requirements.txt for commits before the lock file); every installed package must be at its pinned version.
  4. models    copy the model files into the clone's models/ (standing in for the GitHub release download in start.sh)
               and check each SHA-256 against the value pinned in the clone's inference.py.
  5. serve     for each variant (fp32, int8_dynamic): start the clone's app with uvicorn on 127.0.0.1, wait for
               /api/health, check the variant and weights SHA-256 it reports, then send the 28 test-pack cases
               (held-out TEST cases, public de-identified datasets) once each and compare with the answer sheet of
               saved offline predictions. FP32 should match to about 3 decimals; INT8-dynamic is reported as is.
Local only: the server listens on 127.0.0.1, the API key is a random test value made here and never written anywhere.
This is not the Docker image: Python and OS differ from Render's container (recorded in the output).

Usage:
  python -I e5_rebuild_check.py --repo <render_inference_server> --pack <neuroscan_test_pack.zip>
         --model fp32=<path>.onnx --model int8_dynamic=<path>.onnx --work <empty folder> --out <folder>
"""
import argparse
import csv
import hashlib
import io
import json
import os
import platform
import re
import secrets
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

LABELS = ("normal", "ischemic", "hemorrhagic")
PORT = 8765
STEPS = []


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def step(name):
    """Context manager-free timer: call once at the start, the returned function at the end."""
    t0 = time.perf_counter()

    def done(ok, detail=""):
        STEPS.append({"step": name, "seconds": round(time.perf_counter() - t0, 1), "ok": bool(ok), "detail": detail})
        log(f"{name}: {'OK' if ok else 'FAILED'} in {STEPS[-1]['seconds']} s  {detail}")
        return ok
    return done


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.stdout.strip()


# -------------------------------------------------------------------------------------------- HTTP (stdlib only)
def http(method, url, key, fields=None, files=None, timeout=120):
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


def load_pack(path):
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


# ------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--model", action="append", required=True, help="variant=path, e.g. fp32=models/x.onnx")
    ap.add_argument("--work", required=True, help="empty folder for the clone and the virtual environment")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    work, out = Path(a.work), Path(a.out)
    if work.exists() and any(work.iterdir()):
        sys.exit(f"--work {work} must be empty")
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    models = dict(m.split("=", 1) for m in a.model)
    info = {"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "host": {"os": platform.platform(), "python": sys.version.split()[0], "machine": platform.machine()},
            "script_sha256": sha256(__file__), "note": "local rebuild without Docker; Render builds python:3.11-slim"}
    t_all = time.perf_counter()
    clone = work / "repo"

    # 1. clone
    done = step("clone")
    src_head = run(["git", "-C", a.repo, "rev-parse", "HEAD"])
    # The repository folder is owned by another Windows account (drive created elsewhere); trust it for this command only.
    run(["git", "-c", "safe.directory=*", "clone", "--quiet", "--no-hardlinks", a.repo, str(clone)])
    head = run(["git", "-C", str(clone), "rev-parse", "HEAD"])
    status = run(["git", "-C", str(clone), "status", "--porcelain"])
    tracked = run(["git", "-C", str(clone), "ls-files"]).splitlines()
    info["commit"] = head
    done(head == src_head and status == "", f"commit {head[:7]}, {len(tracked)} tracked files")

    # 2. research/SHA256SUMS.txt
    done = step("sums")
    bad, n = [], 0
    for line in (clone / "research" / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(None, 1)
        rel = rel.lstrip("*").strip()
        p = clone / "research" / rel
        n += 1
        if not p.is_file() or sha256(p) != digest:
            bad.append(rel)
    info["sums"] = {"listed": n, "mismatched_or_missing": bad}
    done(not bad, f"{n - len(bad)}/{n} files match")

    # 3. virtual environment with the pinned requirements
    done = step("venv")
    venv = work / "venv"
    run([sys.executable, "-m", "venv", str(venv)])
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    # The lock file (every package, hash-checked) when the repository has one, as the Dockerfile does; else the top-level pins.
    lock = clone / "requirements.lock"
    req = lock if lock.is_file() else clone / "requirements.txt"
    run([str(py), "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
         *(["--require-hashes"] if req == lock else []), "-r", str(req)])
    freeze = run([str(py), "-m", "pip", "freeze", "--disable-pip-version-check"]).splitlines()
    canon = lambda n: re.sub(r"[-_.]+", "-", re.sub(r"\[.*\]", "", n)).lower()
    have = {canon(l.split("==")[0]): l.split("==")[1] for l in freeze if "==" in l}
    # name -> (version, has an environment marker); markers name packages for other platforms, which may be absent
    pins = {canon(m[0]): (m[1], bool(m[2])) for m in
            re.findall(r"^([A-Za-z0-9_.\-\[\]]+)==([^\s;\\]+)\s*(;[^\\\n]*)?", req.read_text(), re.M)}
    unmet = [f"{n}=={v}" for n, (v, marked) in pins.items() if have.get(n) != v and not (marked and n not in have)]
    extra = sorted(set(have) - set(pins)) if req == lock else []
    info["requirements_file"] = req.name
    info["pip_freeze"] = freeze
    done(not unmet and not extra, f"{req.name}: {len(have)} packages installed, all at the pinned versions"
         if not unmet and not extra else f"{req.name}: unmet {unmet}; not in lock {extra}")

    # 4. model files against the SHA-256 pinned in inference.py
    done = step("models")
    src = (clone / "inference.py").read_text(encoding="utf-8")
    pins = {v: (f, d) for v, f, d in re.findall(r'"(\w+)":\s*\(\s*"([^"]+\.onnx)",\s*"([0-9a-f]{64})"', src)}
    (clone / "models").mkdir(exist_ok=True)
    info["models"] = {}
    ok = True
    for variant, path in models.items():
        fname, pinned_sha = pins[variant]
        dst = clone / "models" / fname
        shutil.copy2(path, dst)
        got = sha256(dst)
        info["models"][variant] = {"file": fname, "bytes": dst.stat().st_size, "sha256": got,
                                   "pinned_sha256": pinned_sha, "match": got == pinned_sha}
        ok &= got == pinned_sha
    done(ok, ", ".join(f"{v} {'match' if m['match'] else 'MISMATCH'}" for v, m in info["models"].items()))

    # 5. serve each variant and replay the test pack
    cases, info["pack_sha256"] = load_pack(a.pack)
    key = secrets.token_urlsafe(24)  # test value for this local run only; never printed or saved
    base = f"http://127.0.0.1:{PORT}"
    rows = []
    info["serve"] = {}
    for variant in models:
        done = step(f"serve_{variant}")
        env = dict(os.environ, NEUROSCAN_API_KEY=key, NEUROSCAN_MODEL_VARIANT=variant, PYTHONUNBUFFERED="1")
        srv_log = open(out / f"server_{variant}.log", "w", encoding="utf-8")
        t_start = time.perf_counter()
        proc = subprocess.Popen([str(py), "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(PORT)],
                                cwd=clone, env=env, stdout=srv_log, stderr=subprocess.STDOUT)
        health = None
        try:
            while time.perf_counter() - t_start < 180 and proc.poll() is None:
                try:
                    code, health = http("GET", f"{base}/api/health", key, timeout=5)
                    if code == 200:
                        break
                except (urllib.error.URLError, ConnectionError, TimeoutError):
                    pass
                time.sleep(0.5)
            ready_s = round(time.perf_counter() - t_start, 1)
            if not health:
                done(False, "server did not become ready (see server log)")
                continue
            ck = health["checkpoint"]
            served_ok = ck["model_variant"] == variant and ck["weights_sha256"] == info["models"][variant]["sha256"]
            n_match = 0
            for c in cases:
                t0 = time.perf_counter()
                code, body = http("POST", f"{base}/api/predict", key, {"modality": c["modality"]}, c["bytes"])
                wall = round(1000 * (time.perf_counter() - t0), 1)
                if code != 200 or body is None:
                    rows.append({"variant": variant, "case": c["case"], "http": code})
                    continue
                probs = [float(body[f"prob_{l}"]) for l in LABELS]
                diff = max(abs(x - y) for x, y in zip(probs, c["offline_probs"]))
                same = body["prediction"] == c["offline_prediction"]
                n_match += same
                t = body.get("timing_ms", {})
                rows.append({"variant": variant, "case": c["case"], "modality": c["modality"], "http": code,
                             "true_label": c["true_label"], "offline_prediction": c["offline_prediction"],
                             "prediction": body["prediction"], "same_prediction": same,
                             **{f"prob_{l}": round(p, 4) for l, p in zip(LABELS, probs)},
                             **{f"offline_prob_{l}": p for l, p in zip(LABELS, c["offline_probs"])},
                             "max_abs_prob_diff": round(diff, 5), "pathway": body.get("pathway", ""),
                             "heatmap": bool(body.get("heatmap_png_base64")), "model_version": body.get("model_version"),
                             "server_model_ms": t.get("model"), "server_request_ms": t.get("request"),
                             "client_ms": wall})
            vr = [r for r in rows if r["variant"] == variant and r.get("http") == 200]
            diffs = sorted(r["max_abs_prob_diff"] for r in vr)
            _, health_after = http("GET", f"{base}/api/health", key)
            info["serve"][variant] = {
                "ready_seconds": ready_s, "health": health, "health_after": health_after,
                "served_variant_and_sha_ok": served_ok, "answered": len(vr), "of": len(cases),
                "same_prediction_as_answer_sheet": n_match,
                "max_abs_prob_diff": {"median": diffs[len(diffs) // 2] if diffs else None,
                                      "max": diffs[-1] if diffs else None,
                                      "n_over_0.001": sum(d > 0.001 for d in diffs)},
                "correct_vs_true_label": sum(r["prediction"] == r["true_label"] for r in vr),
                "all_heatmaps_present": all(r["heatmap"] for r in vr),
                "median_server_model_ms": sorted(r["server_model_ms"] for r in vr)[len(vr) // 2] if vr else None}
            done(served_ok and len(vr) == len(cases),
                 f"ready {ready_s} s; {n_match}/{len(cases)} same prediction; max prob diff {diffs[-1] if diffs else None}")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
            srv_log.close()
            time.sleep(1)

    info["total_seconds"] = round(time.perf_counter() - t_all, 1)
    info["steps"] = STEPS
    info["all_ok"] = all(s["ok"] for s in STEPS)
    (out / "e5_results.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    with open(out / "e5_cases.csv", "w", newline="", encoding="utf-8") as f:
        cols = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    log(f"done in {info['total_seconds']} s; all steps OK: {info['all_ok']}; outputs in {out}")


if __name__ == "__main__":
    main()
