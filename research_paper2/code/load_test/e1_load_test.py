"""
NeuroScan paper 2 -- experiment E1: cold start, latency, concurrency and deployment fidelity of the inference service.

Run by the author on their own PC, against the URL and inside the time window the author chose. The API key is read
from the environment variable NEUROSCAN_API_KEY and is never printed, logged or written to any output file.

Inputs: the web-app test pack (neuroscan_test_pack.zip from kernel neuroscan-q-test-pack): 28 held-out TEST cases from
public, de-identified datasets (16 MRI DWI + ADC from ISLES 2022 / SOOP, 12 CT previous/main/next from AISD / CPAISD),
with an answer sheet of the final model's saved offline predictions. The zip is read in place, never extracted.
No real patient images.

Phases (in this order; every request is logged):
  1. cold      GET /api/health on the (sleeping) service, then one POST /api/predict: time to first answer. Render's
               free tier sleeps after 15 min without traffic, so the service must have been idle >= 16 min before the
               run. --cold-repeats N > 1 waits --idle-minutes between repeats.
  2. fidelity  each of the 28 cases once, one at a time: the live prediction and probabilities against the offline
               answer sheet (same model, pathway and rule, so they should agree to about 3 decimals).
  3. warm      --warm-n sequential requests cycling through the three request kinds (CT with context, MRI DWI+ADC,
               CT single slice): the latency distribution for one user.
  4. steps     closed-loop concurrency steps (default 1, 2, 4, 8 parallel users), --step-n requests each: throughput,
               latency and failures per step.
Safety limits (fixed): total requests <= --max-requests (default 400); a step whose failure rate exceeds 20% ends the
escalation; 5 consecutive failures anywhere end the run. Requests time out after 120 s. A non-local URL needs --live
and a --window START END (ISO 8601 with time zone, e.g. 2026-10-25T02:00+06:00); the script refuses to start, and
stops sending, outside that window.

Outputs (in --out, default ./e1_results_<UTC time>/): requests.csv (one row per request, no image data, no key),
summary.json (per-phase statistics), run_info.json (settings, URL host, script SHA-256, library versions).

Usage:
  set NEUROSCAN_API_KEY=...        (Windows cmd; PowerShell: $env:NEUROSCAN_API_KEY="..."; never pasted into chat)
  python -I e1_load_test.py --pack <path>/neuroscan_test_pack.zip --url https://<service>.onrender.com ^
         --live --window 2026-10-25T02:00+06:00 2026-10-25T04:00+06:00
  python -I e1_load_test.py --pack <zip> --self-test        (no network: checks the script with simulated answers)
"""
import argparse
import csv
import hashlib
import io
import json
import os
import platform
import random
import statistics
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

LABELS = ("normal", "ischemic", "hemorrhagic")
TIMEOUT_S = 120
FAIL_STEP_LIMIT = 0.20
FAIL_STREAK_LIMIT = 5
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
KINDS = ("ct_context", "mri_dwi_adc", "ct_single")


class Stop(Exception):
    """A safety limit was reached; the run ends and everything measured so far is written."""


# ----------------------------------------------------------------------------------------------- test pack
def load_pack(path):
    zf = zipfile.ZipFile(path)
    names = set(zf.namelist())
    rows = list(csv.DictReader(io.TextIOWrapper(zf.open("answer_sheet.csv"), encoding="utf-8")))
    cases = []
    for r in rows:
        c = r["case"]
        if r["modality"] == "MRI":
            files = {"file": f"{c}/1_DWI_main_image.png", "adc_file": f"{c}/2_ADC.png"}
        else:
            files = {"file": f"{c}/2_main_slice.png", "prev_file": f"{c}/1_previous_slice.png",
                     "next_file": f"{c}/3_next_slice.png"}
        missing = [f for f in files.values() if f not in names]
        if missing:
            sys.exit(f"test pack: missing {missing}")
        cases.append({"case": c, "modality": r["modality"], "true_label": r["true_label"],
                      "offline_prediction": r["offline_prediction_deployed_rule"],
                      "offline_probs": [float(r[f"offline_prob_{l}"]) for l in LABELS],
                      "bytes": {k: zf.read(v) for k, v in files.items()}})
    return cases


def request_for(case, kind):
    """(modality, {field: bytes}) for one request kind; ct_single sends the main CT slice alone."""
    b = case["bytes"]
    if kind == "ct_single":
        return "CT", {"file": b["file"]}
    return case["modality"], dict(b)


# ------------------------------------------------------------------------------------------------ transport
class Client:
    def __init__(self, url, key, self_test=False):
        self.url, self.key, self.self_test = url.rstrip("/"), key, self_test
        self.local = threading.local()
        self.rng = random.Random(0)

    def _session(self):
        if not hasattr(self.local, "s"):
            self.local.s = requests.Session()
            self.local.s.headers["X-API-Key"] = self.key
        return self.local.s

    def health(self):
        if self.self_test:
            time.sleep(0.01)
            return 200, {"status": "ok", "checkpoint": {"model_version": "self_test", "model_variant": "self_test"},
                         "memory_mb": {"rss": 0, "peak": 0}, "uptime_s": 0.0}, 50
        r = self._session().get(f"{self.url}/api/health", timeout=TIMEOUT_S)
        return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else None), len(r.content)

    def predict(self, modality, files, case):
        if self.self_test:
            time.sleep(self.rng.uniform(0.01, 0.03))
            if self.rng.random() < 0.02:
                return 503, None, 0
            p = case["offline_probs"]
            return 200, {"prediction": LABELS[p.index(max(p))], "prob_normal": p[0], "prob_ischemic": p[1],
                         "prob_hemorrhagic": p[2], "pathway": "self_test", "warnings": [], "heatmap_png_base64": "x",
                         "model_version": "self_test", "timing_ms": {"model": 1.0, "request": 2.0}}, 900
        up = {k: (f"{k}.png", v, "image/png") for k, v in files.items()}
        r = self._session().post(f"{self.url}/api/predict", files=up, data={"modality": modality}, timeout=TIMEOUT_S)
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else None
        return r.status_code, body, len(r.content)


# -------------------------------------------------------------------------------------------------- the run
class Run:
    def __init__(self, args, client, cases):
        self.a, self.c, self.cases = args, client, cases
        self.rows, self.lock, self.sent, self.streak = [], threading.Lock(), 0, 0
        self.t0 = time.time()
        self.window = [datetime.fromisoformat(w) for w in args.window] if args.window else None

    def _guard(self):
        if self.sent >= self.a.max_requests:
            raise Stop(f"request cap {self.a.max_requests} reached")
        if self.window and not (self.window[0] <= datetime.now(timezone.utc) <= self.window[1]):
            raise Stop("outside the approved time window")
        if self.streak >= FAIL_STREAK_LIMIT:
            raise Stop(f"{FAIL_STREAK_LIMIT} consecutive failures")

    def one(self, phase, step, case, kind):
        with self.lock:
            self._guard()
            self.sent += 1
            n = self.sent
        modality, files = request_for(case, kind)
        t_start = time.time()
        status, body, nbytes, err = None, None, 0, ""
        try:
            status, body, nbytes = self.c.predict(modality, files, case)
        except requests.Timeout:
            err = "timeout"
        except requests.RequestException as e:
            err = type(e).__name__
        lat = time.time() - t_start
        ok = status == 200 and isinstance(body, dict) and "prediction" in body
        row = {"n": n, "phase": phase, "concurrency": step, "case": case["case"], "kind": kind,
               "t_start_s": round(t_start - self.t0, 3), "latency_s": round(lat, 4), "status": status, "error": err,
               "ok": int(ok), "response_bytes": nbytes, "prediction": body.get("prediction") if ok else "",
               "pathway": body.get("pathway") if ok else "", "warnings": len(body.get("warnings", [])) if ok else "",
               "heatmap": int(bool(body.get("heatmap_png_base64"))) if ok else "",
               "detail": (str(body.get("detail"))[:200] if isinstance(body, dict) and not ok else "")}
        if ok:
            for i, l in enumerate(LABELS):
                row[f"prob_{l}"] = body.get(f"prob_{l}")
            tm = body.get("timing_ms") or {}   # servers from 2026-10-25 on report their own time per stage
            row["model_version"] = body.get("model_version", "")
            row["server_model_ms"], row["server_request_ms"] = tm.get("model", ""), tm.get("request", "")
        if kind != "ct_single":
            row["offline_prediction"] = case["offline_prediction"]
            if ok:
                row["matches_offline"] = int(body["prediction"] == case["offline_prediction"])
                row["max_abs_prob_diff_vs_offline"] = round(max(abs(float(body[f"prob_{l}"]) - case["offline_probs"][i])
                                                                for i, l in enumerate(LABELS)), 5)
        with self.lock:
            self.streak = 0 if ok else self.streak + 1
            self.rows.append(row)
        return row

    def cycle(self):
        """Endless (case, kind) sequence: CT with context, MRI, CT single, in turn, cases shuffled with seed 0."""
        rng = random.Random(0)
        ct = [c for c in self.cases if c["modality"] == "CT"]
        mri = [c for c in self.cases if c["modality"] == "MRI"]
        while True:
            for pool in (ct, mri):
                rng.shuffle(pool)
            for i in range(max(len(ct), len(mri))):
                yield ct[i % len(ct)], "ct_context"
                yield mri[i % len(mri)], "mri_dwi_adc"
                yield ct[(i + 3) % len(ct)], "ct_single"

    def health_snapshot(self, summary, tag):
        """Server memory (current, peak), uptime and model variant after a phase; not counted as a load request."""
        if self.window and not (self.window[0] <= datetime.now(timezone.utc) <= self.window[1]):
            return
        try:
            st, body, _ = self.c.health()
            b = body or {}
            summary.setdefault("server_health", {})[tag] = {
                "status": st, "memory_mb": b.get("memory_mb"), "uptime_s": b.get("uptime_s"),
                "model_variant": b.get("checkpoint", {}).get("model_variant"),
                "model_version": b.get("checkpoint", {}).get("model_version")}
        except requests.RequestException as e:
            summary.setdefault("server_health", {})[tag] = {"error": type(e).__name__}

    def phase_cold(self, summary):
        out = []
        for rep in range(self.a.cold_repeats):
            if rep:
                print(f"  waiting {self.a.idle_minutes} min idle before cold repeat {rep + 1}", flush=True)
                time.sleep(self.a.idle_minutes * 60)
            with self.lock:
                self._guard()
                self.sent += 1
            t = time.time()
            try:
                st, body, _ = self.c.health()
                err = ""
            except requests.RequestException as e:
                st, body, err = None, None, type(e).__name__
            health_s = time.time() - t
            row = self.one("cold", 1, self.cases[0], "mri_dwi_adc" if self.cases[0]["modality"] == "MRI" else "ct_context")
            out.append({"repeat": rep + 1, "health_status": st, "health_error": err, "health_seconds": round(health_s, 3),
                        "first_predict_seconds": row["latency_s"], "first_predict_ok": row["ok"],
                        "model_version": (body or {}).get("checkpoint", {}).get("model_version")})
            print(f"  cold {rep + 1}: health {st} in {health_s:.1f}s, first predict {row['latency_s']:.2f}s ok={row['ok']}", flush=True)
        summary["cold"] = out

    def phase_fidelity(self, summary):
        rows = [self.one("fidelity", 1, c, "mri_dwi_adc" if c["modality"] == "MRI" else "ct_context") for c in self.cases]
        ok = [r for r in rows if r["ok"]]
        summary["fidelity"] = {"cases": len(rows), "answered": len(ok),
                               "same_prediction_as_offline": sum(r["matches_offline"] for r in ok),
                               "max_abs_prob_diff_vs_offline": max((r["max_abs_prob_diff_vs_offline"] for r in ok), default=None),
                               "heatmaps_returned": sum(r["heatmap"] for r in ok)}
        print(f"  fidelity: {summary['fidelity']}", flush=True)

    def phase_warm(self, summary, seq):
        for _ in range(self.a.warm_n):
            self.one("warm", 1, *next(seq))
        summary["warm"] = stats([r for r in self.rows if r["phase"] == "warm"])
        print(f"  warm: {summary['warm']['all']}", flush=True)

    def phase_steps(self, summary, seq):
        summary["steps"] = {}
        seq_lock = threading.Lock()

        def worker(level, quota):
            while True:
                with seq_lock:
                    if quota[0] <= 0:
                        return
                    quota[0] -= 1
                    item = next(seq)
                self.one("steps", level, *item)

        for level in self.a.steps:
            t = time.time()
            quota = [self.a.step_n]
            with ThreadPoolExecutor(level) as ex:
                futs = [ex.submit(worker, level, quota) for _ in range(level)]
                for f in futs:
                    f.result()   # re-raises Stop
            wall = time.time() - t
            rows = [r for r in self.rows if r["phase"] == "steps" and r["concurrency"] == level]
            s = stats(rows)
            s["wall_seconds"] = round(wall, 2)
            s["throughput_ok_per_min"] = round(60 * sum(r["ok"] for r in rows) / wall, 2) if wall else None
            summary["steps"][str(level)] = s
            print(f"  step {level}: {s['all']} throughput {s['throughput_ok_per_min']}/min", flush=True)
            if s["failure_rate"] > FAIL_STEP_LIMIT:
                summary["steps_stopped"] = f"failure rate {s['failure_rate']:.2f} > {FAIL_STEP_LIMIT} at {level} users"
                print(f"  stopping escalation: {summary['steps_stopped']}", flush=True)
                break


def stats(rows):
    def s(v):
        if not v:
            return {"n": 0}
        v = sorted(v)
        return {"n": len(v), "median_s": round(statistics.median(v), 3), "p95_s": round(v[min(len(v) - 1, int(0.95 * len(v)))], 3),
                "max_s": round(v[-1], 3), "mean_s": round(statistics.fmean(v), 3)}
    okr = [r for r in rows if r["ok"]]
    out = {"requests": len(rows), "failures": len(rows) - len(okr),
           "failure_rate": round((len(rows) - len(okr)) / len(rows), 4) if rows else 0.0,
           "failure_kinds": {}, "all": s([r["latency_s"] for r in okr])}
    for r in rows:
        if not r["ok"]:
            k = r["error"] or f"HTTP {r['status']}"
            out["failure_kinds"][k] = out["failure_kinds"].get(k, 0) + 1
    for k in KINDS:
        out[k] = s([r["latency_s"] for r in okr if r["kind"] == k])
    return out


# ------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pack", required=True, help="neuroscan_test_pack.zip")
    ap.add_argument("--url", default=os.environ.get("NEUROSCAN_URL", ""))
    ap.add_argument("--live", action="store_true", help="required for any non-local URL")
    ap.add_argument("--window", nargs=2, metavar=("START", "END"), help="approved window, ISO 8601 with time zone")
    ap.add_argument("--self-test", action="store_true", help="no network; simulated answers")
    ap.add_argument("--phases", nargs="+", default=["cold", "fidelity", "warm", "steps"],
                    choices=["cold", "fidelity", "warm", "steps"])
    ap.add_argument("--cold-repeats", type=int, default=1)
    ap.add_argument("--idle-minutes", type=float, default=16)
    ap.add_argument("--warm-n", type=int, default=60)
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--step-n", type=int, default=40)
    ap.add_argument("--max-requests", type=int, default=400)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if a.self_test:
        a.url, key, a.window = "http://self-test.invalid", "self-test", None
        a.idle_minutes = min(a.idle_minutes, 0.01)
    else:
        key = os.environ.get("NEUROSCAN_API_KEY", "")
        if not key:
            sys.exit("NEUROSCAN_API_KEY is not set (set it in this terminal; never paste it into chat or a file).")
        host = urlparse(a.url).hostname or ""
        if not host:
            sys.exit("--url (or NEUROSCAN_URL) is required.")
        if host not in LOCAL_HOSTS:
            if not a.live or not a.window:
                sys.exit(f"{host} is not local: --live and --window START END are required.")
            w = [datetime.fromisoformat(x) for x in a.window]
            if any(x.tzinfo is None for x in w):
                sys.exit("--window times need a time zone, e.g. 2026-10-25T02:00+06:00")
            if not (w[0] <= datetime.now(timezone.utc) <= w[1]):
                sys.exit(f"now ({datetime.now().astimezone().isoformat(timespec='minutes')}) is outside the approved window.")
        if max(a.steps) > 8:
            print("note: steps above 8 parallel users go beyond the planned test", flush=True)

    cases = load_pack(a.pack)
    planned = a.cold_repeats * 2 * ("cold" in a.phases) + len(cases) * ("fidelity" in a.phases) + \
        a.warm_n * ("warm" in a.phases) + a.step_n * len(a.steps) * ("steps" in a.phases)
    out = Path(a.out or f"e1_results_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%MZ')}{'_selftest' if a.self_test else ''}")
    out.mkdir(parents=True, exist_ok=True)
    info = {"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "url_host": urlparse(a.url).hostname, "self_test": a.self_test, "phases": a.phases,
            "cold_repeats": a.cold_repeats, "idle_minutes": a.idle_minutes, "warm_n": a.warm_n, "steps": a.steps,
            "step_n": a.step_n, "max_requests": a.max_requests, "planned_requests": planned, "timeout_s": TIMEOUT_S,
            "window": a.window, "test_pack_sha256": hashlib.sha256(Path(a.pack).read_bytes()).hexdigest(),
            "test_pack_cases": len(cases), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python": platform.python_version(), "requests": requests.__version__, "client_os": platform.platform()}
    print(f"E1: {len(cases)} cases, phases {a.phases}, planned {planned} requests (cap {a.max_requests}) -> {out}", flush=True)

    run = Run(a, Client(a.url, key, a.self_test), cases)
    summary, seq = {}, run.cycle()
    try:
        for ph in a.phases:
            print(f"phase {ph}", flush=True)
            if ph == "cold":
                run.phase_cold(summary)
            elif ph == "fidelity":
                run.phase_fidelity(summary)
            elif ph == "warm":
                run.phase_warm(summary, seq)
            else:
                run.phase_steps(summary, seq)
            run.health_snapshot(summary, f"after_{ph}")
    except Stop as e:
        summary["stopped"] = str(e)
        print(f"STOPPED: {e}", flush=True)
    except KeyboardInterrupt:
        summary["stopped"] = "interrupted by the operator"
        print("interrupted; writing what was measured", flush=True)
    finally:
        info["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        info["requests_sent"] = run.sent
        cols = sorted({k for r in run.rows for k in r}, key=lambda k: (list(run.rows[0]).index(k) if k in run.rows[0] else 99, k)) if run.rows else []
        with open(out / "requests.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(run.rows)
        json.dump(summary, open(out / "summary.json", "w"), indent=2)
        json.dump(info, open(out / "run_info.json", "w"), indent=2)
        print(f"written: {out / 'requests.csv'} ({len(run.rows)} rows), summary.json, run_info.json", flush=True)


if __name__ == "__main__":
    main()
