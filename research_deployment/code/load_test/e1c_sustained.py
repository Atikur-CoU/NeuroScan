"""
NeuroScan paper 2 -- experiment E1c (JTCSST second revision, comment 5): sustained open-loop load on one live service.

Why: the earlier load tests held each rate for a few minutes. The reviewer asks how the service behaves under a
sustained workload.

What: after --warm-n sequential requests, requests arrive as a Poisson process at one fixed rate (--rate per minute)
for --minutes, each sent at its scheduled time whether or not earlier ones were answered (open loop, as in E1b). Every
--health-every minutes the script reads /api/health (memory, uptime, model variant; not counted as a load request), so
a restart of the service shows as a drop in uptime. Run by the author on their own PC, against the URL and inside the
time window the author approved; for the two matched services, run one copy per service at the same time, with the
same seed (same schedule and the same sequence of cases).

Reported (summary.json): latency, failures and achieved rate over the whole run and per --bin-minutes bin; first half
vs second half; fidelity of every request that has a saved answer (CT with context, MRI DWI + ADC: prediction and
largest probability difference against the answer sheet), over the whole run; memory and uptime at every health check;
the M/D/1 mean response for comparison (S from the warm-up).

Uses the E1 client unchanged (e1_load_test.py: test pack, request kinds, 120 s timeout, logging). The API key is read
from NEUROSCAN_API_KEY and is never printed, logged or written to any output file.
Safety limits (fixed): total requests <= --max-requests (default 450); 5 consecutive failures end the run; at most 64
requests in flight; rate <= 20 per minute; duration <= 60 minutes; a non-local URL needs --live and --window START END,
and the script stops sending outside the window.

Usage:
  python -I e1c_sustained.py --pack <zip> --url https://<service>.onrender.com --live ^
         --window 2026-11-26T02:00+06:00 2026-11-26T03:30+06:00 --rate 10 --minutes 40 --out e1c_fp32
  python -I e1c_sustained.py --pack <zip> --self-test        (no network: checks the script with simulated answers)
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1_load_test as E  # noqa: E402

MAX_IN_FLIGHT = 64


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--url", default=os.environ.get("NEUROSCAN_URL", ""))
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--window", nargs=2, metavar=("START", "END"))
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--warm-n", type=int, default=6)
    ap.add_argument("--rate", type=float, default=10)
    ap.add_argument("--minutes", type=float, default=40)
    ap.add_argument("--health-every", type=float, default=5)
    ap.add_argument("--bin-minutes", type=float, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-requests", type=int, default=450)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if a.rate > 20 or a.minutes > 60:
        sys.exit("rate above 20 per minute or a run longer than 60 minutes is outside the planned test")
    if a.self_test:
        a.url, key, a.window = "http://self-test.invalid", "self-test", None
    else:
        key = os.environ.get("NEUROSCAN_API_KEY", "")
        if not key:
            sys.exit("NEUROSCAN_API_KEY is not set (set it in this terminal; never paste it into chat or a file).")
        host = urlparse(a.url).hostname or ""
        if not host:
            sys.exit("--url is required.")
        if host not in E.LOCAL_HOSTS:
            if not a.live or not a.window:
                sys.exit(f"{host} is not local: --live and --window START END are required.")
            w = [datetime.fromisoformat(x) for x in a.window]
            if any(x.tzinfo is None for x in w):
                sys.exit("--window times need a time zone, e.g. 2026-11-26T02:00+06:00")
            if not (w[0] <= datetime.now(timezone.utc) <= w[1]):
                sys.exit("now is outside the approved window.")

    # the schedule is fixed before the run: same seed -> same arrival times and cases on both services
    rng = random.Random(a.seed)
    gaps, t = [], 0.0
    while True:
        g = rng.expovariate(a.rate / 60.0)
        if t + g > a.minutes * 60:
            break
        t += g
        gaps.append(g)
    planned = a.warm_n + len(gaps)
    if planned > a.max_requests:
        sys.exit(f"planned {planned} requests exceed the cap {a.max_requests}")
    scale = 600.0 if a.self_test else 1.0   # self-test: the same schedule, 600 times faster
    cases = E.load_pack(a.pack)
    out = Path(a.out or f"e1c_results_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%MZ')}{'_selftest' if a.self_test else ''}")
    out.mkdir(parents=True, exist_ok=True)
    info = {"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "url_host": urlparse(a.url).hostname,
            "self_test": a.self_test, "warm_n": a.warm_n, "rate_per_min": a.rate, "minutes": a.minutes,
            "scheduled_requests": len(gaps), "planned_requests": planned, "health_every_min": a.health_every,
            "seed": a.seed, "max_requests": a.max_requests, "timeout_s": E.TIMEOUT_S, "max_in_flight": MAX_IN_FLIGHT,
            "window": a.window, "test_pack_sha256": hashlib.sha256(Path(a.pack).read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "e1_script_sha256": hashlib.sha256(Path(E.__file__).read_bytes()).hexdigest(),
            "python": platform.python_version(), "requests": requests.__version__, "client_os": platform.platform()}
    print(f"E1c: {a.rate}/min for {a.minutes} min = {len(gaps)} scheduled requests (+{a.warm_n} warm-up; cap "
          f"{a.max_requests}) -> {out}", flush=True)

    run = E.Run(argparse.Namespace(max_requests=a.max_requests, window=a.window), E.Client(a.url, key, a.self_test), cases)
    seq = run.cycle()
    summary, health = {}, []

    def snapshot(tag):
        s = {}
        run.health_snapshot(s, tag)
        h = s.get("server_health", {}).get(tag, {})
        h.update({"tag": tag, "t_s": round(time.time() - run.t0, 1)})
        health.append(h)
        print(f"  health {tag}: memory {h.get('memory_mb')} uptime {h.get('uptime_s')}", flush=True)

    try:
        print("phase warm", flush=True)
        for _ in range(a.warm_n):
            run.one("warm", 0, *next(seq))
        warm = [r for r in run.rows if r["phase"] == "warm" and r["ok"]]
        srv = [float(r["server_request_ms"]) / 1000 for r in warm[1:] if r.get("server_request_ms") not in ("", None)]
        s_time = statistics.median(srv) if srv else statistics.median(r["latency_s"] for r in warm[1:])
        net = statistics.median(r["latency_s"] - float(r["server_request_ms"]) / 1000 for r in warm[1:]
                                if r.get("server_request_ms") not in ("", None)) if srv else 0.0
        rho = a.rate * s_time / 60
        summary["warm"] = {"answered": len(warm), "service_time_s": round(s_time, 3), "network_overhead_s": round(net, 3),
                           "rho": round(rho, 3),
                           "md1_mean_response_s": round(s_time + net + rho * s_time / (2 * (1 - rho)), 2) if rho < 1 else None}
        print(f"  warm: S = {s_time:.2f} s, network {net:.2f} s, load rho = {rho:.2f}", flush=True)
        snapshot("start")

        print(f"phase sustained {a.rate:g}/min", flush=True)
        t0, sched, stop = time.time(), 0.0, []
        next_health = a.health_every * 60
        in_flight = threading.Semaphore(MAX_IN_FLIGHT)

        def send(when, item):
            try:
                row = run.one("sustained", a.rate, *item)
                row["scheduled_s"] = round(when, 3)
                row["send_delay_s"] = round(row["t_start_s"] - (t0 - run.t0) - when / scale, 3)
            except E.Stop as e:
                stop.append(str(e))
            finally:
                in_flight.release()

        with ThreadPoolExecutor(MAX_IN_FLIGHT) as ex:
            for g in gaps:
                sched += g
                if sched >= next_health:
                    snapshot(f"{next_health / 60:g}min")
                    next_health += a.health_every * 60
                delay = t0 + sched / scale - time.time()
                if delay > 0:
                    time.sleep(delay)
                if stop:
                    break
                in_flight.acquire()
                ex.submit(send, sched, next(seq))
        snapshot("end")
        if stop:
            raise E.Stop(stop[0])
    except E.Stop as e:
        summary["stopped"] = str(e)
        print(f"STOPPED: {e}", flush=True)
    except KeyboardInterrupt:
        summary["stopped"] = "interrupted by the operator"
        print("interrupted; writing what was measured", flush=True)
    finally:
        rows = [r for r in run.rows if r["phase"] == "sustained"]
        summary["all"] = E.stats(rows)
        if rows:
            t_first = min(r["t_start_s"] for r in rows)
            span = max(r["t_start_s"] for r in rows) - t_first
            summary["all"]["achieved_arrivals_per_min"] = round(60 * (len(rows) - 1) / span * (1 / scale), 2) if span else None
            binw = a.bin_minutes * 60 / scale
            bins = {}
            for r in rows:
                bins.setdefault(int((r["t_start_s"] - t_first) // binw), []).append(r)
            summary["by_bin"] = {f"{k * a.bin_minutes:g}-{(k + 1) * a.bin_minutes:g} min": E.stats(v)
                                 for k, v in sorted(bins.items())}
            half = sorted(rows, key=lambda r: r["t_start_s"])
            summary["first_half"], summary["second_half"] = E.stats(half[:len(half) // 2]), E.stats(half[len(half) // 2:])
            fid = [r for r in rows if r["ok"] and "matches_offline" in r]
            summary["fidelity"] = {"requests_with_saved_answer": len(fid),
                                   "same_prediction_as_offline": sum(r["matches_offline"] for r in fid),
                                   "max_abs_prob_diff_vs_offline": max((r["max_abs_prob_diff_vs_offline"] for r in fid),
                                                                       default=None),
                                   "heatmaps_returned": sum(int(r["heatmap"] or 0) for r in rows if r["ok"]),
                                   "answered": sum(r["ok"] for r in rows)}
            up = [h.get("uptime_s") for h in health if h.get("uptime_s") is not None]
            summary["restarts_seen"] = sum(1 for x, y in zip(up, up[1:]) if y < x)
        summary["health"] = health
        info["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        info["requests_sent"] = run.sent
        cols = []
        for r in run.rows:
            cols += [k for k in r if k not in cols]
        with open(out / "requests.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(run.rows)
        json.dump(summary, open(out / "summary.json", "w"), indent=2)
        json.dump(info, open(out / "run_info.json", "w"), indent=2)
        print(f"summary: {json.dumps({k: summary.get(k) for k in ('all', 'fidelity', 'restarts_seen', 'stopped')})}", flush=True)
        print(f"written: {out / 'requests.csv'} ({len(run.rows)} rows), summary.json, run_info.json", flush=True)


if __name__ == "__main__":
    main()
