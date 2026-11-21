"""
NeuroScan paper 2 -- experiment E1b (JTCSST revision, R1-10 and R2-5): open-loop load test of one inference service.

Requests arrive as a Poisson process at a fixed rate, each sent at its scheduled time whether or not earlier requests
have been answered (open loop), unlike E1's closed-loop users, who wait for each answer. Run by the author on their own
PC, against the URL and inside the time window the author approved; for the two matched services, run one copy of this
script per service at the same time. The API key is read from NEUROSCAN_API_KEY and is never printed, logged or
written to any output file. Same test pack, request kinds, client, timeout and logging as E1 (e1_load_test.py, which
this script imports).

Phases:
  1. warm   --warm-n sequential requests (default 6): wakes the service and measures its own time per request S
            (median of the server's timing_ms.request) and the network overhead (round trip minus server time).
  2. rates  for each rate in --rates (requests per minute), --n requests (--n-first for the first rate) with
            exponential gaps drawn with a fixed seed; then the script waits for every answer and pauses --gap-s seconds.
For each rate the summary gives the response-time distribution and, for comparison, the single-server queue with a
fixed service time S (M/D/1): mean wait Wq = rho*S / (2*(1 - rho)), rho = rate*S/60, plus S and the network overhead.
A short run starts with an empty queue, so near saturation its waits fall below the steady-state value.
Safety limits (fixed): total requests <= --max-requests (default 250); 5 consecutive failures end the run; at most 64
requests in flight; requests time out after 120 s; a non-local URL needs --live and --window START END, and the script
stops sending outside the window.

Outputs (in --out): requests.csv (one row per request: scheduled and actual send time, latency, status, server timing;
no image data, no key), summary.json, run_info.json.

Usage:
  python -I e1b_open_loop.py --pack <path>/neuroscan_test_pack.zip --url https://<service>.onrender.com ^
         --live --window 2026-11-23T02:00+06:00 2026-11-23T03:30+06:00 --rates 8 12 14 --out e1b_fp32
  python -I e1b_open_loop.py --pack <zip> --self-test        (no network: checks the script with simulated answers)
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


def md1(rate_per_min, s):
    rho = rate_per_min * s / 60.0
    return {"rho": round(rho, 3), "wq_s": round(rho * s / (2 * (1 - rho)), 2) if rho < 1 else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--url", default=os.environ.get("NEUROSCAN_URL", ""))
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--window", nargs=2, metavar=("START", "END"))
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--warm-n", type=int, default=6)
    ap.add_argument("--rates", type=float, nargs="+", default=[8, 12, 14])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--n-first", type=int, default=40)
    ap.add_argument("--gap-s", type=float, default=30)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-requests", type=int, default=250)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

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
                sys.exit("--window times need a time zone, e.g. 2026-11-23T02:00+06:00")
            if not (w[0] <= datetime.now(timezone.utc) <= w[1]):
                sys.exit("now is outside the approved window.")
    if max(a.rates) > 30:
        sys.exit("rates above 30 per minute are outside the planned test")

    cases = E.load_pack(a.pack)
    counts = [a.n_first] + [a.n] * (len(a.rates) - 1)
    planned = a.warm_n + sum(counts)
    if planned > a.max_requests:
        sys.exit(f"planned {planned} requests exceed the cap {a.max_requests}")
    out = Path(a.out or f"e1b_results_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%MZ')}{'_selftest' if a.self_test else ''}")
    out.mkdir(parents=True, exist_ok=True)
    info = {"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "url_host": urlparse(a.url).hostname,
            "self_test": a.self_test, "warm_n": a.warm_n, "rates_per_min": a.rates, "requests_per_rate": counts,
            "gap_s": a.gap_s, "seed": a.seed, "max_requests": a.max_requests, "planned_requests": planned,
            "timeout_s": E.TIMEOUT_S, "max_in_flight": MAX_IN_FLIGHT, "window": a.window,
            "test_pack_sha256": hashlib.sha256(Path(a.pack).read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "e1_script_sha256": hashlib.sha256(Path(E.__file__).read_bytes()).hexdigest(),
            "python": platform.python_version(), "requests": requests.__version__, "client_os": platform.platform()}
    print(f"E1b: rates {a.rates}/min, {counts} requests, planned {planned} (cap {a.max_requests}) -> {out}", flush=True)

    args = argparse.Namespace(max_requests=a.max_requests, window=a.window)
    run = E.Run(args, E.Client(a.url, key, a.self_test), cases)
    seq = run.cycle()
    summary = {}
    try:
        print("phase warm", flush=True)
        for _ in range(a.warm_n):
            run.one("warm", 0, *next(seq))
        warm = [r for r in run.rows if r["phase"] == "warm" and r["ok"]]
        srv = [float(r["server_request_ms"]) / 1000 for r in warm[1:] if r.get("server_request_ms") not in ("", None)]
        s = statistics.median(srv) if srv else statistics.median(r["latency_s"] for r in warm[1:])
        net = statistics.median(r["latency_s"] - float(r["server_request_ms"]) / 1000 for r in warm[1:]
                                if r.get("server_request_ms") not in ("", None)) if srv else 0.0
        summary["warm"] = {"answered": len(warm), "service_time_s": round(s, 3), "network_overhead_s": round(net, 3)}
        print(f"  warm: S = {s:.2f} s, network {net:.2f} s", flush=True)
        if a.self_test:   # the simulated client answers in ~0.02 s; scale the schedule so the test stays short
            a.rates = [r * 100 for r in a.rates]
        rng = random.Random(a.seed)
        summary["rates"] = {}
        for rate, n in zip(a.rates, counts):
            gaps = [rng.expovariate(rate / 60.0) for _ in range(n)]
            items = [next(seq) for _ in range(n)]
            print(f"phase rate {rate:g}/min: {n} requests over about {sum(gaps):.0f} s", flush=True)
            t0, sched, stop = time.time(), 0.0, []
            in_flight = threading.Semaphore(MAX_IN_FLIGHT)

            def send(i, when, item):
                try:
                    row = run.one("open", rate, *item)
                    row["scheduled_s"] = round(when, 3)
                    row["send_delay_s"] = round(row["t_start_s"] - (t0 - run.t0) - when, 3)
                except E.Stop as e:
                    stop.append(str(e))
                finally:
                    in_flight.release()

            with ThreadPoolExecutor(MAX_IN_FLIGHT) as ex:
                for i, (g, item) in enumerate(zip(gaps, items)):
                    sched += g
                    delay = t0 + sched - time.time()
                    if delay > 0:
                        time.sleep(delay)
                    if stop:
                        break
                    in_flight.acquire()
                    ex.submit(send, i, sched, item)
            rows = [r for r in run.rows if r["phase"] == "open" and r["concurrency"] == rate]
            st = E.stats(rows)
            okr = [r for r in rows if r["ok"]]
            span = max((r["t_start_s"] for r in rows), default=0) - min((r["t_start_s"] for r in rows), default=0)
            q = md1(rate / (100 if a.self_test else 1), s)
            st.update({"offered_per_min": rate, "achieved_arrivals_per_min": round(60 * (len(rows) - 1) / span, 2) if span else None,
                       "server_model_s_median": round(statistics.median(float(r["server_model_ms"]) / 1000 for r in okr
                                                                        if r.get("server_model_ms") not in ("", None)), 3)
                       if any(r.get("server_model_ms") not in ("", None) for r in okr) else None,
                       "md1": q, "md1_mean_response_s": round(s + net + q["wq_s"], 2) if q["wq_s"] is not None else None})
            summary["rates"][f"{rate:g}"] = st
            print(f"  rate {rate:g}: {st['all']} (M/D/1 mean response {st['md1_mean_response_s']} s)", flush=True)
            if stop:
                raise E.Stop(stop[0])
            if rate != a.rates[-1]:
                time.sleep(a.gap_s / (100 if a.self_test else 1))
        run.health_snapshot(summary, "after_rates")
    except E.Stop as e:
        summary["stopped"] = str(e)
        print(f"STOPPED: {e}", flush=True)
    except KeyboardInterrupt:
        summary["stopped"] = "interrupted by the operator"
        print("interrupted; writing what was measured", flush=True)
    finally:
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
        print(f"written: {out / 'requests.csv'} ({len(run.rows)} rows), summary.json, run_info.json", flush=True)


if __name__ == "__main__":
    main()
