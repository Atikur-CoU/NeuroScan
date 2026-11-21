"""E1b analysis: measured open-loop response times against queueing predictions, per service and rate.
Uses the achieved arrival rate of each phase and the service time measured during it (server timing_ms.request).
Also replays each phase through one first-come-first-served server with the measured send and service times.
M/D/1: Wq = rho*S/(2(1-rho)); M/G/1 (Pollaczek-Khinchine): Wq = rho*S*(1+cv^2)/(2(1-rho)).
    python -I e1b_analyse.py e1b_fp32 e1b_int8   -> prints a table and writes e1b_analysis.json"""
import csv
import json
import statistics as st
import sys
from pathlib import Path

out = {}
for d in sys.argv[1:]:
    rows = list(csv.DictReader(open(Path(d) / "requests.csv", encoding="utf-8")))
    summ = json.load(open(Path(d) / "summary.json"))
    net = summ["warm"]["network_overhead_s"]
    res = {}
    for rate in sorted({float(r["concurrency"]) for r in rows if r["phase"] == "open"}):
        rr = sorted([r for r in rows if r["phase"] == "open" and float(r["concurrency"]) == rate],
                    key=lambda r: float(r["t_start_s"]))
        ok = [r for r in rr if r["ok"] == "1" and r["server_request_ms"]]
        t = [float(r["t_start_s"]) for r in rr]
        lam = (len(t) - 1) / (t[-1] - t[0])                     # arrivals per second
        s = [float(r["server_request_ms"]) / 1000 for r in ok]
        S, cv2 = st.mean(s), st.pvariance(s) / st.mean(s) ** 2
        rho = lam * S
        lat = [float(r["latency_s"]) for r in ok]
        wait = [float(r["latency_s"]) - float(r["server_request_ms"]) / 1000 - net for r in ok]
        free, fifo = 0.0, []      # replay: one first-come-first-served server, measured arrivals and service times
        for r in ok:
            a = float(r["t_start_s"]) + net / 2
            free = max(a, free) + float(r["server_request_ms"]) / 1000
            fifo.append(free + net / 2 - float(r["t_start_s"]))
        q = lambda c: S + net + rho * S * (1 + c) / (2 * (1 - rho)) if rho < 1 else None
        res[f"{rate:g}"] = {
            "offered_per_min": rate, "achieved_per_min": round(lam * 60, 2), "sent": len(rr), "answered": len(ok),
            "failures": {k: sum(1 for r in rr if r["ok"] != "1" and r["error"] == k) for k in
                         {r["error"] for r in rr if r["ok"] != "1"}},
            "service_mean_s": round(S, 3), "service_cv2": round(cv2, 3), "rho_achieved": round(rho, 3),
            "measured_mean_s": round(st.mean(lat), 2), "measured_median_s": round(st.median(lat), 2),
            "measured_p95_s": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 2), "max_s": round(max(lat), 2),
            "measured_mean_wait_s": round(st.mean(wait), 2),
            "md1_mean_s": round(q(0), 2) if q(0) else None, "mg1_mean_s": round(q(cv2), 2) if q(cv2) else None,
            "fifo_replay_mean_s": round(st.mean(fifo), 2),
            "send_delay_max_s": max(float(r["send_delay_s"] or 0) for r in rr),
            "matches_offline": f"{sum(r['matches_offline'] == '1' for r in ok)}/{len(ok)}",
            "max_prob_diff_vs_offline": max(float(r["max_abs_prob_diff_vs_offline"] or 0) for r in ok)}
    out[d] = {"warm": summ["warm"], "rates": res, "health": summ.get("server_health")}
json.dump(out, open("e1b_analysis.json", "w"), indent=2)
for d, v in out.items():
    print(d, v["warm"])
    for k, r in v["rates"].items():
        print(" ", k, {x: r[x] for x in r if x != "offered_per_min"})
