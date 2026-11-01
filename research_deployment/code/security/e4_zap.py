"""
E4: OWASP ZAP spider, passive and active scan of the LOCAL copy, once without login and once per logged-in role.

For a logged-in role, a fresh session cookie of the made-up test account is added to every request (ZAP replacer
rule); logout.php is excluded so the scan keeps its session. Each scan's findings go to a JSON report; this script
summarises them by risk.

Usage: python -I e4_zap.py --site http://localhost/neuroscan_e4/public --accounts <json> --zap <ZAP dir>
                           --java-home <JRE dir> --work <dir> --out <results dir> [--minutes 15]
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent))
from e4_suite import Site  # noqa: E402

PLAN = """env:
  contexts:
    - name: neuroscan
      urls: ["{base}/"]
      includePaths: ["{base_re}.*"]
      excludePaths: ["{base_re}logout\\\\.php.*"]
  parameters:
    failOnError: false
    progressToStdout: true
jobs:
{replacer}  - type: spider
    parameters: {{context: neuroscan, url: "{start}", maxDuration: 5}}
  - type: passiveScan-wait
    parameters: {{maxDuration: 5}}
  - type: activeScan
    parameters: {{context: neuroscan, maxScanDurationInMins: {minutes}, maxRuleDurationInMins: 3}}
  - type: passiveScan-wait
    parameters: {{maxDuration: 5}}
  - type: report
    parameters: {{template: traditional-json, reportDir: "{rdir}", reportFile: "{rfile}"}}
"""
REPLACER = """  - type: replacer
    rules:
      - description: session
        matchType: req_header
        matchString: Cookie
        replacementString: "PHPSESSID={sid}"
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--accounts", required=True)
    ap.add_argument("--zap", required=True)
    ap.add_argument("--java-home", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=int, default=15)
    ap.add_argument("--roles", nargs="+", default=["anonymous", "patient_a", "doctor_a", "admin"])
    a = ap.parse_args()
    if urlparse(a.site).hostname not in ("localhost", "127.0.0.1"):
        sys.exit("refusing non-local target")
    site = Site(a.site, json.loads(Path(a.accounts).read_text()))
    work, out = Path(a.work), Path(a.out)
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    base = a.site.rstrip("/")
    env = dict(os.environ, JAVA_HOME=a.java_home, PATH=str(Path(a.java_home) / "bin") + os.pathsep + os.environ["PATH"])
    summary = {}
    for role in a.roles:
        if role == "anonymous":
            replacer, start = "", f"{base}/login.php"
        else:
            sid = site.logged_in(role).cookies.get("PHPSESSID")
            replacer, start = REPLACER.format(sid=sid), f"{base}/index.php"
        plan = work / f"plan_{role}.yaml"
        plan.write_text(PLAN.format(base=base, base_re=base.replace(".", "\\\\.") + "/", start=start, replacer=replacer,
                                    minutes=a.minutes, rdir=str(work).replace("\\", "/"), rfile=f"zap_{role}"))
        home = work / f"home_{role}"
        print(f"ZAP: {role}", flush=True)
        p = subprocess.run([str(Path(a.zap) / "zap.bat"), "-cmd", "-dir", str(home), "-autorun", str(plan)],
                           cwd=a.zap, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=3 * 3600, shell=False)
        (work / f"zap_{role}.log").write_text(p.stdout + "\n" + p.stderr, encoding="utf-8")
        rep = work / f"zap_{role}.json"
        if not rep.exists():
            print(f"  no report (exit {p.returncode}); see {work / f'zap_{role}.log'}", flush=True)
            summary[role] = {"error": f"no report, exit {p.returncode}"}
            continue
        alerts = [al for s in json.loads(rep.read_text(encoding="utf-8"))["site"] for al in s.get("alerts", [])]
        by = {}
        for al in alerts:
            by.setdefault(al["riskdesc"].split(" ")[0], []).append(
                {"alert": al["alert"], "pluginid": al["pluginid"], "count": int(al.get("count", len(al.get("instances", [])))),
                 "example": (al.get("instances") or [{}])[0].get("uri", "")})
        summary[role] = by
        print("  " + ", ".join(f"{k}: {len(v)}" for k, v in sorted(by.items())), flush=True)
    (out / "e4_zap.json").write_text(json.dumps({"zap": Path(a.zap).name, "minutes_per_active_scan": a.minutes,
                                                 "by_role": summary}, indent=1))


if __name__ == "__main__":
    main()
