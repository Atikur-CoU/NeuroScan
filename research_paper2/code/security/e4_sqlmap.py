"""
E4: sqlmap against every parameter that reaches a database query, on the LOCAL copy only.

Logs in as the made-up test accounts to get session cookies and API tokens, then runs sqlmap (level 3, risk 1: no
statements that could modify data) per target and records whether sqlmap found any parameter injectable.

Usage: python -I e4_sqlmap.py --site http://localhost/neuroscan_e4/public --accounts <json> --sqlmap <sqlmap.py>
                               --work <dir for sqlmap output> --out <results dir>
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests

sys.path.insert(0, str(Path(__file__).parent))
from e4_suite import Site, upload, uuid_from  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--accounts", required=True)
    ap.add_argument("--sqlmap", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    # name=reason; e.g. after the fix the API login answers 429 before its query once the address limit is reached,
    # and sqlmap keeps retrying 429, so that target no longer reaches the database
    ap.add_argument("--skip", action="append", default=[])
    ap.add_argument("--only", nargs="*", default=[])
    a = ap.parse_args()
    skip = dict(s.split("=", 1) for s in a.skip)
    if urlparse(a.site).hostname not in ("localhost", "127.0.0.1"):
        sys.exit("refusing non-local target")
    site = Site(a.site, json.loads(Path(a.accounts).read_text()))
    imgs = Path(a.images)

    pat = site.logged_in("patient_a")
    scan = uuid_from(upload(site, pat, {"file": ("m.png", (imgs / "2_main_slice.png").read_bytes())}))
    cookie = lambda s: f"PHPSESSID={s.cookies.get('PHPSESSID')}"
    adm, doc = site.logged_in("admin"), site.logged_in("doctor_a")
    tok = site.api_token("doctor_a")
    u = site.url
    common = ["--batch", "--level=3", "--risk=1", "--flush-session", "--disable-coloring"]
    targets = {
        "login_form": [f"--url={u('login.php')}", "--data=email=a%40e4.test&password=x&csrf_token=0",
                       "-p", "email,password", "--csrf-token=csrf_token", f"--csrf-url={u('login.php')}"],
        # an existing email keeps the form on screen ("already exists"); a new one would register and redirect away
        "register_form": [f"--url={u('register.php')}",
                          "--data=full_name=x&email=admin%40e4.test&password=E4-test-pass-123&password_confirm=E4-test-pass-123&consent=1&csrf_token=0",
                          "-p", "full_name,email", "--csrf-token=csrf_token"],
        # a wrong login answers 401 by design, so 401 must not stop sqlmap
        "api_login_json": [f"--url={u('api/v1/login.php')}", '--data={"email":"a@e4.test","password":"x"}',
                           "--headers=Content-Type: application/json", "--ignore-code=401"],
        "audit_log_filters": [f"--url={u('admin_audit_log.php')}?actor_id=1&action=login_failed&date_from=2026-01-01&date_to=2026-12-31&page=1",
                              f"--cookie={cookie(adm)}", "-p", "actor_id,action,date_from,date_to,page"],
        # flagged by ZAP's administrator scan; patient 5 and doctor_a (id 2) are valid, so the form stays on its page
        "admin_assign_form": [f"--url={u('admin_assign.php')}",
                              "--data=csrf_token=0&action=assign&patient_id=5&provider_id=2&relationship_type=primary_doctor",
                              f"--cookie={cookie(adm)}", "-p", "patient_id,provider_id", "--csrf-token=csrf_token"],
        "admin_user_edit": [f"--url={u('admin_user_edit.php')}?id=2", f"--cookie={cookie(adm)}", "-p", "id"],
        "scan_view_uuid": [f"--url={u('scan_view2.php')}?uuid={scan}", f"--cookie={cookie(pat)}", "-p", "uuid"],
        "scan_image_uuid": [f"--url={u('scan_image.php')}?uuid={scan}&type=original", f"--cookie={cookie(pat)}", "-p", "uuid"],
        "provider_patient": [f"--url={u('provider_patient.php')}?patient_id=5", f"--cookie={cookie(doc)}", "-p", "patient_id"],
        "review_form_uuid": [f"--url={u('review_form.php')}?scan_uuid={scan}", f"--cookie={cookie(doc)}", "-p", "scan_uuid"],
        "api_scans": [f"--url={u('api/v1/scans.php')}?patient_id=5", f"--headers=Authorization: Bearer {tok}", "-p", "patient_id"],
        "api_scan_detail": [f"--url={u('api/v1/scan_detail.php')}?uuid={scan}", f"--headers=Authorization: Bearer {tok}", "-p", "uuid"],
    }
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, args in targets.items():
        if a.only and name not in a.only:
            continue
        if name in skip:
            results[name] = {"skipped": skip[name]}
            print(f"sqlmap: {name} skipped ({skip[name]})", flush=True)
            continue
        print(f"sqlmap: {name}", flush=True)
        p = subprocess.run([sys.executable, a.sqlmap, *common, f"--output-dir={work / name}", *args],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600)
        log = p.stdout
        (work / f"{name}.log").write_text(log, encoding="utf-8")
        vuln = bool(re.search(r"is vulnerable|sqlmap identified the following injection", log))
        clean = "do not appear to be injectable" in log
        tested = sorted(set(re.findall(r"testing if (?:GET|POST|\(custom\) POST|JSON) parameter '([^']+)' is dynamic", log)) |
                        set(re.findall(r"(?:GET|POST|JSON) parameter '([^']+)'", log)))
        results[name] = {"injectable": vuln, "sqlmap_says_not_injectable": clean, "parameters_seen": tested,
                         "requests": len(re.findall(r"\n", log))}
        print(f"  injectable={vuln} clean={clean} params={tested}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ver = subprocess.run([sys.executable, a.sqlmap, "--version"], capture_output=True, text=True, input="\n").stdout.strip().splitlines()[0]
    (out / "e4_sqlmap.json").write_text(json.dumps({"sqlmap_version": ver, "options": common, "targets": results}, indent=1))
    print(f"{sum(r.get('injectable', False) for r in results.values())} of {len(results)} targets injectable")


if __name__ == "__main__":
    main()
