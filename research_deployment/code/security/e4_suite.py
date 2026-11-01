"""
NeuroScan paper 2 -- experiment E4: security tests of the web application and the inference API.

Runs ONLY against a local copy (XAMPP on this PC, test database neuroscan_e4test, made-up accounts) and a local
inference server on 127.0.0.1; it refuses any other host. Test credentials and the local inference key are read from
files outside the repository and are never printed or written to the outputs.

Each test states the expected secure behaviour and records what happened. Categories:
  AUTH  authentication, lockout, sessions, forced password change
  AC    access control: unauthenticated access, role checks, other patients' data (IDOR), internal files
  CSRF  state-changing forms without a valid token
  XSS   stored script payloads in every free-text field, checked where they are displayed
  UP    upload validation: wrong type, polyglot, decompression bomb, size, file name
  HDR   security headers and cookie flags
  INF   inference API: key check, malformed input

Usage:
  python -I e4_suite.py --site http://localhost/neuroscan_e4/public --inference http://127.0.0.1:8765 \
         --accounts <test_accounts.json> --inference-key-file <key file> --images <dir with case23 PNGs> --out <dir>
"""
import argparse
import io
import json
import re
import struct
import sys
import time
import uuid
import zlib
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

LOCAL = {"localhost", "127.0.0.1"}
XSS = '<script>alert("e4")</script><img src=x onerror=alert(1)>'
RESULTS = []


def record(tid, cat, what, expected, observed, ok):
    RESULTS.append({"id": tid, "category": cat, "test": what, "expected": expected, "observed": observed, "pass": bool(ok)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {tid} {what}: {observed}", flush=True)


# ------------------------------------------------------------------------------------------------ helpers
class Site:
    def __init__(self, base, accounts):
        self.base, self.acc = base.rstrip("/"), accounts

    def url(self, path):
        return f"{self.base}/{path.lstrip('/')}"

    def session(self):
        return requests.Session()

    @staticmethod
    def token(html):
        m = re.search(r'name="csrf_token" value="([0-9a-f]+)"', html)
        return m.group(1) if m else ""

    def get_token(self, s, page):
        return self.token(s.get(self.url(page)).text)

    def login(self, name, password=None, s=None):
        s = s or self.session()
        a = self.acc[name]
        tok = self.get_token(s, "login.php")
        r = s.post(self.url("login.php"), data={"csrf_token": tok, "email": a["email"],
                                               "password": a["password"] if password is None else password},
                   allow_redirects=False)
        return s, r

    def logged_in(self, name):
        s, r = self.login(name)
        if r.status_code != 302:
            sys.exit(f"setup: could not log in as {name} (HTTP {r.status_code})")
        return s

    def api_token(self, name):
        a = self.acc[name]
        r = requests.post(self.url("api/v1/login.php"), json={"email": a["email"], "password": a["password"]})
        return r.json().get("token") if r.status_code == 200 else None


def png_bytes(w, h, extra=b""):
    """A minimal valid greyscale PNG of w x h (all zero), optionally with bytes appended after IEND."""
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + b"\x00" * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"") + extra)


def upload(site, s, files, modality="CT", notes="", token=None):
    tok = site.get_token(s, "scan_upload.php") if token is None else token
    up = {k: (fn, data, "image/png") for k, (fn, data) in files.items()}
    return s.post(site.url("scan_upload.php"), data={"csrf_token": tok, "modality": modality, "notes": notes},
                  files=up, allow_redirects=False)


def uuid_from(r):
    m = re.search(r"uuid=([0-9a-f-]{36})", r.headers.get("Location", ""))
    return m.group(1) if m else None


def denied(r):
    """Access refused: 401/403/404, or a redirect to the login or password-change page."""
    return r.status_code in (401, 403, 404) or (r.status_code == 302 and
                                                 re.search(r"(login|change_password)\.php", r.headers.get("Location", "")))


# ------------------------------------------------------------------------------------------------- tests
def t_auth(site, db_reset_lock, db_reset_ip):
    print("AUTH", flush=True)
    # AUTH-1/2: lockout after 5 wrong passwords, web and API
    for tid, via in (("AUTH-1", "web"), ("AUTH-2", "api")):
        db_reset_lock("lock_victim")
        a = site.acc["lock_victim"]
        for _ in range(5):
            if via == "web":
                site.login("lock_victim", password="wrong-password")
            else:
                requests.post(site.url("api/v1/login.php"), json={"email": a["email"], "password": "wrong-password"})
        if via == "web":
            _, r = site.login("lock_victim")
            ok = r.status_code == 200 and "Too many failed attempts" in r.text
        else:
            r = requests.post(site.url("api/v1/login.php"), json={"email": a["email"], "password": a["password"]})
            ok = r.status_code == 429
        record(tid, "AUTH", f"account locked after 5 wrong passwords ({via})", "correct password refused while locked",
               f"HTTP {r.status_code}" + (", locked message" if ok else ""), ok)
    db_reset_lock("lock_victim")

    # AUTH-3: same answer for unknown account and wrong password (no account enumeration by message)
    s = site.session()
    tok = site.get_token(s, "login.php")
    r1 = s.post(site.url("login.php"), data={"csrf_token": tok, "email": "nobody@e4.test", "password": "x" * 10})
    tok = site.get_token(s, "login.php")
    r2 = s.post(site.url("login.php"), data={"csrf_token": tok, "email": site.acc["patient_b"]["email"], "password": "x" * 10})
    m = lambda t: re.search(r'class="flash err">([^<]*)<', t)
    e1, e2 = (m(r1.text) or [None, None])[1], (m(r2.text) or [None, None])[1]
    record("AUTH-3", "AUTH", "same error for unknown email and wrong password", "identical message",
           f"'{e1}' vs '{e2}'", e1 == e2 and e1 is not None)
    db_reset_lock("patient_b")

    # AUTH-4: throttling of many wrong passwords spread over many accounts from one address (password spraying)
    db_reset_ip()
    t = time.time()
    blocked = 0
    for i in range(30):
        s = site.session()
        tok = site.get_token(s, "login.php")
        r = s.post(site.url("login.php"), data={"csrf_token": tok, "email": f"spray{i}@e4.test", "password": "Password123"})
        blocked += "Too many" in r.text or r.status_code == 429
    record("AUTH-4", "AUTH", "30 wrong logins across 30 accounts from one address are throttled",
           "address-level throttling after a few attempts", f"{blocked} of 30 refused ({time.time() - t:.1f} s)", blocked > 0)
    db_reset_ip()

    # AUTH-5: session id changes at login (no fixation)
    s = site.session()
    s.get(site.url("login.php"))
    before = s.cookies.get("PHPSESSID")
    site.login("patient_a", s=s)
    after = s.cookies.get("PHPSESSID")
    record("AUTH-5", "AUTH", "new session id after login", "different id", "changed" if before != after else "unchanged",
           before and after and before != after)

    # AUTH-6: logout ends the session; the old cookie no longer works
    s = site.logged_in("patient_a")
    old = s.cookies.get("PHPSESSID")
    tok = site.get_token(s, "dashboard_patient.php")
    s.post(site.url("logout.php"), data={"csrf_token": tok})
    s2 = site.session()
    s2.cookies.set("PHPSESSID", old)
    r = s2.get(site.url("dashboard_patient.php"), allow_redirects=False)
    record("AUTH-6", "AUTH", "old session cookie refused after logout", "redirect to login", f"HTTP {r.status_code}", denied(r))

    # AUTH-7: logout is not triggerable cross-site with a plain GET (CSRF logout)
    s = site.logged_in("patient_a")
    r = s.get(site.url("logout.php"), allow_redirects=False)
    r2 = s.get(site.url("dashboard_patient.php"), allow_redirects=False)
    record("AUTH-7", "AUTH", "logout needs a POST with a token (a link on another site cannot log a user out)",
           "GET logout.php does nothing", "GET logged the user out" if denied(r2) else "GET ignored", not denied(r2))

    # AUTH-8/9: forced password change for admin-created accounts, web and API
    s = site.logged_in("new_doctor")
    r = s.get(site.url("dashboard_provider.php"), allow_redirects=False)
    record("AUTH-8", "AUTH", "account with a temporary password must change it first (web)",
           "every page redirects to change_password.php", f"HTTP {r.status_code} -> {r.headers.get('Location')}",
           r.status_code == 302 and "change_password.php" in r.headers.get("Location", ""))
    tok = site.api_token("new_doctor")
    r = requests.get(site.url("api/v1/my_patients.php"), headers={"Authorization": f"Bearer {tok}"})
    record("AUTH-9", "AUTH", "account with a temporary password must change it first (API)",
           "API refuses until the password is changed", f"HTTP {r.status_code}", r.status_code in (401, 403))

    # AUTH-10: registration rate limit
    db_reset_ip()
    blocked = 0
    s = site.session()
    for i in range(10):
        tok = site.get_token(s, "register.php")
        r = s.post(site.url("register.php"), data={"csrf_token": tok, "full_name": f"E4 reg {i}",
                   "email": f"reg{i}-{uuid.uuid4().hex[:6]}@e4.test", "password": "E4-test-pass-123",
                   "password_confirm": "E4-test-pass-123", "consent": "1"}, allow_redirects=False)
        blocked += r.status_code == 429 or "Too many" in r.text
        s = site.session()
    record("AUTH-10", "AUTH", "10 registrations in a row from one address are throttled", "throttling",
           f"{blocked} of 10 refused", blocked > 0)
    db_reset_ip()

    # AUTH-11: password length policy at registration
    s = site.session()
    tok = site.get_token(s, "register.php")
    em = f"short-{uuid.uuid4().hex[:6]}@e4.test"
    s.post(site.url("register.php"), data={"csrf_token": tok, "full_name": "E4 short", "email": em, "password": "abc",
           "password_confirm": "abc", "consent": "1"}, allow_redirects=False)
    r = requests.post(site.url("api/v1/login.php"), json={"email": em, "password": "abc"})
    record("AUTH-11", "AUTH", "3-character password refused at registration", "account not created",
           "account created" if r.status_code == 200 else "refused", r.status_code != 200)


def t_access(site, scan_a, sql_value):
    print("AC", flush=True)
    pages = ["dashboard_patient.php", "dashboard_provider.php", "dashboard_admin.php", "admin_users.php",
             "admin_settings.php", "admin_assign.php", "admin_audit_log.php", "admin_user_edit.php?id=2",
             "scan_upload.php", "change_password.php", f"scan_view2.php?uuid={scan_a}",
             f"scan_image.php?uuid={scan_a}", f"review_form.php?scan_uuid={scan_a}", "provider_patient.php?patient_id=5"]
    bad = [p for p in pages if not denied(requests.get(site.url(p), allow_redirects=False))]
    record("AC-1", "AC", f"{len(pages)} protected pages without logging in", "all refused",
           f"{len(pages) - len(bad)} of {len(pages)} refused" + (f"; open: {bad}" if bad else ""), not bad)
    api = ["api/v1/me.php", "api/v1/scans.php", f"api/v1/scan_detail.php?uuid={scan_a}",
           f"api/v1/scan_image.php?uuid={scan_a}", "api/v1/my_patients.php"]
    bad = [p for p in api if requests.get(site.url(p)).status_code != 401]
    bad += [p for p in api if requests.get(site.url(p), headers={"Authorization": "Bearer " + "0" * 64}).status_code != 401]
    record("AC-2", "AC", f"{len(api)} API endpoints without a token or with a made-up token", "all 401",
           f"{2 * len(api) - len(bad)} of {2 * len(api)} refused", not bad)

    # AC-3: role matrix (who may open which page)
    matrix = {"patient_a": ["dashboard_admin.php", "admin_users.php", "admin_settings.php", "admin_audit_log.php",
                            "dashboard_provider.php", "provider_patient.php?patient_id=5"],
              "doctor_a": ["dashboard_admin.php", "admin_users.php", "admin_settings.php", "scan_upload.php"],
              "radiologist": ["dashboard_admin.php", "admin_settings.php", "scan_upload.php"],
              "admin": ["scan_upload.php"]}
    tried, bad = 0, []
    for who, ps in matrix.items():
        s = site.logged_in(who)
        for p in ps:
            tried += 1
            if not denied(s.get(site.url(p), allow_redirects=False)):
                bad.append(f"{who}:{p}")
    record("AC-3", "AC", f"role checks ({tried} page-role pairs a role must not open)", "all refused",
           f"{tried - len(bad)} of {tried} refused" + (f"; open: {bad}" if bad else ""), not bad)

    # AC-4..7: another patient's scan (IDOR), unassigned provider, assigned provider
    for tid, who, should in (("AC-4", "patient_b", False), ("AC-5", "doctor_b", False),
                             ("AC-6", "doctor_a", True), ("AC-7", "radiologist", True)):
        s = site.logged_in(who)
        tok = site.api_token(who)
        h = {"Authorization": f"Bearer {tok}"}
        checks = {"scan page": s.get(site.url(f"scan_view2.php?uuid={scan_a}"), allow_redirects=False),
                  "image": s.get(site.url(f"scan_image.php?uuid={scan_a}"), allow_redirects=False),
                  "heatmap": s.get(site.url(f"scan_image.php?uuid={scan_a}&type=heatmap"), allow_redirects=False),
                  "API detail": requests.get(site.url(f"api/v1/scan_detail.php?uuid={scan_a}"), headers=h),
                  "API image": requests.get(site.url(f"api/v1/scan_image.php?uuid={scan_a}"), headers=h),
                  "API list": requests.get(site.url("api/v1/scans.php?patient_id=5"), headers=h)}
        if who.startswith("doctor") or who == "radiologist":
            checks["patient page"] = s.get(site.url("provider_patient.php?patient_id=5"), allow_redirects=False)
            checks["review form"] = s.get(site.url(f"review_form.php?scan_uuid={scan_a}"), allow_redirects=False)
        got = {}
        for k, r in checks.items():
            if k == "API list":
                visible = r.status_code == 200 and scan_a in r.text
            else:
                visible = r.status_code == 200
            got[k] = visible
        if should:
            ok = all(got.values())
            obs = "all visible" if ok else f"not visible: {[k for k, v in got.items() if not v]}"
        else:
            ok = not any(got.values())
            obs = f"0 of {len(got)} visible" if ok else f"visible: {[k for k, v in got.items() if v]}"
        desc = {"AC-4": "another patient opens patient A's scan", "AC-5": "unassigned doctor opens patient A's scan",
                "AC-6": "assigned doctor opens patient A's scan (control)",
                "AC-7": "assigned radiologist opens patient A's scan (control)"}[tid]
        record(tid, "AC", desc, "visible" if should else "refused everywhere", obs, ok)

    # AC-8: internal files are not served
    paths = ["includes/config.php", "includes/db.php", "includes/.htaccess", "storage/.htaccess", "sql/schema.sql",
             "storage/logs/", "storage/uploads/"]
    bad = []
    for p in paths:
        r = requests.get(site.url(p), allow_redirects=False)
        if r.status_code == 200 and (len(r.content) > 0):
            bad.append(f"{p} ({len(r.content)} B)")
    record("AC-8", "AC", "internal folders (includes, storage, sql) not served over HTTP", "403/404 for every path",
           "all refused" if not bad else f"served: {bad}", not bad)

    # AC-9: the one-time admin-creation page is not part of the deployed folder
    r = requests.get(site.url("_bootstrap_admin.php"), allow_redirects=False)
    record("AC-9", "AC", "one-time admin-creation page absent from the web folder", "404",
           f"HTTP {r.status_code}" + (" (page present, guarded by a token in its source)" if r.status_code != 404 else ""),
           r.status_code == 404)

    # AC-10: an assignment links only an active patient to an active doctor or radiologist (another patient as
    # "provider", or a made-up account number, must be refused rather than stored or crashing)
    adm = site.logged_in("admin")
    pa, pb = site.acc["patient_a"]["id"], site.acc["patient_b"]["id"]
    before = sql_value(f"SELECT COUNT(*) FROM doctor_patient WHERE provider_id={pb}")
    r = adm.post(site.url("admin_assign.php"), data={"csrf_token": site.get_token(adm, "admin_assign.php"), "action": "assign",
                 "patient_id": pa, "provider_id": pb, "relationship_type": "primary_doctor"})
    stored = sql_value(f"SELECT COUNT(*) FROM doctor_patient WHERE provider_id={pb}") != before
    record("AC-10", "AC", "admin links a patient as another patient's provider", "refused, nothing stored",
           f"HTTP {r.status_code}, link {'stored' if stored else 'not stored'}", not stored and r.status_code == 200)


def t_csrf(site, scan_a):
    print("CSRF", flush=True)
    forms = {
        "patient_a": [("scan_upload.php", {"modality": "CT"}), ("change_password.php", {"current_password": "x", "new_password": "y" * 12, "new_password_confirm": "y" * 12}),
                      (f"scan_view2.php?uuid={scan_a}", {"action": "retry"}),
                      ("logout.php", {})],
        "doctor_a": [(f"review_form.php?scan_uuid={scan_a}", {"scan_uuid": scan_a, "agrees_with_ai": "yes", "notes": "csrf", "do": "finalize"})],
        "admin": [("admin_users.php", {"action": "create_clinician", "full_name": "csrf", "email": "csrf@e4.test", "role": "doctor"}),
                  ("admin_users.php", {"action": "toggle_active", "user_id": 6}),
                  ("admin_settings.php", {"action": "save", "inference_endpoint_url": "http://evil.invalid", "inference_api_key": ""}),
                  ("admin_settings.php", {"action": "test"}),
                  ("admin_assign.php", {"action": "assign", "patient_id": 6, "provider_id": 3, "relationship_type": "primary_doctor"}),
                  ("admin_assign.php", {"action": "unassign", "link_id": 1}),
                  ("admin_user_edit.php?id=6", {"id": 6, "full_name": "csrf", "email": "patient_b@e4.test"})],
        None: [("login.php", {"email": site.acc["patient_a"]["email"], "password": site.acc["patient_a"]["password"]}),
               ("register.php", {"full_name": "csrf", "email": "csrfreg@e4.test", "password": "E4-test-pass-123",
                                 "password_confirm": "E4-test-pass-123", "consent": "1"})],
    }
    tried, bad = 0, []
    for who, fs in forms.items():
        s = site.logged_in(who) if who else site.session()
        if not who:
            s.get(site.url("login.php"))
        for page, data in fs:
            for variant, tok in (("missing", None), ("wrong", "0" * 64)):
                tried += 1
                d = dict(data)
                if tok:
                    d["csrf_token"] = tok
                r = s.post(site.url(page), data=d, allow_redirects=False)
                if r.status_code != 403:
                    bad.append(f"{page} ({variant}: HTTP {r.status_code})")
    record("CSRF-1", "CSRF", f"{tried // 2} state-changing forms posted without a token and with a wrong token",
           "all 403", f"{tried - len(bad)} of {tried} refused" + (f"; accepted: {bad}" if bad else ""), not bad)


def t_upload(site, images):
    print("UP", flush=True)
    s = site.logged_in("patient_a")
    main = (images / "2_main_slice.png").read_bytes()
    cases = [
        ("UP-1", "PHP script named .png", {"file": ("shell.png", b"<?php echo 'E4-RCE'; ?>")}, False),
        ("UP-2", "HTML page named .png", {"file": ("page.png", b"<html><script>alert(1)</script></html>")}, False),
        ("UP-3", "SVG with script", {"file": ("x.svg", b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>')}, False),
        ("UP-4", "GIF image", {"file": ("x.gif", b"GIF89a\x01\x00\x01\x00\x00\x00\x00;")}, False),
        ("UP-5", "decompression bomb (6000 x 6000 PNG, small file)", {"file": ("bomb.png", png_bytes(6000, 6000))}, False),
        ("UP-6", "file over the size limit (11 MB)", {"file": ("big.png", png_bytes(64, 64, extra=b"\x00" * (11 * 1024 * 1024)))}, False),
        ("UP-7", "empty file", {"file": ("empty.png", b"")}, False),
    ]
    for tid, what, files, should_accept in cases:
        r = upload(site, s, files)
        accepted = r.status_code == 302 and uuid_from(r)
        record(tid, "UP", f"upload rejected: {what}", "rejected with a message", "accepted" if accepted else f"rejected (HTTP {r.status_code})",
               not accepted)

    # UP-8: polyglot (valid PNG + PHP after IEND) is accepted but re-encoded: payload gone, served as image only
    payload = b"<?php echo 'E4-POLYGLOT'; ?>"
    r = upload(site, s, {"file": ("../../../evil.php", png_bytes(224, 224, extra=payload))})
    u = uuid_from(r)
    if not u:
        record("UP-8", "UP", "PNG with PHP appended: stored re-encoded under a random name", "accepted, payload stripped",
               f"not accepted (HTTP {r.status_code})", False)
    else:
        img = s.get(site.url(f"scan_image.php?uuid={u}"))
        ok = img.status_code == 200 and payload not in img.content and img.headers.get("Content-Type", "").startswith("image/png")
        record("UP-8", "UP", "PNG with PHP appended and a '../' file name: re-encoded, random name", "payload absent from stored file",
               f"stored file {'has no' if payload not in img.content else 'CONTAINS the'} payload; type {img.headers.get('Content-Type')}", ok)


def t_xss(site, images):
    print("XSS", flush=True)
    raw = XSS
    hits = []
    # stored via patient notes on upload, via registration name, via failed-login email (audit log), via review notes
    main = (images / "2_main_slice.png").read_bytes()
    s = site.logged_in("patient_a")
    r = upload(site, s, {"file": (raw + ".png", main), "prev_file": ("p.png", (images / "1_previous_slice.png").read_bytes()),
                         "next_file": ("n.png", (images / "3_next_slice.png").read_bytes())}, notes=raw)
    u = uuid_from(r)
    for page in ["dashboard_patient.php", f"scan_view2.php?uuid={u}"]:
        if raw in s.get(site.url(page)).text:
            hits.append(f"patient:{page}")
    d = site.logged_in("doctor_a")
    for page in ["dashboard_provider.php", "provider_patient.php?patient_id=5", f"scan_view2.php?uuid={u}", f"review_form.php?scan_uuid={u}"]:
        if raw in d.get(site.url(page)).text:
            hits.append(f"doctor:{page}")
    tok = site.get_token(d, f"review_form.php?scan_uuid={u}")
    d.post(site.url(f"review_form.php?scan_uuid={u}"), data={"csrf_token": tok, "scan_uuid": u, "agrees_with_ai": "yes",
                                                              "notes": raw, "do": "finalize"})
    for who, sess in (("patient", s), ("doctor", d)):
        if raw in sess.get(site.url(f"scan_view2.php?uuid={u}")).text:
            hits.append(f"{who}:scan_view2 after review")
    # registration name and failed login email -> admin pages
    rs = site.session()
    tok = site.get_token(rs, "register.php")
    rs.post(site.url("register.php"), data={"csrf_token": tok, "full_name": raw, "email": f"xss-{uuid.uuid4().hex[:6]}@e4.test",
                                            "password": "E4-test-pass-123", "password_confirm": "E4-test-pass-123", "consent": "1"})
    ls = site.session()
    tok = site.get_token(ls, "login.php")
    ls.post(site.url("login.php"), data={"csrf_token": tok, "email": raw, "password": "x"})
    a = site.logged_in("admin")
    for page in ["admin_users.php", "admin_audit_log.php", "dashboard_admin.php", "admin_assign.php"]:
        if raw in a.get(site.url(page)).text:
            hits.append(f"admin:{page}")
    # reflected: login email field, audit log filter
    ls2 = site.session()
    tok = site.get_token(ls2, "login.php")
    if raw in ls2.post(site.url("login.php"), data={"csrf_token": tok, "email": raw, "password": "x"}).text:
        hits.append("reflected:login email")
    if raw in a.get(site.url("admin_audit_log.php"), params={"action": raw, "date_from": raw}).text:
        hits.append("reflected:audit filter")
    record("XSS-1", "XSS", "script payload in upload name, notes, review notes, registration name, login email and "
           "audit filter; checked on every page that shows them, for three roles", "always escaped", "escaped everywhere" if not hits else f"raw on: {hits}", not hits)
    return u


def t_headers(site):
    print("HDR", flush=True)
    r = requests.get(site.url("login.php"))
    want = ["X-Content-Type-Options", "X-Frame-Options", "Content-Security-Policy", "Referrer-Policy", "Permissions-Policy"]
    missing = [h for h in want if h not in r.headers]
    record("HDR-1", "HDR", "security headers on every page", "5 headers present",
           "all present" if not missing else f"missing {missing}", not missing)
    csp = r.headers.get("Content-Security-Policy", "")
    record("HDR-2", "HDR", "Content-Security-Policy blocks inline script", "no 'unsafe-inline' in script-src",
           "script-src allows 'unsafe-inline'" if "script-src 'self' 'unsafe-inline'" in csp else "inline script blocked",
           "unsafe-inline" not in csp.split("script-src", 1)[-1].split(";")[0])
    sc = r.headers.get("Set-Cookie", "")
    record("HDR-3", "HDR", "session cookie flags", "HttpOnly and SameSite (Secure needs HTTPS: checked on the host)",
           sc.split(";", 1)[-1].strip() if sc else "no cookie", "HttpOnly" in sc and "SameSite" in sc)
    r = requests.get(site.url("includes/"), allow_redirects=False)
    r2 = requests.get(site.url("api/"), allow_redirects=False)
    record("HDR-4", "HDR", "no directory listings", "403/404", f"includes/ {r.status_code}, api/ {r2.status_code}",
           "Index of" not in r.text and "Index of" not in r2.text)
    r = requests.get(site.url("login.php"))
    leak = [h for h in ("X-Powered-By",) if h in r.headers]
    record("HDR-5", "HDR", "no PHP version in response headers", "no X-Powered-By",
           f"X-Powered-By: {r.headers.get('X-Powered-By')}" if leak else "absent", not leak)

    # HDR-6: a database fault (an assignment to an account number that does not exist breaks a foreign key on the
    # original code) must not print PHP's error text, file paths or the query into the page
    adm = site.logged_in("admin")
    r = adm.post(site.url("admin_assign.php"), data={"csrf_token": site.get_token(adm, "admin_assign.php"), "action": "assign",
                 "patient_id": site.acc["patient_a"]["id"], "provider_id": 999999, "relationship_type": "primary_doctor"})
    leak = [m for m in ("Fatal error", "PDOException", "SQLSTATE", "Stack trace", " on line ") if m in r.text]
    record("HDR-6", "HDR", "no PHP error text in pages after a database fault", "no error text, file path or query",
           f"HTTP {r.status_code}" + (f", page shows {leak}" if leak else ", no error text"), not leak)


def t_inference(base, key, images):
    print("INF", flush=True)
    main = (images / "2_main_slice.png").read_bytes()
    f = {"file": ("m.png", main, "image/png")}
    tries = {"no key": {}, "wrong key": {"X-API-Key": "wrong"}, "empty key": {"X-API-Key": ""},
             "key with a space": {"X-API-Key": key + " "}, "key prefix": {"X-API-Key": key[:-1]}}
    bad = []
    for k, h in tries.items():
        if requests.get(f"{base}/api/health", headers=h).status_code != 401:
            bad.append(f"health/{k}")
        if requests.post(f"{base}/api/predict", headers=h, files=f, data={"modality": "CT"}).status_code != 401:
            bad.append(f"predict/{k}")
    record("INF-1", "INF", "health and predict without the correct key (5 variants each)", "all 401",
           f"{10 - len(bad)} of 10 refused" + (f"; accepted: {bad}" if bad else ""), not bad)
    h = {"X-API-Key": key}
    cases = {"PHP script as image": ("x.png", b"<?php echo 1; ?>"), "empty file": ("e.png", b""),
             "bomb 6000x6000": ("b.png", png_bytes(6000, 6000)), "bomb 12000x12000": ("b2.png", png_bytes(12000, 12000)),
             "truncated PNG": ("t.png", main[:200])}
    out = []
    for k, (fn, data) in cases.items():
        r = requests.post(f"{base}/api/predict", headers=h, files={"file": (fn, data, "image/png")}, data={"modality": "CT"}, timeout=120)
        out.append((k, r.status_code))
    alive = requests.get(f"{base}/api/health", headers=h).status_code == 200
    record("INF-2", "INF", "malformed images sent with the correct key", "4xx answer, server stays up",
           ", ".join(f"{k}: {c}" for k, c in out) + f"; server {'up' if alive else 'DOWN'}",
           alive and all(400 <= c < 500 for _, c in out))
    r = requests.post(f"{base}/api/predict", headers=h, files=f, data={"modality": "XRAY"})
    record("INF-3", "INF", "unknown modality", "4xx", f"HTTP {r.status_code}", 400 <= r.status_code < 500)
    r = requests.options(f"{base}/api/predict", headers={"Origin": "https://evil.invalid", "Access-Control-Request-Method": "POST"})
    acao = r.headers.get("Access-Control-Allow-Origin")
    record("INF-4", "INF", "no cross-origin access for browsers", "no Access-Control-Allow-Origin",
           f"ACAO={acao}" if acao else "absent", not acao)


# --------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--inference", required=True)
    ap.add_argument("--accounts", required=True)
    ap.add_argument("--inference-key-file", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--mysql", default=r"C:\xampp\mysql\bin\mysql.exe")
    ap.add_argument("--db", default="neuroscan_e4test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=[])
    a = ap.parse_args()
    for u in (a.site, a.inference):
        if urlparse(u).hostname not in LOCAL:
            sys.exit(f"refusing non-local target {u}")
    import subprocess
    accounts = json.loads(Path(a.accounts).read_text())
    key = Path(a.inference_key_file).read_text().strip()
    images = Path(a.images)
    site = Site(a.site, accounts)

    def sql(q):
        subprocess.run([a.mysql, "-u", "root", a.db, "-e", q], check=True)

    def sql_value(q):
        return subprocess.run([a.mysql, "-u", "root", "-N", a.db, "-e", q], check=True, capture_output=True,
                              text=True).stdout.strip()

    def reset_lock(name):
        sql(f"UPDATE users SET failed_login_count=0, locked_until=NULL WHERE email='{accounts[name]['email']}'")

    def reset_ip():
        # clears this address's recent failed logins and self-registrations, so one test's throttling cannot
        # affect the next (the throttle tests themselves need a clean start)
        sql("DELETE FROM login_attempts WHERE success=0; "
            "UPDATE audit_log SET action='user_created_e4_reset' WHERE action='user_created'")

    # a scan of patient A used by the access-control and CSRF tests
    s = site.logged_in("patient_a")
    r = upload(site, s, {"file": ("main.png", (images / "2_main_slice.png").read_bytes()),
                         "prev_file": ("prev.png", (images / "1_previous_slice.png").read_bytes()),
                         "next_file": ("next.png", (images / "3_next_slice.png").read_bytes())})
    scan_a = uuid_from(r)
    if not scan_a:
        sys.exit(f"setup: upload by patient A failed (HTTP {r.status_code})")
    page = s.get(site.url(f"scan_view2.php?uuid={scan_a}")).text
    print(f"setup: patient A scan {scan_a}; ischemic 99.7% shown: {'yes' if '99.7%' in page else 'no'}", flush=True)

    run = lambda c: not a.only or c in a.only
    reset_ip()
    if run("AUTH"): t_auth(site, reset_lock, reset_ip)
    if run("AC"): t_access(site, scan_a, sql_value)
    if run("CSRF"): t_csrf(site, scan_a)
    if run("UP"): t_upload(site, images)
    if run("XSS"): t_xss(site, images)
    if run("HDR"): t_headers(site)
    if run("INF"): t_inference(a.inference, key, images)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "site": a.site, "inference": a.inference,
            "tests": len(RESULTS), "passed": sum(r["pass"] for r in RESULTS)}
    (out / "e4_results.json").write_text(json.dumps({"run": meta, "results": RESULTS}, indent=1))
    print(f"\n{meta['passed']} of {meta['tests']} passed -> {out}")


if __name__ == "__main__":
    main()
