# NeuroScan clinical web app (PHP + MySQL)

Multi-user version of the NeuroScan demo: patients, radiologists, doctors,
and admins with accounts, uploads, and AI-assisted predictions backed by a
database. Vanilla PHP + PDO, no framework, no Composer, no build step —
built to run on ordinary shared cPanel hosting (PHP + MySQL only).

**M0–M4 are built**: auth/RBAC, patient self-registration, scan upload →
AI prediction, admin-provisioned doctor/radiologist accounts (with a
forced password change on first login), patient–provider assignments,
the full review workflow (pending-review queue, review/annotate/
finalize, corrections that preserve the original as an audit trail,
patient-side display of finalized reviews), an admin overview dashboard
(live counts + endpoint health), user editing, and a filterable/
paginated audit log viewer.

**M6** (hosting the model at a public URL) is done — the inference
service runs as an always-on deployment at `render_inference_server/`
(see that folder's README for setup). A notebook-based fallback also
exists at `kaggle_kernels/phase_o_inference_server/inference_server.ipynb`
for local/manual runs, but isn't what the live site points at.

**M5** (code-level hardening) is done — see "Security audit findings"
below. What's left of M5 is the parts that need your actual cPanel
account to verify (HTTPS/AutoSSL, PHP version/extensions, file-layout
permissions) — see the Deployment section.

**Not a diagnostic device.** Every AI prediction must be confirmed by a
licensed radiologist or doctor. This applies regardless of hosting.

## Architecture

The PyTorch model (`webapp/inference.py`'s `ThreeClassPipeline` — Phase O +
TTA, unchanged) cannot run on shared PHP hosting. It runs separately, as
an always-on Docker deployment on Render (`render_inference_server/`),
and this site calls it over HTTPS as a JSON API. The endpoint's URL and
a shared API key are configured by an admin in Settings; unlike the
earlier Kaggle/Colab + ngrok approach, the URL is permanent and doesn't
need updating in normal operation.

## Local development

Requires PHP 7.4+ with `pdo_mysql`, `gd`, `fileinfo`, `mbstring`, `curl`,
`openssl`, and a local MySQL server (e.g. XAMPP on Windows).

```bash
# 1. Create the database, import the schema, then the migrations in this order
mysql -u root -e "CREATE DATABASE neuroscan CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
mysql -u root neuroscan < sql/schema.sql
for f in migrate_002_api_tokens migrate_003_heatmap migrate_004_ood_guard migrate_005_pathway \
         migration_gate_patient_results; do mysql -u root neuroscan < sql/$f.sql; done

# 2. Set DB credentials (or edit includes/config.php's defaults directly)
export NEUROSCAN_DB_HOST=127.0.0.1
export NEUROSCAN_DB_NAME=neuroscan
export NEUROSCAN_DB_USER=root
export NEUROSCAN_DB_PASS=

# 3. Create the first admin account
php sql/seed_admin.php

# 4. Put includes/ and storage/ inside the web root (the pages load ./includes/), then serve public/
cp -r includes storage public/
php -S localhost:8000 -t public
```

Under Apache (for example XAMPP), the database settings can instead be given with `SetEnv NEUROSCAN_DB_NAME ...`
(and the other three variables) in an `.htaccess` file in the folder above `public/`.

Then open http://localhost:8000. Log in as the admin you just created,
go to Settings, and point `inference_endpoint_url` at a running instance
of the existing `webapp/app.py` (e.g. `http://localhost:8000` if you run
that Python app on a different port) to test the full upload → predict
flow without needing the real Render deployment up locally.

## Deploying to cPanel shared hosting

1. **Database**: cPanel → *MySQL Databases* → create a database and a
   dedicated DB user (cPanel prefixes both with your account username,
   e.g. `youruser_neuroscan`). Grant that user only what it needs
   (SELECT/INSERT/UPDATE/DELETE). Import `sql/schema.sql` via cPanel →
   *phpMyAdmin* → Import.
2. **File layout**: upload the contents of `public/` into the web folder
   (for example `public_html/neuroscan/`), and put `includes/` and
   `storage/` **inside that same folder**: every page loads
   `./includes/`, and `includes/config.php` places `storage/` next to
   `includes/`. The `.htaccess` files in `includes/` and `storage/` deny
   all direct HTTP access. Keep `sql/` (schema, migrations and the
   command-line `seed_admin.php`) **outside** the web folder.
3. **`includes/config.php`**: edit directly on the server with the real
   DB host/name/user/password from step 1. Never commit real credentials
   to any repository.
4. **PHP version/extensions**: cPanel → *MultiPHP Manager* → select PHP
   8.1+ for the domain. cPanel → *Select PHP Version* → enable
   `pdo_mysql`, `curl`, `gd`, `fileinfo`, `mbstring`, `openssl`.
5. **SSL**: cPanel → *SSL/TLS Status* → run AutoSSL if not already active.
   Then uncomment the HTTPS redirect block in `public/.htaccess`, and set
   `FORCE_HTTPS_COOKIES` to `true` in `includes/config.php`.
6. **First admin**: SSH in and run `php sql/seed_admin.php` if your plan
   has shell access; otherwise temporarily place a guarded one-time
   script to call the same logic over HTTP and delete it immediately
   after use — do not leave an unauthenticated admin-creation path live.
7. **Inference endpoint**: deploy `render_inference_server/` (see its
   README), then paste its permanent URL and the matching API key into
   Settings, and use *Test connection* before telling anyone to rely on
   the site.

## Release and rebuild check

This folder is the web application as deployed after the security tests reported in the deployment paper, with no
credentials: `includes/config.php` holds only local defaults, and the inference address and key are entered in the
admin Settings page. It was rebuilt from this folder on a fresh local XAMPP copy (Apache 2.4.43, PHP 7.4.7,
MariaDB 10.4.13) with a new database built from `sql/`, invented accounts and a local inference server with a random
key, and the 41-test security suite in `../research_deployment/code/security/` passed 41 of 41
(`../research_deployment/outputs/security/4_released_code/`). `includes/result_gate.php` and the setting
`gate_patient_results` (an option to hold a prediction from the patient until a clinician has reviewed it, off by
default) were added on 7 Nov 2026, after the security tests of 31 Oct; the rebuild check covers them.
`SHA256SUMS.txt` fingerprints every file.

## Security notes

- All queries use PDO prepared statements; uploads are re-encoded
  server-side (strips EXIF/embedded payloads) and stored outside
  `public_html` under random filenames, served only through
  `scan_image.php` after a session/ownership check — there is no direct
  static URL to a patient's image.
- CSRF tokens on every state-changing form; login is rate-limited (5
  failed attempts locks the account for 15 minutes).
- Patients can self-register. Doctor/radiologist accounts are
  admin-provisioned only, via `admin_users.php` — those accounts can see
  other people's medical data, so that privilege is never self-granted.
  Admin-created accounts get a random temporary password and
  `must_change_password`, which forces a password change on first login
  before anything else in the app is reachable.
- A doctor/radiologist can only see or review a patient's scans if an
  admin has created an active `doctor_patient` link between them
  (`admin_assign.php`) — checked server-side on every scan/review page,
  not just hidden in the UI.
- A finalized review is never edited in place: a correction inserts a
  new `reviews` row with `amends_review_id` pointing at the original,
  which is preserved as-is — a real audit trail for clinical
  accountability rather than silently rewritten history.
- Every login, scan upload, prediction attempt, settings change,
  patient–provider assignment, and review action is written to
  `audit_log`.

## Security audit findings (M5)

A systematic pass across every file in `public/` and `includes/`, checking
CSRF coverage, SQL injection, XSS output escaping, open redirects, audit
log completeness, and upload validation.

**Fixed:**
- **Decompression-bomb upload DoS**: the 10 MB file-size cap didn't stop a
  small, highly-compressed image (e.g. a huge solid-color PNG) from
  decoding into a multi-gigabyte pixel buffer once `imagecreatefrompng()`
  ran. `upload_validate.php` now rejects images over 4096px per side
  *before* decoding, using `getimagesize()`'s header-only read. Verified
  with a 120 KB / 6000×6000 test file (rejected) and a real sample image
  (still accepted, no regression).
- Two extra security headers added (`Referrer-Policy: same-origin`,
  a restrictive `Permissions-Policy`) in `public/.htaccess`.

**Checked, no issues found:**
- CSRF: every state-changing form has a token, every POST handler
  validates it (13/13 forms; the one form without a token is the audit
  log's read-only GET filter, correctly exempt).
- SQL injection: every query with a variable uses PDO bound parameters;
  the one query with dynamic `WHERE`-clause construction
  (`admin_audit_log.php`) builds it only from a fixed set of hardcoded
  condition strings, never from interpolated user input.
- XSS: every user-controlled value reaching HTML output is either
  `htmlspecialchars(..., ENT_QUOTES)`-escaped, cast to `(int)`/`(float)`,
  or drawn from a fixed enum array — including audit log `details`
  (JSON built from user-submitted text like a failed-login email is
  still escaped at render time, so embedded HTML can't execute).
- Open redirects: every `Location:` header target is a hardcoded relative
  path; the only variable part is `urlencode()`-d and can't change the
  redirect's host.
- Audit log completeness: all 19 documented actions are confirmed firing
  in code.

**Deliberately deferred, not forgotten:**
- **Field-level encryption** for `patient_profiles.notes` /
  `reviews.notes`: still not implemented. Recommendation stands from the
  original design — defer until there's a real deployment with real
  patient data, since it adds real key-management risk (a lost key means
  unrecoverable clinical notes) for a benefit that mostly duplicates what
  TLS-in-transit and restricted DB-account access already provide on
  typical shared hosting. Revisit if that risk calculus changes.
- **No rate limiting on patient self-registration**: a script could spam
  `register.php` with fake patient accounts. Lower severity than login
  brute-forcing (no data exposure, just DB clutter) and not fixed in this
  pass — flagged here rather than silently left out.
- Everything that needs your actual cPanel account to verify (HTTPS via
  AutoSSL, PHP version/extensions, whether your plan allows writing
  outside `public_html`) is necessarily still open — see Deployment above.
