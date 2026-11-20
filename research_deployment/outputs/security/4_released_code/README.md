# Rebuild check of the released web application (21 Nov 2026)

The web application in `../../../webapp/` was copied to a fresh local XAMPP folder (Apache 2.4.43, PHP 7.4.7,
MariaDB 10.4.13 on Windows 11), laid out as its README describes (`includes/` and `storage/` inside the web root,
database name given by `SetEnv`), with a new database built from `webapp/sql/` (schema, then the migrations in the
README's order). Invented accounts and a random inference key were created with
`../../code/security/seed_test_accounts.py`, and a local inference server was started from this repository with the
hash-checked `requirements.lock` (ONNX Runtime 1.20.1) on 127.0.0.1. The 41-test suite
`../../code/security/e4_suite.py`, unchanged since the tests of 31 Oct 2026, then passed **41 of 41**
(`e4_results.json`).

Nothing was run against the public site or the live inference service. The accounts and the key stayed in a folder
outside the repository; the result file contains neither.

One correction came out of the check: the README's cPanel instructions had told deployers to put `includes/` and
`storage/` above the web folder, but every page loads `./includes/`, so the site only works with them inside it, as on
the public host and in the earlier test copies. The README now says so.
