"""Seed the invented test accounts used by e4_suite.py into a local test database (local XAMPP only), point the
database's inference settings at a local inference server on 127.0.0.1:8765, and write a random throw-away inference
key. Accounts and key are written to a folder OUTSIDE the repository, never to this folder.

    python -I seed_test_accounts.py <test database> <output folder outside the repository>

Then start the inference server with NEUROSCAN_API_KEY set to the key in <output folder>/inference_test_key.txt and run
e4_suite.py with --accounts <output folder>/test_accounts.json --inference-key-file <output folder>/inference_test_key.txt.
Used for the rebuild check of the released web application (outputs/security/4_released_code/)."""
import json
import secrets
import sys
import subprocess
from pathlib import Path

if len(sys.argv) != 3:
    sys.exit(__doc__)
HERE = Path(sys.argv[2])
HERE.mkdir(parents=True, exist_ok=True)
MYSQL = r"C:\xampp\mysql\bin\mysql.exe"
PHP = r"C:\xampp\php\php.exe"
DB = sys.argv[1]
KEY = secrets.token_hex(24)


def sql(q):
    return subprocess.run([MYSQL, "-u", "root", "-N", DB, "-e", q], check=True, capture_output=True,
                          text=True).stdout.strip()


def phash(pw):
    return subprocess.run([PHP, "-r", "echo password_hash(stream_get_contents(STDIN), PASSWORD_DEFAULT);"],
                          input=pw, capture_output=True, text=True, check=True).stdout


SPEC = [("admin", "admin", 0), ("doctor_a", "doctor", 0), ("doctor_b", "doctor", 0), ("radiologist", "radiologist", 0),
        ("patient_a", "patient", 0), ("patient_b", "patient", 0), ("lock_victim", "patient", 0),
        ("new_doctor", "doctor", 1)]
acc = {}
for name, role, must in SPEC:
    pw = "T!" + secrets.token_urlsafe(12)
    email = f"{name}@rel.test"
    sql(f"INSERT INTO users (role, full_name, email, password_hash, must_change_password) VALUES "
        f"('{role}', 'Test {name}', '{email}', '{phash(pw)}', {must})")
    uid = int(sql(f"SELECT id FROM users WHERE email='{email}'"))
    if role == "patient":
        sql(f"INSERT INTO patient_profiles (user_id) VALUES ({uid})")
    acc[name] = {"id": uid, "role": role, "email": email, "password": pw}
adm = acc["admin"]["id"]
sql(f"INSERT INTO doctor_patient (patient_id, provider_id, relationship_type, assigned_by) VALUES "
    f"({acc['patient_a']['id']}, {acc['doctor_a']['id']}, 'primary_doctor', {adm}), "
    f"({acc['patient_a']['id']}, {acc['radiologist']['id']}, 'radiologist', {adm})")
sql(f"UPDATE settings SET setting_value='http://127.0.0.1:8765' WHERE setting_key='inference_endpoint_url'; "
    f"UPDATE settings SET setting_value='{KEY}' WHERE setting_key='inference_api_key'")
(HERE / "test_accounts.json").write_text(json.dumps(acc, indent=1))
(HERE / "inference_test_key.txt").write_text(KEY)
print("seeded", {k: (v["id"], v["role"]) for k, v in acc.items()})
