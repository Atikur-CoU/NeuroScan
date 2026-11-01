"""Builds peak.py (the pushed kernel) from peak_src.py by inserting the committed server inference.py, app.py and
requirements.txt verbatim. Usage: python -I build_kernel.py"""
import hashlib
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER = HERE.parents[1] / "render_inference_server"


def head(name):
    return subprocess.run(["git", "-C", str(SERVER), "show", f"HEAD:{name}"], capture_output=True, check=True).stdout.decode("utf-8")


commit = subprocess.run(["git", "-C", str(SERVER), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                        check=True).stdout.strip()
src = (HERE / "peak_src.py").read_text(encoding="utf-8")
subs = {}
for tag, fname in (("INFERENCE_PY", "inference.py"), ("APP_PY", "app.py")):
    text = head(fname)
    subs[f'"__{tag}__"'] = repr(text)
    subs[f'"__{tag}_SHA__"'] = repr(hashlib.sha256(text.encode("utf-8")).hexdigest())
subs['"__REQUIREMENTS_TXT__"'] = repr(head("requirements.txt"))
for k, v in subs.items():
    assert src.count(k) == 1, k
    src = src.replace(k, v)
(HERE / "peak.py").write_text(src, encoding="utf-8")
print(f"peak.py written from render_inference_server@{commit}")
