"""Builds e2e3.py (the pushed kernel) from e2e3_src.py by inserting the committed server inference.py verbatim.
Usage: python -I build_kernel.py"""
import hashlib
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER = HERE.parents[1] / "render_inference_server"
commit = subprocess.run(["git", "-C", str(SERVER), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                        check=True).stdout.strip()
inf = subprocess.run(["git", "-C", str(SERVER), "show", "HEAD:inference.py"], capture_output=True, check=True).stdout.decode("utf-8")
sha = hashlib.sha256(inf.encode("utf-8")).hexdigest()
src = (HERE / "e2e3_src.py").read_text(encoding="utf-8")
assert src.count('"__INFERENCE_PY__"') == 1 and src.count('"__INFERENCE_PY_SHA__"') == 1
out = src.replace('"__INFERENCE_PY__"', repr(inf)).replace('"__INFERENCE_PY_SHA__"', repr(sha))
(HERE / "e2e3.py").write_text(out, encoding="utf-8")
print(f"e2e3.py written: inference.py from render_inference_server@{commit}, sha256 {sha}")
