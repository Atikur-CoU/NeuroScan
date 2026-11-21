"""Builds single_slice.py (the pushed kernel) from single_src.py, inserting the committed server inference.py and the
fidelity-detail kernel's fidelity.py verbatim. Usage: python -I build_kernel.py"""
import hashlib
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER = HERE.parents[1] / "render_inference_server"
commit = subprocess.run(["git", "-C", str(SERVER), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                        check=True).stdout.strip()
inf = subprocess.run(["git", "-C", str(SERVER), "show", "HEAD:inference.py"], capture_output=True, check=True).stdout.decode("utf-8")
sha = hashlib.sha256(inf.encode("utf-8")).hexdigest()
src = (HERE / "single_src.py").read_text(encoding="utf-8")
for k in ('"__INFERENCE_PY__"', '"__INFERENCE_PY_SHA__"'):
    assert src.count(k) == 1, k
out = src.replace('"__INFERENCE_PY__"', repr(inf)).replace('"__INFERENCE_PY_SHA__"', repr(sha))
(HERE / "single_slice.py").write_text(out, encoding="utf-8")
print(f"single_slice.py written: inference.py from render_inference_server@{commit} (sha256 {sha[:16]})")
