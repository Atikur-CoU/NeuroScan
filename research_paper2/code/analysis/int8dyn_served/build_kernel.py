"""Builds served.py (the pushed kernel) from served_src.py, inserting the committed server inference.py and the
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
fid = (HERE.parent / "paper2_fidelity_detail" / "fidelity.py").read_text(encoding="utf-8")
src = (HERE / "served_src.py").read_text(encoding="utf-8")
for k in ('"__INFERENCE_PY__"', '"__INFERENCE_PY_SHA__"', '"__FIDELITY_PY__"'):
    assert src.count(k) == 1, k
out = src.replace('"__INFERENCE_PY__"', repr(inf)).replace('"__INFERENCE_PY_SHA__"', repr(sha)).replace('"__FIDELITY_PY__"', repr(fid))
(HERE / "served.py").write_text(out, encoding="utf-8")
print(f"served.py written: inference.py from render_inference_server@{commit} (sha256 {sha[:16]}), fidelity.py {hashlib.sha256(fid.encode()).hexdigest()[:16]}")
