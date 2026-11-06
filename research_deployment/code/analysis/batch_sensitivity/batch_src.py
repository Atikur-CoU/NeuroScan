"""
NeuroScan paper 2 -- how far INT8-dynamic's outputs depend on the batch (2026-11-05). CPU only; no training; no new
classification result; held-out slices are only scored. Plan: paper/second_paper/PLAN_batch_sensitivity.md.

Stage 0, controls (any failure stops the run): the one-request run reproduces the saved served outputs; the same batch
run twice is identical; 1 and 4 threads agree; FP32 and INT8-static do not change in batches or beside a stress partner.
Stage 1: each of a slice's four views run alone, then averaged, against the served run (four views together).
Stage 2: 476 target slices (all FP32 near-ties, plus 15 others per dataset-and-label stratum), each beside each of 12
partner slices (10 chosen by rule from the decoded images, 2 stress images), as a batch of two slices.
Stage 3: all 10,602 slices shuffled into batches of 2, 4, 8, 16 and 32, plus a second shuffle at 16.
Every condition is compared with the served run and with FP32, using the served notebook's compare() unchanged.
"""
import hashlib
import io
import json
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

T0 = time.time()


def log(m):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {m}", flush=True)


t = time.time()
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime==1.20.1"], capture_output=True, text=True)
log(f"pip onnxruntime==1.20.1: exit {r.returncode} ({time.time() - t:.0f}s) {r.stderr[-300:]}")
if r.returncode:
    sys.exit(1)

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

VALID_LABELS = ("normal", "ischemic", "hemorrhagic")
LABEL_TO_IDX = {k: i for i, k in enumerate(VALID_LABELS)}
TARGET_SIZE = (224, 224)
MRI_PAIR_DATASETS = ("ISLES2022_dwi_adc", "SOOP_dwi_adc")
EXCLUDED_DATASETS = ("MRI_Images",)
N_BOOT = 2000
N_CHECK = 256
WORKERS = 4
SIZES = (16, 2, 32, 8, 4)  # priority order
MAX_RSS_GB = 24.0
LIMIT_START_H = 10.5
LIMIT_PROJECTED_MIN = 600
KEYS = ["accuracy", "balanced_accuracy", "macro_f1_present_classes", "normal_recall", "ischemic_recall", "ischemic_precision",
        "hemorrhagic_recall", "hemorrhagic_precision", "ischemic_auroc", "hemorrhagic_auroc", "macro_ovr_auroc"]
OUT = Path("/kaggle/working/output")
OUT.mkdir(parents=True, exist_ok=True)
R = {"note": __doc__.strip().splitlines()[0], "onnxruntime": ort.__version__, "skipped": [], "timing_min": {}}


def save():
    (OUT / "batch_sensitivity_results.json").write_text(json.dumps(R, indent=2, default=float))


def first(*cands, what=""):
    """Mount-path fallback: exists() on each candidate, never a listing."""
    for c in cands:
        if Path(c).exists():
            return Path(c)
    print(f"[ERROR] {what} not found; tried {[str(c) for c in cands]}")
    sys.exit(1)


def ds_root(slug):
    return first(f"/kaggle/input/datasets/atik68/{slug}", f"/kaggle/input/{slug}", what=f"dataset {slug}")


def nb_file(slug, name):
    roots = (f"/kaggle/input/notebooks/atik68/{slug}", f"/kaggle/input/{slug}")
    return first(*(f"{r}{sub}/{name}" for r in roots for sub in ("", "/output")), what=f"{slug}/{name}")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

t = time.time()
SPLIT_DIR = ds_root("neuroscan-phase-p-patient-split-v3")
PHASE_B = ds_root("neuroscan-phase-b-images")
RSNA_J = ds_root("neuroscan-phase-j-rsna-images")
RSNA_L = ds_root("neuroscan-phase-l-rsna-more-images")
AISD_K = ds_root("neuroscan-phase-k-aisd-images")
CPAISD_V2_IMG = ds_root("neuroscan-phase-n-cpaisd-v2-images")
SAVED = ds_root("neuroscan-q-round30as123-preds")
CG = "neuroscan-p2-compression-gradcam"
MODELS = {"FP32": nb_file("neuroscan-q-onnx-export-check", "neuroscan_r30a_s123.onnx"),
          "INT8-dyn": nb_file(CG, "models/neuroscan_r30a_s123_int8_dynamic.onnx"),
          "INT8-static": nb_file(CG, "models/neuroscan_r30a_s123_int8_static.onnx")}
PARTIAL = {k: np.load(nb_file(CG, f"partial/{k}.npz")) for k in MODELS}
PER_ROW = pd.read_csv(nb_file(CG, "e2_per_row_probs.csv"))
ZIPS = {"AISD/normal/": nb_file("neuroscan-q-aisd-normals-prep", "aisd_normal_images.zip"),
        "RSNA_ICH/normal/": nb_file("neuroscan-q-rsna-normals-prep", "rsna_normal_images.zip"),
        "SOOP/": nb_file("neuroscan-q-soop-prep", "soop_images.zip")}
CSV = {"aisd_normal": nb_file("neuroscan-q-aisd-normals-prep", "aisd_normal_rows.csv"),
       "rsna_normal": nb_file("neuroscan-q-rsna-normals-prep", "rsna_normal_rows.csv"),
       "soop": nb_file("neuroscan-q-soop-prep", "soop_rows.csv")}
SERVED = np.load(nb_file("neuroscan-p2-int8dyn-served", "INT8-dyn_served.npz"))
R["int8_dynamic_sha256"] = sha256(MODELS["INT8-dyn"])
assert R["int8_dynamic_sha256"].startswith("ad2f6e0ee22b28ec"), R["int8_dynamic_sha256"]
log(f"inputs resolved ({time.time() - t:.1f}s); INT8-dynamic {R['int8_dynamic_sha256'][:16]}")

# The server's inference.py, verbatim (inserted by build_kernel.py): alert tiers and the display rule.
INFERENCE_PY = "__INFERENCE_PY__"
INFERENCE_PY_SHA = "__INFERENCE_PY_SHA__"
assert hashlib.sha256(INFERENCE_PY.encode("utf-8")).hexdigest() == INFERENCE_PY_SHA
R["inference_py_sha256"] = INFERENCE_PY_SHA
SRV = Path("/tmp/srv")
SRV.mkdir(exist_ok=True)
(SRV / "inference.py").write_text(INFERENCE_PY)
sys.path.insert(0, str(SRV))
import inference as srv  # noqa: E402

# === DATA START ===
# ------------------------------------------------------------------- held-out rows, as in the compression kernel
t = time.time()
split_test = pd.read_csv(SPLIT_DIR / "test.csv")
aisd_n, rsna_n, soop = (pd.read_csv(CSV[k]) for k in ("aisd_normal", "rsna_normal", "soop"))
in_cols = ["prev_out_path", "next_out_path", "has_context", "dwi_out_path", "adc_out_path"]
keys = ["source_key", "out_path"]


def keep(df):
    return df[df["label"].isin(VALID_LABELS) & ~df["dataset"].isin(EXCLUDED_DATASETS)]


test_manifest = pd.concat([keep(split_test), aisd_n[aisd_n["split"] == "test"], rsna_n[rsna_n["split"] == "test"]],
                          ignore_index=True)
held = []
for saved, man in ((pd.read_csv(SAVED / "predictions_test.csv"), test_manifest),
                   (pd.read_csv(SAVED / "predictions_soop_external.csv"), soop[soop["split"] == "external"])):
    held.append(saved[keys].merge(man[keys + in_cols + ["dataset"]], on=keys, how="left", validate="one_to_one"))
held = pd.concat(held, ignore_index=True)
assert len(held) == len(PER_ROW) == 10602
assert (held["out_path"].to_numpy() == PER_ROW["out_path"].to_numpy()).all(), "row order differs from the saved run"
for c in ("label", "cohort", "pk"):
    held[c] = PER_ROW[c].to_numpy()
log(f"held-out rows rebuilt in the saved order: {len(held)} ({time.time() - t:.1f}s)")

# --------------------------------------------------------------------------------------- files by direct path
ZF, MEMBERS, LOCK, READS = {}, {}, threading.Lock(), {}
for pre, zp in ZIPS.items():
    t = time.time()
    ZF[pre] = zipfile.ZipFile(zp)
    MEMBERS[pre] = set(ZF[pre].namelist())
    log(f"zip {zp.name}: central directory read ({time.time() - t:.1f}s)")


def raw(p):
    p = str(p)
    with LOCK:
        READS[p.split("/")[0]] = READS.get(p.split("/")[0], 0) + 1
        for pre, zf in ZF.items():
            if p.startswith(pre):
                assert p in MEMBERS[pre], f"{p} not in {pre} zip"
                with zf.open(p) as f:
                    return f.read()
    name = Path(p).name
    if p.startswith("CPAISD_v2/"):
        cands = [CPAISD_V2_IMG / name]
    elif p.startswith("RSNA_ICH/"):
        cands = [RSNA_L / "hemorrhagic" / name, RSNA_J / "hemorrhagic" / name]
    elif p.startswith("AISD/"):
        cands = [AISD_K / "ischemic" / name]
    else:
        cands = [PHASE_B / p]
    return first(*cands, what=p).read_bytes()


def gray(p):
    return Image.open(io.BytesIO(raw(p))).convert("L").resize(TARGET_SIZE)


def merged(row):
    """Loader.merged, unchanged."""
    if row["dataset"] in MRI_PAIR_DATASETS:
        dwi, adc = gray(row["dwi_out_path"]), gray(row["adc_out_path"])
        return Image.merge("RGB", (dwi, adc, dwi))
    s = gray(row["out_path"])
    if bool(row["has_context"]):
        return Image.merge("RGB", (gray(row["prev_out_path"]), s, gray(row["next_out_path"])))
    return Image.merge("RGB", (s, s, s))


t = time.time()
X = np.empty((len(held), 224, 224, 3), dtype=np.uint8)
ROWS = [r for _, r in held.iterrows()]


def dec(i):
    X[i] = np.asarray(merged(ROWS[i]))


with ThreadPoolExecutor(8) as ex:
    list(ex.map(dec, range(len(ROWS))))
R["file_reads_by_source"] = dict(READS)
log(f"decoded {len(X)} rows ({time.time() - t:.0f}s, {X.nbytes / 1e9:.2f} GB); reads {READS}")

# === DATA END ===
N = len(held)
PS, CS = SERVED["probs"], SERVED["cams"]
assert (SERVED["rows"] == np.arange(N)).all() and len(PS) == N
PF, CF = PARTIAL["FP32"]["probs"], PARTIAL["FP32"]["cams"]
assert len(PF) == N
PK = held["pk"].to_numpy()
DATASET = held["dataset"].to_numpy()
LABEL = held["label"].to_numpy()
MRI = held["dataset"].isin(MRI_PAIR_DATASETS).to_numpy()


# ------------------------------------------------------------------------------------------------ inference engine
def rss_gb():
    try:
        with open("/proc/self/status") as f:
            for l in f:
                if l.startswith("VmRSS:"):
                    return int(l.split()[1]) / 1e6
    except OSError:
        pass
    return 0.0


def hwm_gb():
    try:
        with open("/proc/self/status") as f:
            for l in f:
                if l.startswith("VmHWM:"):
                    return int(l.split()[1]) / 1e6
    except OSError:
        pass
    return 0.0


def session(path, threads=1):
    o = ort.SessionOptions()
    o.intra_op_num_threads, o.inter_op_num_threads = threads, 1
    o.enable_cpu_mem_arena = False  # as on the server; the reference control checks that outputs are unaffected
    return ort.InferenceSession(str(path), sess_options=o, providers=["CPUExecutionProvider"])


def views(img):
    return np.stack([srv._to_array(v) for v in srv._tta_views(Image.fromarray(img))])


def run_jobs(path, jobs, n_out, tag, threads=1, workers=WORKERS, cap=True):
    """jobs: list of (positions, images). One batch per job: every image's four views, in order. The first len(positions)
    images are scored into probs/cams at those positions; the others only share the batch. Returns None if memory passed
    the cap."""
    probs, cams = np.zeros((n_out, 3)), np.zeros((n_out, 3, 7, 7), dtype=np.float32)
    local, t0, done, stop = threading.local(), time.time(), [0], threading.Event()

    def one(j):
        if stop.is_set():
            return
        if not hasattr(local, "s"):
            local.s = session(path, threads)
        pos, imgs = jobs[j]
        x = np.concatenate([views(im) for im in imgs])
        lg, cm = local.s.run(["logits", "gradcam"], {"image": x})
        p = srv._softmax(lg.astype(np.float64))
        for k, q in enumerate(pos):
            probs[q] = p[4 * k:4 * k + 4].mean(0)
            cams[q] = cm[4 * k]  # the map comes from the original view, as on the server
        with LOCK:
            done[0] += 1
            d = done[0]
        if d % 200 == 0 or d == len(jobs):
            el = time.time() - t0
            r = rss_gb()
            if d % 1000 == 0 or d == len(jobs):
                log(f"  [{tag}] {d}/{len(jobs)} batches ({el:.0f}s; projected {el / d * len(jobs) / 60:.0f} min; RSS {r:.1f} GB)")
            if cap and r > MAX_RSS_GB:
                log(f"  [{tag}] RSS {r:.1f} GB above the {MAX_RSS_GB} GB cap: condition stopped")
                stop.set()

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, range(len(jobs))))
    return None if stop.is_set() else (probs, cams)


def jobs_alone(rows):
    return [([k], [X[i]]) for k, i in enumerate(rows)]


def jobs_chunks(rows, b):
    return [(list(range(a, min(a + b, len(rows)))), [X[rows[q]] for q in range(a, min(a + b, len(rows)))])
            for a in range(0, len(rows), b)]


def views_alone(path, rows, tag):
    """Each of a slice's four views in a batch of its own, then the softmax outputs averaged."""
    probs, cams = np.zeros((len(rows), 3)), np.zeros((len(rows), 3, 7, 7), dtype=np.float32)
    local, t0, done = threading.local(), time.time(), [0]

    def one(j):
        if not hasattr(local, "s"):
            local.s = session(path)
        ps = []
        for v, x in enumerate(views(X[rows[j]])):
            lg, cm = local.s.run(["logits", "gradcam"], {"image": x[None]})
            ps.append(srv._softmax(lg.astype(np.float64))[0])
            if v == 0:
                cams[j] = cm[0]
        probs[j] = np.mean(ps, axis=0)
        with LOCK:
            done[0] += 1
            d = done[0]
        if d % 1000 == 0 or d == len(rows):
            el = time.time() - t0
            log(f"  [{tag}] {d}/{len(rows)} ({el:.0f}s; projected {el / d * len(rows) / 60:.0f} min)")

    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(one, range(len(rows))))
    return probs, cams


def diff(p, c, pref, cref):
    return float(np.abs(p - pref).max()), float(np.abs(c - cref).max() / (np.abs(cref).max() + 1e-12))


def shuffled_jobs(b, seed):
    """All slices in random batches of b. A short final batch is filled with unscored random slices."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    jobs, dup = [], 0
    for a in range(0, N, b):
        mem = perm[a:a + b]
        imgs = [X[i] for i in mem]
        if len(mem) < b:
            fill = rng.choice(np.setdiff1d(perm, mem), b - len(mem), replace=False)
            imgs += [X[i] for i in fill]
        jobs.append(([int(i) for i in mem], imgs))
        dup += int(len(set(PK[mem])) < len(mem))
    return jobs, dup / len(jobs)


BLACK = np.zeros((224, 224, 3), dtype=np.uint8)
WHITE = np.full((224, 224, 3), 255, dtype=np.uint8)
R["rss_after_decode_gb"] = rss_gb()

# ------------------------------------------------------------------------------------------------ Stage 2 selection
margin = np.sort(PF, axis=1)
margin = margin[:, 2] - margin[:, 1]
rng_t = np.random.default_rng(123)
near = np.where(margin < 0.1)[0]
others = []
for key, g in held.assign(_near=margin < 0.1)[margin >= 0.1].groupby(["dataset", "label"]):
    others.extend(rng_t.choice(g.index.to_numpy(), 15, replace=False).tolist())
TARGETS = np.concatenate([near, np.array(sorted(others), dtype=int)])
IS_NEAR = np.array([True] * len(near) + [False] * len(others))
MEAN_I = X.reshape(N, -1).mean(1)
GROUPS = {"CT_normal": ~MRI & (LABEL == "normal"), "CT_ischemic": ~MRI & (LABEL == "ischemic"),
          "CT_hemorrhagic": ~MRI & (LABEL == "hemorrhagic"), "MRI_normal": MRI & (LABEL == "normal"),
          "MRI_ischemic": MRI & (LABEL == "ischemic")}
RANKED = {}
for name, mask in GROUPS.items():
    g = np.where(mask)[0]
    RANKED[f"{name}_median"] = g[np.argsort(np.abs(MEAN_I[g] - np.median(MEAN_I[g])), kind="stable")]
    RANKED[f"{name}_highest"] = g[np.argsort(-MEAN_I[g], kind="stable")]
PARTNERS = list(RANKED) + ["stress_black", "stress_white"]
replaced = [0]


def partner_image(t, name):
    if name == "stress_black":
        return BLACK
    if name == "stress_white":
        return WHITE
    for rank, i in enumerate(RANKED[name]):
        if PK[i] != PK[t]:
            replaced[0] += int(rank > 0)
            return X[i]
    raise RuntimeError(f"no partner for {t} in {name}")


def partner_jobs(targets):
    return [([a * len(PARTNERS) + b], [X[t], partner_image(t, name)])
            for a, t in enumerate(targets) for b, name in enumerate(PARTNERS)]


R["stage2_selection"] = {"targets": int(len(TARGETS)), "near_ties": int(len(near)), "others": int(len(others)),
                         "partners": {k: {"rank0_out_path": str(held["out_path"].iloc[int(v[0])]),
                                          "rank0_mean_intensity": float(MEAN_I[int(v[0])])} for k, v in RANKED.items()}}
log(f"stage 2: {len(TARGETS)} targets ({len(near)} near-ties, {len(others)} others), {len(PARTNERS)} partners")
save()

# ------------------------------------------------------------------------------------------------ Stage 0 controls
D = MODELS["INT8-dyn"]
idx = np.arange(N_CHECK)
ctl = R["controls"] = {}


def check(name, res, pmax, cmax):
    ctl[name] = {"max_prob_diff": res[0], "max_rel_map_diff": res[1], "limits": [pmax, cmax]}
    log(f"control {name}: max prob diff {res[0]:.2e}, map {res[1]:.2e}")
    save()
    if res[0] > pmax or res[1] > cmax:
        print(f"[ERROR] control {name} failed; stopping")
        sys.exit(1)


t = time.time()
rows = idx.tolist()
p, c = run_jobs(D, jobs_alone(rows), N_CHECK, "reference")
check("reference_vs_saved_served", diff(p, c, PS[idx], CS[idx]), 1e-5, 1e-3)
pa, ca = run_jobs(D, jobs_chunks(rows, 8), N_CHECK, "repeat 1")
pb, cb = run_jobs(D, jobs_chunks(rows, 8), N_CHECK, "repeat 2")
check("repeatability_same_batches", diff(pa, ca, pb, cb), 1e-5, 1e-3)
few = rows[:64]
p1, c1 = run_jobs(D, jobs_chunks(few, 4), 64, "1 thread", threads=1, workers=1)
p4, c4 = run_jobs(D, jobs_chunks(few, 4), 64, "4 threads", threads=4, workers=1)
check("thread_count_1_vs_4", diff(p1, c1, p4, c4), 1e-5, 1e-3)
for k in ("FP32", "INT8-static"):
    pr, cr = PARTIAL[k]["probs"][idx], PARTIAL[k]["cams"][idx]
    p, c = run_jobs(MODELS[k], jobs_chunks(rows, 8), N_CHECK, f"{k} batches of 8")
    check(f"{k}_batches_of_8", diff(p, c, pr, cr), 1e-4, 1e-3)
    p, c = run_jobs(MODELS[k], [([q], [X[q], WHITE]) for q in rows], N_CHECK, f"{k} white partner")
    check(f"{k}_white_partner", diff(p, c, pr, cr), 1e-4, 1e-3)
# the effect must exist at all for the rest to mean anything: logged, not a stop condition
ctl["INT8-dyn_batches_of_8_vs_served"] = dict(zip(("max_prob_diff", "max_rel_map_diff"), diff(pa, ca, PS[idx], CS[idx])))
R["timing_min"]["controls"] = round((time.time() - t) / 60, 1)
log(f"controls passed ({R['timing_min']['controls']} min); {json.dumps(ctl['INT8-dyn_batches_of_8_vs_served'])}")
save()

# ------------------------------------------------------------------------------------------------ dry run and projection
dry = R["dry_run"] = {}
rng_d = np.random.default_rng(321)
perm_d = rng_d.permutation(N)
t = time.time()
views_alone(D, perm_d[:64].tolist(), "dry views alone")
dry["views_alone_s_per_slice"] = (time.time() - t) / 64
for b in sorted(SIZES):
    n = max(64, 4 * b)
    t = time.time()
    if run_jobs(D, jobs_chunks(perm_d[:n].tolist(), b), n, f"dry b={b}") is None:
        dry[f"b{b}_memory_cap"] = True
    dry[f"b{b}_s_per_slice"] = (time.time() - t) / n
t = time.time()
run_jobs(D, partner_jobs(TARGETS[:16]), 16 * len(PARTNERS), "dry partners")
dry["partner_s_per_pair"] = (time.time() - t) / (16 * len(PARTNERS))
proj = (dry["views_alone_s_per_slice"] * N + sum(dry[f"b{b}_s_per_slice"] for b in SIZES) * N
        + dry["b16_s_per_slice"] * N + dry["partner_s_per_pair"] * len(TARGETS) * len(PARTNERS)) / 60 + 30
dry["projected_total_min"] = proj + (time.time() - T0) / 60
dry["hwm_gb"] = hwm_gb()
log(f"dry run: {json.dumps(dry)}")
save()
if dry["projected_total_min"] > LIMIT_PROJECTED_MIN:
    print(f"[ERROR] projected {dry['projected_total_min']:.0f} min is above {LIMIT_PROJECTED_MIN}; stopping before the full run")
    sys.exit(1)

# ------------------------------------------------------------------------------------------------ the conditions
PART = OUT / "partial"
PART.mkdir(exist_ok=True)
RESULT = {"served_four_views_together": (PS, CS), "file_order_batches_of_16": (PARTIAL["INT8-dyn"]["probs"], PARTIAL["INT8-dyn"]["cams"])}


def out_of_time(name):
    if (time.time() - T0) / 3600 > LIMIT_START_H:
        R["skipped"].append(name)
        log(f"{name}: not started, past {LIMIT_START_H} h")
        save()
        return True
    return False


def keep_result(name, res, minutes):
    R["timing_min"][name] = round(minutes, 1)
    if res is None:
        R["skipped"].append(name)
    else:
        RESULT[name] = res
        np.savez_compressed(PART / f"{name}.npz", probs=res[0], cams=res[1])
    log(f"{name}: {'done' if res is not None else 'NOT RUN (memory cap)'} in {minutes:.1f} min; peak RSS {hwm_gb():.1f} GB")
    save()


# Stage 1
if not out_of_time("views_alone"):
    t = time.time()
    keep_result("views_alone", views_alone(D, list(range(N)), "views alone"), (time.time() - t) / 60)

# Stage 2
S2 = None
replaced[0] = 0  # the dry run also chose partners
if not out_of_time("partner_stage"):
    t = time.time()
    jobs = partner_jobs(TARGETS)
    S2 = run_jobs(D, jobs, len(TARGETS) * len(PARTNERS), "partners")
    R["stage2_selection"]["partner_replaced_for_same_patient"] = replaced[0]
    R["timing_min"]["partner_stage"] = round((time.time() - t) / 60, 1)
    if S2 is None:
        R["skipped"].append("partner_stage")
    else:
        np.savez_compressed(PART / "partner_stage.npz", probs=S2[0], cams=S2[1], targets=TARGETS, is_near=IS_NEAR, partners=np.array(PARTNERS))
    save()

# Stage 3
R["same_patient_batch_share"] = {}
for b, seed in [(s, 123) for s in SIZES] + [(16, 124)]:
    name = f"random_b{b}_seed{seed}"
    if out_of_time(name):
        continue
    t = time.time()
    jobs, share = shuffled_jobs(b, seed)
    R["same_patient_batch_share"][name] = share
    keep_result(name, run_jobs(D, jobs, N, name), (time.time() - t) / 60)


# ------------------------------------------------------------------- E2 agreement (compression kernel, unchanged)
def decide(p):
    return p.argmax(1)


def tier(p, mri):
    i = int(p.argmax())
    if mri and i == 2:
        return "not_validated"
    c = srv.ThreeClassPipeline._risk_tier_color(VALID_LABELS[i], *map(float, p))
    return {(192, 57, 43): "red", (230, 126, 34): "orange", (127, 140, 141): "grey", (41, 128, 185): "blue"}[c]


def auroc(yb, s):
    n1 = int(yb.sum())
    n0 = len(yb) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return float((r[yb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def slice_metrics(y, pred, p):
    r = {"n": int(len(y)), "accuracy": float((pred == y).mean())}
    present = [c for c in range(3) if (y == c).any()]
    cm = np.bincount(y * 3 + pred, minlength=9).reshape(3, 3)
    r["confusion_matrix_rows_true_cols_pred"] = cm.tolist()
    recs, f1s, aucs = [], [], []
    for c in range(3):
        tp, fn, fp = cm[c, c], cm[c].sum() - cm[c, c], cm[:, c].sum() - cm[c, c]
        if tp + fn:
            r[f"{VALID_LABELS[c]}_recall"] = tp / (tp + fn)
        if tp + fp:
            r[f"{VALID_LABELS[c]}_precision"] = tp / (tp + fp)
        if c in present:
            recs.append(tp / (tp + fn))
            f1s.append(2 * tp / (2 * tp + fp + fn))
            a = auroc((y == c).astype(int), p[:, c])
            if a == a:
                aucs.append(a)
            if c > 0:
                r[f"{VALID_LABELS[c]}_auroc"] = a
    r["balanced_accuracy"] = float(np.mean(recs))
    r["macro_f1_present_classes"] = float(np.mean(f1s))
    if len(aucs) >= 2:
        r["macro_ovr_auroc"] = float(np.mean(aucs))
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r.items()}


def clusters(df):
    d0 = df.reset_index(drop=True)
    return [[g.index.to_numpy()[ix] for ix in g.groupby("pk").indices.values()] for _, g in d0.groupby("dataset")]


def boot_idx(groups, rng):
    return np.concatenate([np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))]) for cl in groups])


def compare(df, pa, pb, seed):
    """a = FP32, b = variant, deployed rule; patient-cluster bootstrap (2,000) for agreement and metric deltas."""
    y = df["label"].map(LABEL_TO_IDX).to_numpy()
    da, db = decide(pa), decide(pb)
    mri = df["dataset"].isin(MRI_PAIR_DATASETS).to_numpy()
    ta = np.array([tier(p, m) for p, m in zip(pa, mri)])
    tb = np.array([tier(p, m) for p, m in zip(pb, mri)])
    agree, tagree = (da == db).astype(float), (ta == tb).astype(float)
    ma, mb = slice_metrics(y, da, pa), slice_metrics(y, db, pb)
    groups, rng = clusters(df), np.random.default_rng(seed)
    bs = {"agreement": [], "tier_agreement": [], **{k: [] for k in KEYS}}
    for _ in range(N_BOOT):
        ix = boot_idx(groups, rng)
        bs["agreement"].append(agree[ix].mean())
        bs["tier_agreement"].append(tagree[ix].mean())
        a, b = slice_metrics(y[ix], da[ix], pa[ix]), slice_metrics(y[ix], db[ix], pb[ix])
        for k in KEYS:
            if k in a and k in b and a[k] == a[k] and b[k] == b[k]:
                bs[k].append(b[k] - a[k])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    dp = np.abs(pa - pb)
    out = {"n": int(len(df)), "patients": int(df["pk"].nunique()),
           "argmax_agreement": float(agree.mean()), "argmax_agreement_ci95": ci(bs["agreement"]),
           "flips": int((da != db).sum()), "flip_table_rows_fp32_cols_variant": np.bincount(da * 3 + db, minlength=9).reshape(3, 3).tolist(),
           "tier_agreement": float(tagree.mean()), "tier_agreement_ci95": ci(bs["tier_agreement"]),
           "tier_changes": pd.Series([f"{a}->{b}" for a, b in zip(ta, tb) if a != b]).value_counts().to_dict(),
           "prob_abs_diff_max": float(dp.max()), "prob_abs_diff_mean": float(dp.mean()),
           "prob_abs_diff_p99": float(np.percentile(dp.max(1), 99)), "secondary_metric_deltas_variant_minus_fp32": {}}
    for k in KEYS:
        if k in ma and k in mb and len(bs[k]) > 50:
            out["secondary_metric_deltas_variant_minus_fp32"][k] = {"fp32": ma[k], "variant": mb[k], "diff": mb[k] - ma[k],
                                                                    "ci95": ci(bs[k])}
    out["by_source"] = {ds: {"n": int(len(g)), "argmax_agreement": float(agree[g.index.to_numpy()].mean()),
                             "flips": int((da != db)[g.index.to_numpy()].sum())}
                        for ds, g in df.reset_index(drop=True).groupby("dataset")}
    return out

COHORTS = {"pooled_internal_test": held["cohort"] != "SOOP", "internal_test": held["cohort"] == "internal_test",
           "CPAISD": held["cohort"] == "CPAISD", "SOOP_holdout": held["cohort"] == "SOOP"}


# ------------------------------------------------------------------------------------------------ scoring
def display_region(cam7):
    return np.asarray(srv.ThreeClassPipeline._display_cam(cam7), dtype=np.float32) / 255.0 > 0.6


Y = held["label"].map(LABEL_TO_IDX).to_numpy()
FP32_CLS = PF.argmax(1)
SERVED_CLS = PS.argmax(1)
SERVED_TIER = np.array([tier(q, m) for q, m in zip(PS, MRI)])
q_ = lambda s: {"median": float(np.nanmedian(s)), "q1": float(np.nanpercentile(s, 25)), "q3": float(np.nanpercentile(s, 75))}


def directions(a, b):
    ch = a != b
    return {"to_normal": int((ch & (b == 0)).sum()), "from_normal": int((ch & (a == 0)).sum()),
            "between_stroke_classes": int((ch & (a != 0) & (b != 0)).sum())}


def detection(pred):
    out = {}
    isch = held["label"].to_numpy() == "ischemic"
    for cname, sel in (("pooled_internal_test", held["cohort"].to_numpy() != "SOOP"), ("SOOP_holdout", held["cohort"].to_numpy() == "SOOP")):
        ix = np.where(isch & sel)[0]
        g = pd.DataFrame({"pk": PK[ix], "dataset": DATASET[ix], "hit": pred[ix] == 1}).groupby(["dataset", "pk"])["hit"].any()
        out[cname] = {"ischemic_patients": int(len(g)), "detected": int(g.sum())}
    return out


def map_measures(rows, cams, ref_cams, cls):
    a = cams.reshape(len(rows), -1)
    r = ref_cams.reshape(len(rows), -1)
    rel = np.abs(a - r).max(1) / (np.abs(r).max(1) + 1e-12)
    peak, iou = [], []
    for j in range(len(rows)):
        x_, y_ = ref_cams[j, cls[j]], cams[j, cls[j]]
        peak.append(int(x_.argmax() == y_.argmax()))
        ra, rb = display_region(x_), display_region(y_)
        un = (ra | rb).sum()
        iou.append(float((ra & rb).sum() / un) if un else np.nan)
    iou = np.array(iou)
    return {"map_rel_diff": {"max": float(rel.max()), "p99": float(np.percentile(rel, 99)), "median": float(np.median(rel))},
            "same_peak_cell": float(np.mean(peak)), "display_iou": q_(iou), "iou_below_0.5": float(np.nanmean(iou < 0.5))}


R["conditions"] = {}
changed = []
ALL = np.arange(N)
for name, (P, C) in RESULT.items():
    t = time.time()
    cls = P.argmax(1)
    ch = cls != SERVED_CLS
    dp = np.abs(P - PS).max(1)
    tiers = np.array([tier(q, m) for q, m in zip(P, MRI)])
    res = {"vs_served": {"predictions_differ": int(ch.sum()), **directions(SERVED_CLS, cls),
                         "fp32_margin_of_differing": {"median": float(np.median(margin[ch])) if ch.any() else None,
                                                      "max": float(margin[ch].max()) if ch.any() else None},
                         "near_tie_share": float((margin[ch] < 0.1).mean()) if ch.any() else None,
                         "patients_with_a_difference": int(len(set(PK[ch]))),
                         "tier_changes": int((tiers != SERVED_TIER).sum()),
                         "prob_diff": {"max": float(dp.max()), "p99": float(np.percentile(dp, 99)), "median": float(np.median(dp))},
                         **map_measures(ALL, C, CS, FP32_CLS)},
           "vs_fp32": {"changed": int((cls != FP32_CLS).sum()), **directions(FP32_CLS, cls),
                       "correct_to_wrong": int(((FP32_CLS == Y) & (cls != Y)).sum()),
                       "wrong_to_correct": int(((FP32_CLS != Y) & (cls == Y)).sum()),
                       **{f"maps_{k}": v for k, v in map_measures(ALL, C, CF, FP32_CLS).items()}},
           "ischemic_detection": detection(cls), "agreement_with_fp32": {}}
    for seed, (cname, mask) in enumerate(COHORTS.items(), start=901):
        sel = np.where(mask.to_numpy())[0]
        c_ = compare(held.iloc[sel].reset_index(drop=True), PF[sel], P[sel], seed)
        res["agreement_with_fp32"][cname] = {k: c_[k] for k in ("argmax_agreement", "argmax_agreement_ci95", "flips", "tier_agreement", "tier_agreement_ci95", "prob_abs_diff_max")}
    R["conditions"][name] = res
    for i in np.where(ch)[0]:
        changed.append({"condition": name, "held_i": int(i), "out_path": held["out_path"].iloc[int(i)], "dataset": DATASET[i], "label": LABEL[i],
                        "pk": PK[i], "fp32": VALID_LABELS[FP32_CLS[i]], "served": VALID_LABELS[SERVED_CLS[i]], "condition_pred": VALID_LABELS[cls[i]],
                        "fp32_margin": float(margin[i]), "prob_diff": float(dp[i])})
    log(f"{name}: differs from served on {res['vs_served']['predictions_differ']}, {json.dumps(res['agreement_with_fp32']['pooled_internal_test'])[:150]} ({time.time() - t:.0f}s)")
    save()
pd.DataFrame(changed).to_csv(OUT / "predictions_differing_from_served.csv", index=False)

base = held[["out_path", "dataset", "label", "cohort", "pk"]].copy()
for name, (P, _) in RESULT.items():
    for c_i, lab in enumerate(VALID_LABELS):
        base[f"p_{lab}_{name}"] = P[:, c_i].round(6)
base.to_csv(OUT / "per_row_probs.csv", index=False)

# the reference must reproduce the paper's numbers: logged prominently, the file is the served run's own
ref = R["conditions"]["served_four_views_together"]
log(f"reference check: changed vs FP32 {ref['vs_fp32']['changed']} (paper 32), detection {ref['ischemic_detection']} (paper 263/281, 63/66), "
    f"pooled agreement {ref['agreement_with_fp32']['pooled_internal_test']['argmax_agreement']:.4f} (paper 0.9971)")

# Stage 2 scoring
if S2 is not None:
    Pp, Cc = S2
    rec = []
    for a, ti in enumerate(TARGETS):
        for b, pname in enumerate(PARTNERS):
            q = a * len(PARTNERS) + b
            pred = int(Pp[q].argmax())
            rel_ = float(np.abs(Cc[q] - CS[ti]).max() / (np.abs(CS[ti]).max() + 1e-12))
            rec.append({"target_i": int(ti), "near_tie": bool(IS_NEAR[a]), "dataset": DATASET[ti], "label": LABEL[ti], "fp32_margin": float(margin[ti]),
                        "partner": pname, "stress": pname.startswith("stress"), "served_pred": int(SERVED_CLS[ti]), "pred": pred,
                        "differs": pred != int(SERVED_CLS[ti]), "to_normal": pred == 0 and int(SERVED_CLS[ti]) != 0,
                        "from_normal": pred != 0 and int(SERVED_CLS[ti]) == 0, "prob_diff": float(np.abs(Pp[q] - PS[ti]).max()), "map_rel_diff": rel_,
                        "same_peak": int(Cc[q, FP32_CLS[ti]].argmax() == CS[ti, FP32_CLS[ti]].argmax())})
    d = pd.DataFrame(rec)
    d.to_csv(OUT / "partner_stage.csv", index=False)
    sm = {}
    for lab, sub in (("near_ties_realistic", d[d["near_tie"] & ~d["stress"]]), ("others_realistic", d[~d["near_tie"] & ~d["stress"]]),
                     ("near_ties_stress", d[d["near_tie"] & d["stress"]]), ("others_stress", d[~d["near_tie"] & d["stress"]])):
        per_t = sub.groupby("target_i")
        sm[lab] = {"targets": int(per_t.ngroups), "pairs": int(len(sub)), "pairs_differing": int(sub["differs"].sum()),
                   "targets_flipping_for_at_least_one_partner": int(per_t["differs"].any().sum()),
                   "to_normal": int(sub["to_normal"].sum()), "from_normal": int(sub["from_normal"].sum()),
                   "prob_diff": {"max": float(sub["prob_diff"].max()), "p99": float(sub["prob_diff"].quantile(.99)), "median": float(sub["prob_diff"].median())},
                   "range_across_partners_max_prob_diff": q_(per_t["prob_diff"].max().to_numpy()),
                   "map_rel_diff": {"max": float(sub["map_rel_diff"].max()), "median": float(sub["map_rel_diff"].median())},
                   "same_peak": float(sub["same_peak"].mean())}
    sm["by_partner"] = {k: {"pairs_differing": int(g["differs"].sum()), "median_prob_diff": float(g["prob_diff"].median()),
                            "max_prob_diff": float(g["prob_diff"].max()), "median_map_rel_diff": float(g["map_rel_diff"].median())}
                        for k, g in d.groupby("partner")}
    R["partner_stage"] = sm
    log(f"partner stage: {json.dumps({k: v for k, v in sm.items() if k != 'by_partner'})[:1500]}")
    save()
log(f"Done. Peak RSS {hwm_gb():.1f} GB; skipped {R['skipped']}")
save()
