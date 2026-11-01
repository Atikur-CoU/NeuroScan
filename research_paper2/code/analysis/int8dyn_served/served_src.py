"""
NeuroScan paper 2 -- INT8-dynamic as served (2026-10-31). CPU only; no training; no new classification result.

Why: ONNX Runtime's dynamic quantization chooses each activation scale from the whole input tensor, so INT8-dynamic's
outputs depend on what else is in the batch. Kernel neuroscan-p2-compression-gradcam (v4) ran every copy in batches
of 16 slices (64 views); the server runs one request at a time, i.e. one slice's 4 TTA views. Kernel
neuroscan-p2-fig4-cases (v3) found per-request INT8-dynamic probabilities up to 0.014 away from the batched ones and
one prediction that differs. FP32 and INT8-static (fixed scales) matched exactly.

What: INT8-dynamic (the same file, from neuroscan-p2-compression-gradcam's output) is re-run on all 10,602 held-out
slices exactly as served: one slice per run, its 4 views as the batch, 4 worker threads with one session each
(1 intra-op thread; the batch, not the thread count, sets the scales). Everything downstream is the original code:
  - E2 agreement with FP32 under the deployed rule (plain argmax of the 4-view mean), patient-cluster bootstrap, alert
    tiers, by cohort and source: compare() of the compression kernel, unchanged (seeds 901-904);
  - E3 maps under compression (Pearson r, same peak cell, display-region IoU, outline changes): unchanged;
  - the fidelity follow-up (transitions, changed slices, patient-level detection): fidelity.py of
    neuroscan-p2-fidelity-detail, verbatim except its input and output paths.
FP32 and INT8-static are batch-independent; this is checked on the first 256 rows (per request vs the saved run, must
match), and their saved outputs are used. The batched INT8-dynamic outputs are compared with the served ones.
Held-out rows are rebuilt as in the compression kernel; files by direct path (no listing, no zip extraction).
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
KEYS = ["accuracy", "balanced_accuracy", "macro_f1_present_classes", "normal_recall", "ischemic_recall", "ischemic_precision",
        "hemorrhagic_recall", "hemorrhagic_precision", "ischemic_auroc", "hemorrhagic_auroc", "macro_ovr_auroc"]
OUT = Path("/kaggle/working/output")
OUT.mkdir(parents=True, exist_ok=True)
R = {"note": __doc__.strip().splitlines()[0], "onnxruntime": ort.__version__}


def save():
    (OUT / "served_results.json").write_text(json.dumps(R, indent=2, default=float))


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
R["int8_dynamic_sha256"] = sha256(MODELS["INT8-dyn"])
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


# ------------------------------------------------------------------------------------- per-request inference
def session(path):
    o = ort.SessionOptions()
    o.intra_op_num_threads, o.inter_op_num_threads = 1, 1
    return ort.InferenceSession(str(path), sess_options=o, providers=["CPUExecutionProvider"])


def per_request(path, idx, tag):
    """One run per slice, its 4 views as the batch: the server's request (inference.predict)."""
    probs, cams = np.zeros((len(idx), 3)), np.zeros((len(idx), 3, 7, 7), dtype=np.float32)
    local, t0, done = threading.local(), time.time(), [0]

    def one(j):
        if not hasattr(local, "s"):
            local.s = session(path)
        x = np.stack([srv._to_array(v) for v in srv._tta_views(Image.fromarray(X[idx[j]]))])
        lg, cm = local.s.run(["logits", "gradcam"], {"image": x})
        probs[j] = srv._softmax(lg.astype(np.float64)).mean(0)
        cams[j] = cm[0]
        with LOCK:
            done[0] += 1
            if done[0] % 1000 == 0 or done[0] == len(idx):
                el = time.time() - t0
                log(f"  [{tag}] {done[0]}/{len(idx)} ({el:.0f}s; projected {el / done[0] * len(idx) / 60:.0f} min)")

    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(one, range(len(idx))))
    return probs, cams


R["batch_independence_check"] = {}
for k in ("FP32", "INT8-static"):
    idx = np.arange(N_CHECK)
    p, c = per_request(MODELS[k], idx, f"{k} check")
    dp = float(np.abs(p - PARTIAL[k]["probs"][idx]).max())
    dc = float(np.abs(c - PARTIAL[k]["cams"][idx]).max() / np.abs(PARTIAL[k]["cams"][idx]).max())
    R["batch_independence_check"][k] = {"rows": N_CHECK, "max_prob_diff": dp, "max_rel_map_diff": dc}
    log(f"{k}: per request vs saved batched run on {N_CHECK} rows: max prob diff {dp:.2e}, map {dc:.2e}")
    if dp > 1e-4 or dc > 1e-3:
        print(f"[ERROR] {k} is not batch-independent; the saved {k} outputs cannot stand for the served ones")
        sys.exit(1)
save()

t = time.time()
ALL = np.arange(len(held))
PS, CS = per_request(MODELS["INT8-dyn"], ALL, "INT8-dyn served")
R["int8_dyn_served_minutes"] = round((time.time() - t) / 60, 1)
np.savez_compressed(OUT / "INT8-dyn_served.npz", probs=PS, cams=CS, rows=ALL)

# ---------------------------------------------------------------------------------- batched vs served
PB, CB = PARTIAL["INT8-dyn"]["probs"], PARTIAL["INT8-dyn"]["cams"]
PF, CF = PARTIAL["FP32"]["probs"], PARTIAL["FP32"]["cams"]
dpb = np.abs(PS - PB).max(1)
rel = np.abs(CS - CB).reshape(len(ALL), -1).max(1) / (np.abs(CB).reshape(len(ALL), -1).max(1) + 1e-12)
flip_bs = PS.argmax(1) != PB.argmax(1)
R["batched_vs_served"] = {
    "prob_abs_diff": {"max": float(dpb.max()), "median": float(np.median(dpb)), "p99": float(np.percentile(dpb, 99))},
    "map_rel_diff": {"max": float(rel.max()), "median": float(np.median(rel)), "p99": float(np.percentile(rel, 99))},
    "predictions_differ": int(flip_bs.sum()),
    "changed_vs_fp32_batched": int((PB.argmax(1) != PF.argmax(1)).sum()),
    "changed_vs_fp32_served": int((PS.argmax(1) != PF.argmax(1)).sum()),
    "rows_where_batched_and_served_differ": held.loc[flip_bs, ["out_path", "dataset", "label"]].assign(
        fp32=[VALID_LABELS[i] for i in PF.argmax(1)[flip_bs]], batched=[VALID_LABELS[i] for i in PB.argmax(1)[flip_bs]],
        served=[VALID_LABELS[i] for i in PS.argmax(1)[flip_bs]]).to_dict("records")}
log(f"batched vs served: {json.dumps({k: v for k, v in R['batched_vs_served'].items() if k != 'rows_where_batched_and_served_differ'})}")
save()

per_row = PER_ROW.copy()
for c, lab in enumerate(VALID_LABELS):
    per_row[f"p_{lab}_INT8-dyn"] = PS[:, c].round(6)
per_row.to_csv(OUT / "e2_per_row_probs.csv", index=False)   # INT8-dyn columns = served; FP32/INT8-static = saved


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
R["E2_agreement_served"] = {}
for seed, (cname, mask) in enumerate(COHORTS.items(), start=901):
    sel = np.where(mask.to_numpy())[0]
    res = compare(held.iloc[sel].reset_index(drop=True), PF[sel], PS[sel], seed)
    R["E2_agreement_served"][cname] = res
    log(f"E2 INT8-dyn served {cname}: n={res['n']} agreement {res['argmax_agreement']:.4f} {res['argmax_agreement_ci95']} "
        f"flips {res['flips']} tier {res['tier_agreement']:.4f} max|dp| {res['prob_abs_diff_max']:.4f}")
    save()


# ------------------------------------------------------------- E3 maps under compression (unchanged), served
def display_region(cam7):
    return np.asarray(srv.ThreeClassPipeline._display_cam(cam7), dtype=np.float32) / 255.0 > 0.6


cls_a, cls_b = PF.argmax(1), PS.argmax(1)
mri = held["dataset"].isin(MRI_PAIR_DATASETS).to_numpy()
rec = []
for i in range(len(ALL)):
    a, b = CF[i, cls_a[i]].ravel(), CS[i, cls_a[i]].ravel()
    r_ = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
    ra, rb = display_region(CF[i, cls_a[i]]), display_region(CS[i, cls_a[i]])
    un = (ra | rb).sum()
    out_a = VALID_LABELS[cls_a[i]] in ("ischemic", "hemorrhagic") and not (mri[i] and cls_a[i] == 2)
    out_b = VALID_LABELS[cls_b[i]] in ("ischemic", "hemorrhagic") and not (mri[i] and cls_b[i] == 2)
    rec.append({"held_i": i, "variant": "INT8-dyn-served", "pearson_r": r_, "same_peak_cell": int(a.argmax() == b.argmax()),
                "display_iou": float((ra & rb).sum() / un) if un else np.nan,
                "outline_fp32": int(out_a), "outline_variant": int(out_b)})
d = pd.DataFrame(rec)
d["cohort"] = held["cohort"].to_numpy()
d.to_csv(OUT / "e3_maps_summary_served.csv", index=False)
q = lambda s: {"median": float(s.median()), "q1": float(s.quantile(.25)), "q3": float(s.quantile(.75)),
               "mean": float(s.mean()), "n": int(s.notna().sum())}
R["E3_maps_served"] = {c: {"pearson_r": q(g["pearson_r"]), "same_peak_cell": float(g["same_peak_cell"].mean()),
                           "display_iou": q(g["display_iou"]),
                           "outline_appears": int(((g["outline_fp32"] == 0) & (g["outline_variant"] == 1)).sum()),
                           "outline_disappears": int(((g["outline_fp32"] == 1) & (g["outline_variant"] == 0)).sum())}
                       for c, g in [("all", d), *d.groupby("cohort")]}
log(f"E3 maps INT8-dyn served: {json.dumps(R['E3_maps_served']['all'])}")
save()

# --------------------------------------------- fidelity follow-up (neuroscan-p2-fidelity-detail, verbatim code)
FIDELITY_PY = "__FIDELITY_PY__"
fd = OUT / "fidelity_served"
fd.mkdir(exist_ok=True)
src = FIDELITY_PY
assert src.count("df = pd.read_csv(SRC)") == 1 and src.count('OUT = Path("/kaggle/working/output")') == 1
src = src.replace('OUT = Path("/kaggle/working/output")', f'OUT = Path("{fd}")')
src = src.replace("df = pd.read_csv(SRC)", f'SRC = Path("{OUT / "e2_per_row_probs.csv"}")\ndf = pd.read_csv(SRC)')
(Path("/tmp") / "fidelity_served.py").write_text(src)
t = time.time()
r = subprocess.run([sys.executable, "/tmp/fidelity_served.py"], capture_output=True, text=True)
print(r.stdout[-6000:], r.stderr[-2000:], flush=True)
log(f"fidelity follow-up on the served CSV: exit {r.returncode} ({time.time() - t:.0f}s)")
R["fidelity_served_exit"] = r.returncode
save()
log("Done.")
