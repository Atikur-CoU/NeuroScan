"""
NeuroScan paper 2 -- K1, JTCSST revision (R2-1): does INT8-dynamic drift more when a CT request carries one slice
instead of three? (2026-11-20). CPU only; no training; no new classification result.

Why: the server accepts a CT slice without its neighbours and then runs the single-image pathway (the slice copied
into all three channels; the response carries a warning, because this is not the evaluated pathway). The paper's
fidelity results cover the evaluated pathways only.

What: every held-out CT slice (pooled internal test and CPAISD; the MRI cohorts have no single-image pathway, since MRI
without ADC is refused) is rebuilt as the single-image request -- channels (slice, slice, slice), as inference.build_input
does -- and run exactly as served: one slice per run, its 4 TTA views as the batch, 4 worker threads with one session
each, for FP32 and INT8-dynamic (the same files as the paper, SHA-256 checked).
  1. INT8-dynamic vs FP32 on the single-image pathway: agreement with the patient-cluster bootstrap (compare() of the
     compression kernel, unchanged), alert tiers, by source; every changed slice with truth, direction and FP32 margin;
     maps (Pearson r, same peak cell, display-region IoU).
  2. The same comparison on the same slices with three-slice input, from the saved served outputs (kernel
     neuroscan-p2-int8dyn-served), so the two pathways are compared like for like.
  3. Each precision's accuracy and per-class recall on both pathways (what one slice costs the model, and whether
     INT8 changes that cost).
Slices without neighbours (has_context False) already ran on the single-image input; they are counted separately.
Files by direct path (no listing, no zip extraction).
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
SHA = {"FP32": "4937ab381ee3e7c29d5ab18ea761a8c19ddffb6488210fb6030511a8041502eb",
       "INT8-dyn": "ad2f6e0ee22b28ec98295b742075a2168e480801073a8df3f1222df37c74d99f"}


def save():
    (OUT / "single_slice_results.json").write_text(json.dumps(R, indent=2, default=float))


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
          "INT8-dyn": nb_file(CG, "models/neuroscan_r30a_s123_int8_dynamic.onnx")}
for k, p in MODELS.items():
    assert sha256(p) == SHA[k], f"{k}: checksum mismatch"
PARTIAL = {"FP32": np.load(nb_file(CG, "partial/FP32.npz"))}
# three-slice reference: FP32 = saved run, INT8-dyn = the served per-request run
PER_ROW = pd.read_csv(nb_file("neuroscan-p2-int8dyn-served", "e2_per_row_probs.csv"))
ZIPS = {"AISD/normal/": nb_file("neuroscan-q-aisd-normals-prep", "aisd_normal_images.zip"),
        "RSNA_ICH/normal/": nb_file("neuroscan-q-rsna-normals-prep", "rsna_normal_images.zip"),
        "SOOP/": nb_file("neuroscan-q-soop-prep", "soop_images.zip")}
CSV = {"aisd_normal": nb_file("neuroscan-q-aisd-normals-prep", "aisd_normal_rows.csv"),
       "rsna_normal": nb_file("neuroscan-q-rsna-normals-prep", "rsna_normal_rows.csv"),
       "soop": nb_file("neuroscan-q-soop-prep", "soop_rows.csv")}
log(f"inputs resolved and checked ({time.time() - t:.1f}s)")

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


CT = np.where(~held["dataset"].isin(MRI_PAIR_DATASETS).to_numpy())[0]
ctx = held["has_context"].fillna(False).astype(bool).to_numpy()
R["rows"] = {"ct_slices": int(len(CT)), "with_neighbours": int(ctx[CT].sum()),
             "by_dataset": held.iloc[CT]["dataset"].value_counts().to_dict(),
             "by_cohort": held.iloc[CT]["cohort"].value_counts().to_dict()}
log(f"CT slices {len(CT)} (with neighbours {int(ctx[CT].sum())}); {R['rows']['by_dataset']}")

t = time.time()
X = np.empty((len(CT), 224, 224, 3), dtype=np.uint8)
ROWS = [held.iloc[i] for i in CT]


def dec(i):
    """The single-image request, as inference.build_input builds it for CT without neighbours."""
    s = gray(ROWS[i]["out_path"])
    X[i] = np.asarray(Image.merge("RGB", (s, s, s)))


with ThreadPoolExecutor(8) as ex:
    list(ex.map(dec, range(len(ROWS))))
R["file_reads_by_source"] = dict(READS)
log(f"decoded {len(X)} single-slice requests ({time.time() - t:.0f}s, {X.nbytes / 1e9:.2f} GB); reads {READS}")
save()


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



# ------------------------------------------------------------------------------------------------- runs
P1, C1 = {}, {}
for k in ("FP32", "INT8-dyn"):
    t = time.time()
    P1[k], C1[k] = per_request(MODELS[k], np.arange(len(CT)), f"{k} single slice")
    R.setdefault("minutes", {})[k] = round((time.time() - t) / 60, 1)
    np.savez_compressed(OUT / f"single_{k}.npz", probs=P1[k], cams=C1[k], rows=CT)
    save()

# Control: on slices without neighbours the single-image request IS the evaluated input, so FP32 must match the saved run.
noctx = np.where(~ctx[CT])[0]
if len(noctx):
    d = float(np.abs(P1["FP32"][noctx] - PARTIAL["FP32"]["probs"][CT[noctx]]).max())
    R["control_no_neighbours"] = {"rows": int(len(noctx)), "fp32_max_prob_diff_vs_saved": d}
    log(f"control: FP32 single-image vs saved run on {len(noctx)} slices without neighbours: max prob diff {d:.2e}")
    if d > 1e-4:
        print("[ERROR] the single-image FP32 run does not reproduce the saved outputs where inputs are identical")
        save()
        sys.exit(1)

P3 = {"FP32": PER_ROW[[f"p_{l}_FP32" for l in VALID_LABELS]].to_numpy(float)[CT],
      "INT8-dyn": PER_ROW[[f"p_{l}_INT8-dyn" for l in VALID_LABELS]].to_numpy(float)[CT]}
sub = held.iloc[CT].reset_index(drop=True)
y = sub["label"].map(LABEL_TO_IDX).to_numpy()
srt = np.sort(P1["FP32"], 1)
margin1 = srt[:, 2] - srt[:, 1]

COH = {"all_ct": np.ones(len(CT), bool),
       "with_neighbours": ctx[CT], "CPAISD": (sub["cohort"] == "CPAISD").to_numpy()}
R["agreement"] = {"single_slice": {}, "three_slice": {}}
seed = 1101
for path, PP in (("single_slice", P1), ("three_slice", P3)):
    for cname, m in COH.items():
        sel = np.where(m)[0]
        if not len(sel):
            continue
        res = compare(sub.iloc[sel].reset_index(drop=True), PP["FP32"][sel], PP["INT8-dyn"][sel], seed)
        seed += 1
        R["agreement"][path][cname] = res
        d = res["secondary_metric_deltas_variant_minus_fp32"]
        log(f"{path} {cname}: n={res['n']} agreement {res['argmax_agreement']:.4f} {res['argmax_agreement_ci95']} "
            f"flips {res['flips']} tier {res['tier_agreement']:.4f}; ischemic recall change "
            f"{d.get('ischemic_recall', {}).get('diff', float('nan')):+.4f} {d.get('ischemic_recall', {}).get('ci95')}")
        save()

# changed slices on the single-image pathway
a, b = P1["FP32"].argmax(1), P1["INT8-dyn"].argmax(1)
ch = a != b
kind = np.where(a == y, "correct_to_wrong", np.where(b == y, "wrong_to_correct", "wrong_to_wrong"))
R["changes_single_slice"] = {
    "changed": int(ch.sum()), "by_kind": pd.Series(kind[ch]).value_counts().to_dict(),
    "correct_to_wrong_by_truth": pd.Series(sub["label"].to_numpy()[ch & (a == y)]).value_counts().to_dict(),
    "to_normal": int((ch & (b == 0)).sum()), "from_normal": int((ch & (a == 0)).sum()),
    "between_stroke_classes": int((ch & (a > 0) & (b > 0)).sum()),
    "fp32_margin_changed": {"median": float(np.median(margin1[ch])) if ch.any() else None,
                            "max": float(margin1[ch].max()) if ch.any() else None},
    "fp32_margin_all_median": float(np.median(margin1)),
    "three_slice_changed_same_rows": int((P3["FP32"].argmax(1) != P3["INT8-dyn"].argmax(1)).sum())}
log(f"single-slice changes: {json.dumps(R['changes_single_slice'])}")
pd.DataFrame({"out_path": sub["out_path"], "dataset": sub["dataset"], "label": sub["label"], "pk": sub["pk"],
              "has_neighbours": ctx[CT], "fp32": [VALID_LABELS[i] for i in a], "int8_dyn": [VALID_LABELS[i] for i in b],
              "kind": kind, "fp32_margin": margin1.round(5)})[ch].to_csv(OUT / "changed_single_slice.csv", index=False)

# what one slice costs each precision (descriptive, on the CT slices with neighbours, where the inputs differ)
R["pathway_cost"] = {}
w = np.where(ctx[CT])[0]
for k in ("FP32", "INT8-dyn"):
    m3 = slice_metrics(y[w], P3[k][w].argmax(1), P3[k][w])
    m1 = slice_metrics(y[w], P1[k][w].argmax(1), P1[k][w])
    keep_ = ("accuracy", "balanced_accuracy", "normal_recall", "ischemic_recall", "hemorrhagic_recall", "ischemic_auroc",
             "hemorrhagic_auroc")
    R["pathway_cost"][k] = {"rows": int(len(w)), "three_slice": {q: m3.get(q) for q in keep_},
                            "single_slice": {q: m1.get(q) for q in keep_}}
    log(f"{k} on {len(w)} CT slices with neighbours: three-slice {json.dumps(R['pathway_cost'][k]['three_slice'])} "
        f"single {json.dumps(R['pathway_cost'][k]['single_slice'])}")
save()


# maps on the single-image pathway (FP32's predicted class)
def display_region(cam7):
    return np.asarray(srv.ThreeClassPipeline._display_cam(cam7), dtype=np.float32) / 255.0 > 0.6


rec = []
for i in range(len(CT)):
    fa, fb = C1["FP32"][i, a[i]].ravel(), C1["INT8-dyn"][i, a[i]].ravel()
    ra, rb = display_region(C1["FP32"][i, a[i]]), display_region(C1["INT8-dyn"][i, a[i]])
    un = (ra | rb).sum()
    rec.append({"pearson_r": float(np.corrcoef(fa, fb)[0, 1]) if fa.std() > 0 and fb.std() > 0 else np.nan,
                "same_peak_cell": int(fa.argmax() == fb.argmax()),
                "display_iou": float((ra & rb).sum() / un) if un else np.nan})
dm = pd.DataFrame(rec)
R["maps_single_slice"] = {"pearson_r_median": float(dm["pearson_r"].median()),
                          "pearson_r_q1": float(dm["pearson_r"].quantile(.25)),
                          "same_peak_cell": float(dm["same_peak_cell"].mean()),
                          "display_iou_median": float(dm["display_iou"].median()),
                          "display_iou_q1": float(dm["display_iou"].quantile(.25))}
log(f"maps, single-slice pathway: {json.dumps(R['maps_single_slice'])}")
per = pd.DataFrame({"out_path": sub["out_path"], "dataset": sub["dataset"], "label": sub["label"], "cohort": sub["cohort"],
                    "pk": sub["pk"], "has_neighbours": ctx[CT]})
for k in ("FP32", "INT8-dyn"):
    for c, lab in enumerate(VALID_LABELS):
        per[f"p_{lab}_{k}_single"] = P1[k][:, c].round(6)
per.to_csv(OUT / "single_slice_per_row_probs.csv", index=False)
save()
log("Done.")
