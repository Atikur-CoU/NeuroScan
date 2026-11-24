"""
NeuroScan paper 2 -- K7 external CT (JTCSST second revision, comment 2). CPU only; no training.

Why: every earlier fidelity result used held-out partitions of sources that also gave training data (SOOP's hold-out
patients are new, but the source is not). The reviewer asks for data never used in development.

What: CT-ICH (Hssayeni et al., PhysioNet 1.0.0; Kaggle mirror vbookshelf/computed-tomography-ct-images): 82 patients,
about 30 brain-window slices each, a per-slice hemorrhage diagnosis and 318 hemorrhage masks, from Al Hilla Teaching
Hospital, Iraq. No CT-ICH patient, slice or file was used in training, validation, model selection or any earlier
evaluation (development used Kaggle_CT = ozguraslank/brain-stroke-ct-dataset and SICH_CT = PHE-SICH-CT-IDS, not CT-ICH).
Labels: hemorrhagic if the slice has any hemorrhage type, otherwise normal (no ischemic slices: ischemic predictions are
false alarms here). Every copy runs exactly as served: one request per slice, its 4 TTA views as the batch, the
server's inference.py (inserted verbatim by build_kernel.py) for input building, views, softmax and display maps.

Arms:
  A (primary) "prepared": each slice is prepared as in development (per-image 0.5-99.5 percentile window to uint8,
    bilinear resize to 224) and sent with its previous and next slices (CT context pathway; a missing neighbour at
    either end is replaced by the slice itself, as in development).
  B (sensitivity, FP32 and INT8-dynamic only) "raw upload": the CT-ICH JPEG files sent as a user would upload them,
    with no preparation, through the same pathway.

Measures (paired against FP32, per copy):
  1. slice-level argmax agreement, changed slices and their direction; patient-cluster bootstrap (2,000, seed 0);
  2. classification: accuracy, hemorrhagic recall, normal recall (specificity), hemorrhagic AUROC; differences from
     FP32 with patient-cluster intervals;
  3. patient level (triage rule as in K9): patients with a changed slice, hemorrhage patients detected (any slice
     predicted hemorrhagic), no-hemorrhage patients left unflagged, call agreement with FP32;
  4. maps vs FP32 (as module E3): Pearson r of the 7x7 maps of FP32's predicted class, same peak cell, display-region IoU;
  5. maps vs the expert hemorrhage masks (318 slices): hemorrhagic-class display map, peak inside the mask (pointing
     hit, with the chance rate = mask share of the head), share of the display region (> 0.6) inside the mask, and
     share of the mask covered; FP32 and every copy.
Files are opened by direct path from hemorrhage_diagnosis.csv (no folder listing). Model files are checked against the
SHA-256 values of the runs that made them.
"""
import hashlib
import io
import json
import subprocess
import sys
import threading
import time
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

CLS = ("normal", "ischemic", "hemorrhagic")
N_BOOT, WORKERS, SEED = 2000, 4, 0
OUT = Path("/kaggle/working/output")
OUT.mkdir(parents=True, exist_ok=True)
R = {"note": __doc__.strip().splitlines()[0], "onnxruntime": ort.__version__}


def save():
    (OUT / "external_ct_results.json").write_text(json.dumps(R, indent=2, default=float))


def first(*cands, what=""):
    """Mount-path fallback: exists() on each candidate, never a listing."""
    for c in cands:
        if Path(c).exists():
            return Path(c)
    print(f"[ERROR] {what} not found; tried {[str(c) for c in cands]}")
    sys.exit(1)


def nb_file(slug, name):
    roots = (f"/kaggle/input/notebooks/atik68/{slug}", f"/kaggle/input/{slug}")
    return first(*(f"{r}{sub}/{name}" for r in roots for sub in ("", "/output")), what=f"{slug}/{name}")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------- inputs
t = time.time()
SUB = "computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0"
ROOT = first(*(f"{b}/{SUB}" for b in ("/kaggle/input/computed-tomography-ct-images",
                                      "/kaggle/input/datasets/vbookshelf/computed-tomography-ct-images")),
             what="CT-ICH root")
CG, SC, MP = "neuroscan-p2-compression-gradcam", "neuroscan-p2-static-calibration", "neuroscan-p2-mixed-precision"
ST = "models/neuroscan_r30a_s123_int8_static"
MODELS = {  # name: (file, sha256 of the run that made it)
    "FP32": (nb_file("neuroscan-q-onnx-export-check", "neuroscan_r30a_s123.onnx"),
             "4937ab381ee3e7c29d5ab18ea761a8c19ddffb6488210fb6030511a8041502eb"),
    "INT8-dynamic": (nb_file(CG, "models/neuroscan_r30a_s123_int8_dynamic.onnx"),
                     "ad2f6e0ee22b28ec98295b742075a2168e480801073a8df3f1222df37c74d99f"),
    "INT8-static (min-max)": (nb_file(CG, f"{ST}.onnx"),
                              "35d8a4007b813f8fb6eca6c10078823fc5b69a654314c36d76c3412a38b02a46"),
    "INT8-static, percentile": (nb_file(SC, f"{ST}_percentile.onnx"),
                                "b47f3fed74b1f0e2e315e37a205741012dbc55ac83bedab147b7b7e7217d5603"),
    "INT8-static, entropy": (nb_file(SC, f"{ST}_entropy.onnx"),
                             "3aba6da5fb87aac111123a0a68c2d8d23a345dc0fc8500b7268409316d75bd7a"),
    "INT8-static, first layers in FP32": (nb_file(SC, f"{ST}_minmax_first_fp32.onnx"),
                                          "6b00850a2eb051793337a444c301ab99fe7f460aebc7ccb52969993bc1562913"),
    "M1: lowest 10% in FP32": (nb_file(MP, f"{ST}_sens10.onnx"),
                               "4c76bbaa3ccce42f2965d0ec7415f0f96b0b150a6fe881e4e9ab9bcb08d6b7a5"),
    "M2: lowest 25% in FP32": (nb_file(MP, f"{ST}_sens25.onnx"),
                               "11ab9327e4d24f88954520df59471f52b648f7cc8c439cf4fe871823ea5a6b17"),
    "M3: M1 and strided convolutions": (nb_file(MP, f"{ST}_sens10_first.onnx"),
                                        "5fbce2e55932275e60ae0c7b8bdbcbbc696bc313ad8b3849996ef79f39fffa56"),
}
for k, (p, want) in MODELS.items():
    got = sha256(p)
    if got != want:
        print(f"[ERROR] {k}: sha256 {got[:16]} != expected {want[:16]}")
        sys.exit(1)
R["models_sha256_checked"] = {k: v[1] for k, v in MODELS.items()}
log(f"inputs resolved, 9 model hashes match ({time.time() - t:.1f}s); CT-ICH root {ROOT}")

# The server's inference.py, verbatim (inserted by build_kernel.py).
INFERENCE_PY = "__INFERENCE_PY__"
INFERENCE_PY_SHA = "__INFERENCE_PY_SHA__"
assert hashlib.sha256(INFERENCE_PY.encode("utf-8")).hexdigest() == INFERENCE_PY_SHA
R["inference_py_sha256"] = INFERENCE_PY_SHA
SRV = Path("/tmp/srv")
SRV.mkdir(exist_ok=True)
(SRV / "inference.py").write_text(INFERENCE_PY)
sys.path.insert(0, str(SRV))
import inference as srv  # noqa: E402

# ------------------------------------------------------------------------------------------------- rows
t = time.time()
diag = pd.read_csv(ROOT / "hemorrhage_diagnosis.csv")
HTYPES = ["Intraventricular", "Intraparenchymal", "Subarachnoid", "Epidural", "Subdural"]
need = {"PatientNumber", "SliceNumber", "No_Hemorrhage", *HTYPES}
assert need <= set(diag.columns), f"unexpected columns {list(diag.columns)}"
diag = diag.sort_values(["PatientNumber", "SliceNumber"]).reset_index(drop=True)
any_type = diag[HTYPES].sum(1) > 0
incons = int((any_type == (diag["No_Hemorrhage"] == 1)).sum())
diag["label"] = np.where(diag["No_Hemorrhage"] == 1, "normal", "hemorrhagic")
diag["pk"] = diag["PatientNumber"].astype(int)
diag["brain"] = [ROOT / "Patients_CT" / f"{p:03d}" / "brain" / f"{s}.jpg" for p, s in zip(diag.pk, diag.SliceNumber)]
diag["mask"] = [b.with_name(f"{b.stem}_HGE_Seg.jpg") for b in diag["brain"]]
exists = np.array([b.exists() for b in diag["brain"]])
R["rows"] = {"csv_rows": int(len(diag)), "brain_files_missing": int((~exists).sum()),
             "label_type_inconsistent_rows": incons}
diag = diag[exists].reset_index(drop=True)
diag["has_mask"] = [m.exists() for m in diag["mask"]]
SLICES = {(p, s): i for i, (p, s) in enumerate(zip(diag.pk, diag.SliceNumber))}
R["rows"].update({"slices": int(len(diag)), "patients": int(diag.pk.nunique()),
                  "hemorrhagic_slices": int((diag.label == "hemorrhagic").sum()),
                  "patients_with_hemorrhage": int(diag[diag.label == "hemorrhagic"].pk.nunique()),
                  "slices_with_mask": int(diag.has_mask.sum()),
                  "hemorrhagic_slices_without_mask": int(((diag.label == "hemorrhagic") & ~diag.has_mask).sum())})
log(f"rows: {R['rows']} ({time.time() - t:.1f}s)")
save()


# ------------------------------------------------------------------------------------------------- images
def normalize_to_uint8(arr):
    """Development preprocessing (phase B), unchanged: per-image 0.5-99.5 percentile window to uint8."""
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


RAW = [None] * len(diag)


def load(i):
    with Image.open(diag.at[i, "brain"]) as im:
        im.load()
        RAW[i] = im.copy()


t = time.time()
with ThreadPoolExecutor(8) as ex:
    list(ex.map(load, range(len(diag))))
log(f"read {len(RAW)} slices ({time.time() - t:.0f}s); first size {RAW[0].size} mode {RAW[0].mode}")


def prepared(i):
    a = np.asarray(RAW[i])
    if a.ndim == 3:
        a = a.mean(axis=2)
    return Image.fromarray(normalize_to_uint8(a)).resize((224, 224), Image.BILINEAR)


def neighbours(i):
    p, s = diag.at[i, "pk"], diag.at[i, "SliceNumber"]
    return SLICES.get((p, s - 1), i), SLICES.get((p, s + 1), i)


t = time.time()
PREP = [None] * len(diag)


def prep(i):
    PREP[i] = prepared(i)


with ThreadPoolExecutor(8) as ex:
    list(ex.map(prep, range(len(diag))))
XA = np.empty((len(diag), 224, 224, 3), np.uint8)   # arm A: prepared, with context
XB = np.empty((len(diag), 224, 224, 3), np.uint8)   # arm B: raw upload, with context
edge = 0
for i in range(len(diag)):
    a, b = neighbours(i)
    edge += (a == i) or (b == i)
    XA[i] = np.asarray(srv.build_input("CT", PREP[i], PREP[a], PREP[b])[0])
    XB[i] = np.asarray(srv.build_input("CT", RAW[i], RAW[a], RAW[b])[0])
R["rows"]["edge_slices_self_as_neighbour"] = int(edge)
HEAD_SHARE = (XA[..., 1] > 10).mean((1, 2))   # head (non-background) share of each prepared slice
log(f"inputs built for both arms ({time.time() - t:.0f}s; {edge} edge slices use themselves as a neighbour)")


# ------------------------------------------------------------------------------------- per-request inference
def session(path):
    o = ort.SessionOptions()
    o.intra_op_num_threads, o.inter_op_num_threads = 1, 1
    return ort.InferenceSession(str(path), sess_options=o, providers=["CPUExecutionProvider"])


LOCK = threading.Lock()


def per_request(path, X, tag):
    """One run per slice, its 4 views as the batch: the server's request (inference.predict)."""
    n = len(X)
    probs, cams = np.zeros((n, 3)), np.zeros((n, 3, 7, 7), dtype=np.float32)
    local, t0, done = threading.local(), time.time(), [0]

    def one(j):
        if not hasattr(local, "s"):
            local.s = session(path)
        x = np.stack([srv._to_array(v) for v in srv._tta_views(Image.fromarray(X[j]))])
        lg, cm = local.s.run(["logits", "gradcam"], {"image": x})
        probs[j] = srv._softmax(lg.astype(np.float64)).mean(0)
        cams[j] = cm[0]
        with LOCK:
            done[0] += 1
            if done[0] % 500 == 0 or done[0] == n:
                el = time.time() - t0
                log(f"  [{tag}] {done[0]}/{n} ({el:.0f}s; projected {el / done[0] * n / 60:.1f} min)")

    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(one, range(n)))
    return probs, cams


P, C = {}, {}
for k, (path, _) in MODELS.items():
    t = time.time()
    P[("A", k)], C[("A", k)] = per_request(path, XA, f"A {k}")
    R.setdefault("minutes", {})[f"A {k}"] = round((time.time() - t) / 60, 1)
for k in ("FP32", "INT8-dynamic"):
    t = time.time()
    P[("B", k)], C[("B", k)] = per_request(MODELS[k][0], XB, f"B {k}")
    R["minutes"][f"B {k}"] = round((time.time() - t) / 60, 1)
np.savez_compressed(OUT / "external_ct_outputs.npz",
                    **{f"{a}|{k}|probs": v for (a, k), v in P.items()},
                    **{f"{a}|{k}|cams": v for (a, k), v in C.items()})
pr = diag[["pk", "SliceNumber", "label", "has_mask"]].copy()
for (a, k), v in P.items():
    for c, lab in enumerate(CLS):
        pr[f"p_{lab}|{a}|{k}"] = v[:, c].round(6)
pr.to_csv(OUT / "external_ct_per_slice.csv", index=False)
log("per-slice outputs saved")

# ------------------------------------------------------------------------------------------------- analysis
y = diag["label"].map({c: i for i, c in enumerate(CLS)}).to_numpy()
pk = diag["pk"].to_numpy()
pats = np.unique(pk)
pidx = np.searchsorted(pats, pk)


def per_patient(v):
    """Sum of a per-slice vector within each patient."""
    return np.bincount(pidx, weights=v.astype(float), minlength=len(pats))


def patient_call(pred):
    call = np.zeros(len(pats), int)
    np.maximum.at(call, pidx, pred)
    return call


def auroc(yb, s):
    n1, n0 = int(yb.sum()), int(len(yb) - yb.sum())
    if not n1 or not n0:
        return float("nan")
    rk = pd.Series(s).rank().to_numpy()
    return float((rk[yb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def compare(arm, k, seed):
    pa, pb = P[(arm, "FP32")], P[(arm, k)]
    a, b = pa.argmax(1), pb.argmax(1)
    hem, nor = y == 2, y == 0
    # per-patient sums for ratio bootstraps
    S = {"n": per_patient(np.ones(len(y))), "agree": per_patient(a == b),
         "hem": per_patient(hem), "nor": per_patient(nor),
         "a_ok": per_patient(a == y), "b_ok": per_patient(b == y),
         "a_hr": per_patient(hem & (a == 2)), "b_hr": per_patient(hem & (b == 2)),
         "a_nr": per_patient(nor & (a == 0)), "b_nr": per_patient(nor & (b == 0)),
         "chg": per_patient(a != b) > 0,
         # patient level (triage rule): hemorrhagic if any slice predicted hemorrhagic, else ischemic if any
         # slice predicted ischemic, else normal (classes are ordered so that the maximum gives this rule)
         "a_call": patient_call(a), "b_call": patient_call(b),
         "has_hem": per_patient(hem) > 0}
    S["a_det"], S["b_det"] = per_patient(hem & (a == 2)) > 0, per_patient(hem & (b == 2)) > 0

    def stat(w):
        sm = {kk: (v * w).sum() for kk, v in S.items()}
        nohem = (~S["has_hem"]).astype(float) * w
        ph = S["has_hem"].astype(float) * w
        return {"agreement": sm["agree"] / sm["n"],
                "d_accuracy": (sm["b_ok"] - sm["a_ok"]) / sm["n"],
                "d_hemorrhagic_recall": (sm["b_hr"] - sm["a_hr"]) / sm["hem"],
                "d_normal_recall": (sm["b_nr"] - sm["a_nr"]) / sm["nor"],
                "share_patients_changed": sm["chg"] / w.sum(),
                "patient_call_agreement": ((S["a_call"] == S["b_call"]) * w).sum() / w.sum(),
                "d_patients_detected": ((S["b_det"].astype(float) - S["a_det"]) * ph).sum() / ph.sum(),
                "d_nohem_unflagged": (((S["b_call"] == 0).astype(float) - (S["a_call"] == 0)) * nohem).sum()
                / nohem.sum()}

    rng = np.random.default_rng(seed)
    bs = [stat(np.bincount(rng.integers(0, len(pats), len(pats)), minlength=len(pats)).astype(float))
          for _ in range(N_BOOT)]
    ci = {kk: [float(np.nanpercentile([s[kk] for s in bs], 2.5)), float(np.nanpercentile([s[kk] for s in bs], 97.5))]
          for kk in bs[0]}
    point = stat(np.ones(len(pats)))
    nohem_p = ~S["has_hem"]
    out = {"slices": int(len(y)), "patients": int(len(pats)), "agreement": point["agreement"],
           "agreement_ci95": ci["agreement"], "changed_slices": int((a != b).sum()),
           "changed_table_rows_fp32_cols_copy": np.bincount(a * 3 + b, minlength=9).reshape(3, 3).tolist(),
           "changed_towards_normal": int(((a != b) & (b == 0)).sum()),
           "changed_away_from_normal": int(((a != b) & (a == 0)).sum()),
           "max_abs_prob_diff": float(np.abs(pa - pb).max()),
           "fp32": {"accuracy": float((a == y).mean()), "hemorrhagic_recall": float((a[y == 2] == 2).mean()),
                    "normal_recall": float((a[y == 0] == 0).mean()), "hemorrhagic_auroc": auroc(y == 2, pa[:, 2]),
                    "ischemic_predictions": int((a == 1).sum())},
           "copy": {"accuracy": float((b == y).mean()), "hemorrhagic_recall": float((b[y == 2] == 2).mean()),
                    "normal_recall": float((b[y == 0] == 0).mean()), "hemorrhagic_auroc": auroc(y == 2, pb[:, 2]),
                    "ischemic_predictions": int((b == 1).sum())},
           "differences_copy_minus_fp32": {kk: {"value": point[kk], "ci95": ci[kk]} for kk in point
                                           if kk.startswith("d_")},
           "patient_level": {"patients_with_changed_slice": int(S["chg"].sum()),
                             "share_changed_ci95": ci["share_patients_changed"],
                             "call_agreement": point["patient_call_agreement"],
                             "call_agreement_ci95": ci["patient_call_agreement"],
                             "hemorrhage_patients": int(S["has_hem"].sum()),
                             "detected_fp32": int(S["a_det"].sum()), "detected_copy": int(S["b_det"].sum()),
                             "no_hemorrhage_patients": int(nohem_p.sum()),
                             "unflagged_fp32": int((S["a_call"][nohem_p] == 0).sum()),
                             "unflagged_copy": int((S["b_call"][nohem_p] == 0).sum())}}
    return out


def display(cam7):
    return np.asarray(srv.ThreeClassPipeline._display_cam(cam7), dtype=np.float32) / 255.0


def maps_vs_fp32(arm, k):
    ca, cb, cls_a = C[(arm, "FP32")], C[(arm, k)], P[(arm, "FP32")].argmax(1)
    r_, peak, iou = [], [], []
    for i in range(len(y)):
        a, b = ca[i, cls_a[i]].ravel(), cb[i, cls_a[i]].ravel()
        r_.append(float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan)
        peak.append(a.argmax() == b.argmax())
        ra, rb = display(ca[i, cls_a[i]]) > 0.6, display(cb[i, cls_a[i]]) > 0.6
        un = (ra | rb).sum()
        iou.append((ra & rb).sum() / un if un else np.nan)
    q = lambda s: {"median": float(np.nanmedian(s)), "q1": float(np.nanpercentile(s, 25)),
                   "q3": float(np.nanpercentile(s, 75))}
    return {"pearson_r": q(np.array(r_)), "same_peak_cell": float(np.mean(peak)), "display_iou": q(np.array(iou))}


t = time.time()
MASKS = {}
for i in np.where(diag["has_mask"].to_numpy())[0]:
    with Image.open(diag.at[i, "mask"]) as m:
        MASKS[i] = np.asarray(m.convert("L").resize((224, 224), Image.NEAREST)) > 127
empty = [i for i, m in MASKS.items() if not m.any()]
log(f"{len(MASKS)} masks read, {len(empty)} empty after resizing ({time.time() - t:.1f}s)")


def maps_vs_masks(arm, k):
    """Hemorrhagic-class display map against the expert mask, on slices with a non-empty mask."""
    cams, rows = C[(arm, k)], []
    for i, m in MASKS.items():
        if not m.any():
            continue
        d = display(cams[i, 2])
        reg = d > 0.6
        pi = np.unravel_index(d.argmax(), d.shape)
        rows.append({"i": int(i), "pk": int(pk[i]), "hit": bool(m[pi]), "chance": float(m.mean() / max(HEAD_SHARE[i], 1e-6)),
                     "region_in_mask": float((reg & m).sum() / reg.sum()) if reg.any() else np.nan,
                     "mask_covered": float((reg & m).sum() / m.sum()),
                     "predicted_hemorrhagic": bool(P[(arm, k)][i].argmax() == 2)})
    d = pd.DataFrame(rows)
    pat = d.groupby("pk")["hit"].mean()
    rng = np.random.default_rng(SEED)
    keys = d["pk"].to_numpy()
    groups = [np.where(keys == p)[0] for p in pat.index]
    bs = [d["hit"].to_numpy()[np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])].mean()
          for _ in range(N_BOOT)]
    return {"slices": int(len(d)), "patients": int(len(pat)), "pointing_hit": float(d["hit"].mean()),
            "pointing_hit_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "chance_hit_mean": float(d["chance"].mean()),
            "region_in_mask_median": float(d["region_in_mask"].median()),
            "mask_covered_median": float(d["mask_covered"].median()),
            "pointing_hit_on_slices_predicted_hemorrhagic": float(d.loc[d.predicted_hemorrhagic, "hit"].mean())
            if d.predicted_hemorrhagic.any() else None,
            "slices_predicted_hemorrhagic": int(d.predicted_hemorrhagic.sum())}, d


R["A_prepared"], R["B_raw_upload"] = {}, {}
for s, k in enumerate(MODELS):
    if k == "FP32":
        continue
    R["A_prepared"][k] = compare("A", k, SEED + s)
    R["A_prepared"][k]["maps_vs_fp32"] = maps_vs_fp32("A", k)
    x = R["A_prepared"][k]
    log(f"A {k}: agreement {x['agreement']:.4f} {x['agreement_ci95']} changed {x['changed_slices']} "
        f"patients changed {x['patient_level']['patients_with_changed_slice']} call agreement "
        f"{x['patient_level']['call_agreement']:.3f} acc {x['fp32']['accuracy']:.4f}->{x['copy']['accuracy']:.4f}")
    save()
R["B_raw_upload"]["INT8-dynamic"] = compare("B", "INT8-dynamic", SEED + 20)
R["B_raw_upload"]["INT8-dynamic"]["maps_vs_fp32"] = maps_vs_fp32("B", "INT8-dynamic")
log(f"B INT8-dynamic: {json.dumps({k: R['B_raw_upload']['INT8-dynamic'][k] for k in ('agreement', 'changed_slices')})}")
# FP32 prepared vs FP32 raw upload: how much the input preparation alone moves the model
fa, fb = P[("A", "FP32")].argmax(1), P[("B", "FP32")].argmax(1)
R["FP32_prepared_vs_raw"] = {"agreement": float((fa == fb).mean()), "accuracy_prepared": float((fa == y).mean()),
                             "accuracy_raw": float((fb == y).mean())}
save()

R["maps_vs_expert_masks"] = {}
allm = []
for k in MODELS:
    res, d = maps_vs_masks("A", k)
    R["maps_vs_expert_masks"][k] = res
    allm.append(d.assign(copy=k))
    log(f"masks {k}: pointing hit {res['pointing_hit']:.3f} {res['pointing_hit_ci95']} (chance {res['chance_hit_mean']:.3f}); "
        f"region in mask {res['region_in_mask_median']:.3f}; mask covered {res['mask_covered_median']:.3f}")
pd.concat(allm).to_csv(OUT / "maps_vs_masks_per_slice.csv", index=False)
R["minutes_total"] = round((time.time() - T0) / 60, 1)
save()
log("Done.")
