"""
NeuroScan -- Figure 3 (2026-10-09): prototype output for held-out test cases. CPU; no training. Six held-out test
rows, chosen at random by reference label only (seed 2026), are run through the served prototype code (inference.py,
ONNX model, unchanged) in the server's package versions; the figure shows input, Grad-CAM overlay, probabilities and
model-alert tier. The data pipeline above main()'s loader construction is the final model's own script, unchanged.
"""
import gzip
import json
import shutil
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path


def ensure_pkg(import_name, pip_name=None):
    try:
        __import__(import_name)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name or import_name], check=True)


ensure_pkg("torch")
ensure_pkg("torchvision")
ensure_pkg("sklearn", "scikit-learn")
ensure_pkg("nibabel")

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

INPUT_ROOT = Path("/kaggle/input")
DS, NB = INPUT_ROOT / "datasets", INPUT_ROOT / "notebooks"
SPLIT_DIR = DS / "atik68/neuroscan-phase-p-patient-split-v3"
PHASE_B = DS / "atik68/neuroscan-phase-b-images"
RSNA_J = DS / "atik68/neuroscan-phase-j-rsna-images"
RSNA_L = DS / "atik68/neuroscan-phase-l-rsna-more-images"
AISD_K = DS / "atik68/neuroscan-phase-k-aisd-images"
CPAISD_V2_IMG = DS / "atik68/neuroscan-phase-n-cpaisd-v2-images"
ISLES_RAW = DS / "atik68/neuroscan-isles2022"
AISD_PREP = NB / "atik68/neuroscan-q-aisd-normals-prep"
RSNA_PREP = NB / "atik68/neuroscan-q-rsna-normals-prep"
SOOP_PREP = NB / "atik68/neuroscan-q-soop-prep"
RSNA_EXTRA = NB / "atik68/neuroscan-q-rsna-extra-prep"
RSNA_DCM_PREP = NB / "atik68/neuroscan-q-rsna-dicom-prep"
R24_DIR = NB / "atik68/neuroscan-q-round-24-expanded"
OWN_DIR = NB / "atik68/neuroscan-q-round30a-convnext-s123-office"
CNX_WEIGHTS = INPUT_ROOT / "weights/convnext_tiny-983f1562.pth"
MICRO_BATCH = 32
SOOP_KEEP_HOLDOUT = 0.15  # fraction of the 437 SOOP hold-out patients kept out of training
EXCLUDED_DATASETS = ("MRI_Images",)
SUPP_ROOTS = [PHASE_B, RSNA_J, RSNA_L, AISD_K, CPAISD_V2_IMG]

OUT_ROOT = Path("/kaggle/working")
CKPT_DIR = OUT_ROOT / "checkpoints"
OUTPUT_DIR = OUT_ROOT / "output"
PREVIEW_DIR = OUTPUT_DIR / "soop_edge_previews"
EXTRA_ROOT = Path("/tmp/extra_normals")
SOOP_ROOT = Path("/tmp/soop_png")
DCM_ROOT = Path("/tmp/rsna_dcm")
for d in (CKPT_DIR, OUTPUT_DIR, PREVIEW_DIR):
    d.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = (224, 224)
BATCH_SIZE = 64
LR = 1e-4
WEIGHT_DECAY = 1e-4
TOTAL_EPOCHS = 45
EARLY_STOP_PATIENCE = 15
ISCHEMIC_WEIGHT_BOOST = 1.5
NUM_WORKERS = 4
VALID_LABELS = ("normal", "ischemic", "hemorrhagic")
LABEL_TO_IDX = {"normal": 0, "ischemic": 1, "hemorrhagic": 2}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}
NEW_PREFIXES = ("AISD/normal/", "RSNA_ICH/normal/", "RSNA_ICH/extra/")
MRI_PAIR_DATASETS = ("ISLES2022_dwi_adc", "SOOP_dwi_adc")
ISLES_KEY_RE = re.compile(r"^(sub-strokecase\d+)/(ses-\d+)/\w+/slice(\d+)$")
SIZE_BINS = [-1, 0, 10, 25, 50, 100, 250, 1000, 10 ** 9]
SIZE_NAMES = ["0", "1-10", "11-25", "26-50", "51-100", "101-250", "251-1000", ">1000"]
SKIP_DESCEND = {"hemorrhagic", "ischemic", "normal", "stroke"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T0 = time.time()


def log(m):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {m}", flush=True)


def need(p):
    if not Path(p).exists():
        print(f"[ERROR] expected input missing: {p}")
        sys.exit(1)
    return Path(p)


def scan(root, wanted, max_depth=4, stop_when_found=False):
    """ONE breadth-first pass per dataset; every folder listed at most once; image folders (named
    leaves or any folder with >500 entries) are never descended into. Logs paths and timing."""
    root, t0 = Path(root), time.time()
    found, visited, level = {w: [] for w in wanted}, [], [root]
    for depth in range(max_depth + 1):
        nxt = []
        for d in level:
            visited.append(d)
            try:
                entries = list(os.scandir(d))
            except (FileNotFoundError, NotADirectoryError, PermissionError):
                continue
            if depth > 0 and len(entries) > 500:
                continue
            for e in entries:
                if e.is_dir():
                    if e.name in found:
                        found[e.name].append(Path(e.path))
                    if e.name not in SKIP_DESCEND and not e.name.startswith("stage_"):
                        nxt.append(Path(e.path))
        level = nxt
        if not level or (stop_when_found and all(found.values())):
            break
    log(f"scan {root}: {time.time() - t0:.1f}s, {len(visited)} dirs, found {{{', '.join(f'{k}: {[str(p) for p in v]}' for k, v in found.items())}}}")
    return found, visited


# ---------------------------------------------------------------- data
NEW_INDEX, SOOP_INDEX, SCANS, DCM_INDEX = {}, {}, {}, {}


def unpack(zip_path, dest, prefixes, index):
    t0 = time.time()
    with zipfile.ZipFile(need(zip_path)) as zf:
        zf.extractall(dest)
    for pre in prefixes:
        for f in (dest / pre).rglob("*.png"):
            index[str(f.relative_to(dest)).replace("\\", "/")] = f
    log(f"Unpacked {zip_path.name}: {len(index)} indexed ({time.time() - t0:.0f}s)")


def load_split(split):
    df = pd.read_csv(need(SPLIT_DIR / f"{split}.csv"))
    return df[df["label"].isin(VALID_LABELS)].reset_index(drop=True)


def find_images_root():
    for d in SCANS[PHASE_B][0]["Kaggle_CT"]:
        if (d / "stroke").exists():
            log(f"Images root: {d.parent}")
            return d.parent
    print(f"[ERROR] Kaggle_CT/stroke not found within {PHASE_B}")
    sys.exit(1)


def build_supplementary_index():
    t0, index = time.time(), {}
    for leaf in ("hemorrhagic", "ischemic"):
        for root in SUPP_ROOTS:
            for d in SCANS[root][0][leaf]:
                for f in d.glob("*.png"):
                    index[f.name] = f
    log(f"Supplementary index: {len(index)} files ({time.time() - t0:.1f}s)")
    return index


def build_cpaisd_v2_index(sample_filename):
    hit = next((d for d in SCANS[CPAISD_V2_IMG][1] if (d / sample_filename).exists()), None)
    index = {f.name: f for f in hit.glob("*.png")} if hit else {}
    log(f"CPAISD_v2 index: {len(index)} files from {hit}")
    return index


class Loader:
    """rsna_dicom=True: RSNA_ICH rows read the DICOM-derived brain-window channel (Round 26a).
    rsna_dicom=False: the old 128 px mirror images (Round 24's own input, for re-scoring).
    All path indexes are attributes (copies of the module-level dicts, filled before the
    Loader is built), so DataLoader worker processes -- spawned fresh on Windows, where
    module globals start empty -- receive them with the pickled dataset. This makes
    NUM_WORKERS > 0 safe on the office PC."""

    def __init__(self, main_root, supp_index, cpaisd_index=None, rsna_dicom=True):
        self.main_root, self.supp_index, self.cpaisd_index = main_root, supp_index, cpaisd_index
        self.rsna_dicom = rsna_dicom
        self.new_index, self.soop_index, self.dcm_index = dict(NEW_INDEX), dict(SOOP_INDEX), dict(DCM_INDEX)

    def _resolve(self, p):
        if p.startswith(NEW_PREFIXES):
            return self.new_index[p]
        if p.startswith("SOOP/"):
            return self.soop_index[p]
        if p.startswith("CPAISD_v2/"):
            return self.cpaisd_index[Path(p).name]
        if p.startswith("RSNA_ICH/") or p.startswith("AISD/"):
            return self.supp_index[Path(p).name]
        return self.main_root / p

    def _slice(self, p):
        if self.rsna_dicom and p.startswith("RSNA_ICH/"):
            return Image.open(self.dcm_index[Path(p).stem]).getchannel("R").resize(TARGET_SIZE)
        return Image.open(self._resolve(p)).convert("L").resize(TARGET_SIZE)

    def merged(self, row):
        if row["dataset"] in MRI_PAIR_DATASETS:
            dwi, adc = self._slice(row["dwi_out_path"]), self._slice(row["adc_out_path"])
            return Image.merge("RGB", (dwi, adc, dwi))
        s = self._slice(row["out_path"])
        if bool(row["has_context"]):
            return Image.merge("RGB", (self._slice(row["prev_out_path"]), s, self._slice(row["next_out_path"])))
        return Image.merge("RGB", (s, s, s))


NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
eval_transform = transforms.Compose([transforms.ToTensor(), NORMALIZE])
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    NORMALIZE,
])


class TrainDataset(Dataset):
    def __init__(self, df, loader, transform):
        self.df, self.loader, self.transform = df.reset_index(drop=True), loader, transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        return self.transform(self.loader.merged(row)), LABEL_TO_IDX[row["label"]]


def compute_class_weights(train_df):
    counts = train_df["label"].value_counts()
    n, k = len(train_df), len(VALID_LABELS)
    w = {l: n / (k * counts.get(l, 1)) for l in VALID_LABELS}
    w["ischemic"] *= ISCHEMIC_WEIGHT_BOOST
    log(f"Class counts {counts.to_dict()}, weights { {k: round(v, 4) for k, v in w.items()} }")
    return torch.tensor([w[l] for l in VALID_LABELS], dtype=torch.float32)


def build_model(pretrained=True):
    """ConvNeXt-Tiny, ImageNet-1k V1 weights; head -> 3 classes."""
    if pretrained and CNX_WEIGHTS.exists():
        m = models.convnext_tiny(weights=None)
        m.load_state_dict(torch.load(CNX_WEIGHTS, map_location="cpu"))
        log(f"ConvNeXt-Tiny ImageNet weights loaded from {CNX_WEIGHTS}")
    else:
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, len(VALID_LABELS))
    return m


def build_resnet18(pretrained=True):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, len(VALID_LABELS))
    return m


def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot, correct, n = 0.0, 0, 0
    with torch.enable_grad() if train else torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if train:
                # exact 64-batch update via micro-batches: weighted-sum CE / full-batch weight sum
                optimizer.zero_grad()
                wsum = criterion.weight[y].sum()
                parts, lval = [], 0.0
                for xs, ys in zip(x.split(MICRO_BATCH), y.split(MICRO_BATCH)):
                    o = model(xs)
                    l = nn.functional.cross_entropy(o, ys, weight=criterion.weight, reduction="sum") / wsum
                    l.backward()
                    parts.append(o.detach())
                    lval += l.item()
                optimizer.step()
                out = torch.cat(parts)
                tot += lval * x.size(0)
            else:
                out = model(x)
                tot += criterion(out, y).item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += y.size(0)
    return tot / n, correct / n


@torch.no_grad()
def predict(models_, df, loaders, tag):
    """Round 19's exact 4-view PIL TTA. loaders: {model name: Loader}; each model sees its own
    input format (Round 26a: DICOM-derived RSNA; Round 24: the old 128 px RSNA images)."""
    for m in models_.values():
        m.eval()
    out, t0 = {k: [] for k in models_}, time.time()
    for start in range(0, len(df), BATCH_SIZE):
        chunk = df.iloc[start:start + BATCH_SIZE]
        batches = {}
        for k, m in models_.items():
            ld = loaders[k]
            if id(ld) not in batches:
                flat = []
                for _, row in chunk.iterrows():
                    img = ld.merged(row)
                    for v in (img, transforms.functional.hflip(img),
                              transforms.functional.rotate(img, 8, fill=0),
                              transforms.functional.rotate(img, -8, fill=0)):
                        flat.append(eval_transform(v))
                batches[id(ld)] = torch.stack(flat).to(DEVICE)
            out[k].append(torch.softmax(m(batches[id(ld)]), 1).view(len(chunk), 4, -1).mean(1).cpu().numpy())
        done = start + len(chunk)
        if done % (BATCH_SIZE * 40) == 0 or done == len(df):
            log(f"  [{tag}] {done}/{len(df)} ({time.time() - t0:.0f}s)")
    return {k: np.concatenate(v) for k, v in out.items()}


def score(df, probs):
    y = df["label"].map(LABEL_TO_IDX).to_numpy()
    p = probs.argmax(1)
    rep = classification_report(y, p, labels=[0, 1, 2], target_names=list(VALID_LABELS), output_dict=True, zero_division=0)
    res = {"n": int(len(df)), "accuracy": float((y == p).mean()), "macro_f1": rep["macro avg"]["f1-score"],
           "confusion_matrix": confusion_matrix(y, p, labels=[0, 1, 2]).tolist()}
    for l in VALID_LABELS:
        res[f"{l}_recall"], res[f"{l}_precision"] = rep[l]["recall"], rep[l]["precision"]
    for l in ("ischemic", "hemorrhagic"):
        yb = (y == LABEL_TO_IDX[l]).astype(int)
        if 0 < yb.sum() < len(yb):
            res[f"{l}_auroc"] = float(roc_auc_score(yb, probs[:, LABEL_TO_IDX[l]]))
            res[f"{l}_pr_auc"] = float(average_precision_score(yb, probs[:, LABEL_TO_IDX[l]]))
    return res


def per_source(df, probs):
    pred = np.array([IDX_TO_LABEL[i] for i in probs.argmax(1)])
    out = {}
    for (src, lbl), g in df.assign(pred=pred).groupby(["dataset", "label"]):
        out.setdefault(src, {})[lbl] = {"n": int(len(g)), "recall": float((g["pred"] == lbl).mean()),
                                        "predicted_as": g["pred"].value_counts().to_dict()}
    return out


def load_nifti(folder, stem):
    for ext in (".nii.gz", ".nii", ".niigz"):
        p = folder / f"{stem}{ext}"
        if p.exists():
            if ext == ".niigz":  # gzip-compressed NIfTI with a non-standard extension (as in Phase B / Phase 1)
                with gzip.open(p, "rb") as f:
                    raw = f.read()
                tmp = tempfile.NamedTemporaryFile(suffix=".nii", delete=False)
                tmp.write(raw)
                tmp.close()
                return nib.load(tmp.name)
            return nib.load(str(p))
    return None


def isles_areas(df):
    t0 = time.time()
    deriv = scan(ISLES_RAW, {"derivatives"}, max_depth=3, stop_when_found=True)[0]["derivatives"]
    if not deriv:
        log("[WARN] ISLES derivatives not found; ISLES size tables skipped")
        return {}
    deriv, cache, area = deriv[0], {}, {}
    for i, r in df[(df["dataset"] == "ISLES2022_dwi_adc") & (df["label"] == "ischemic")].iterrows():
        m = ISLES_KEY_RE.match(str(r["source_key"]))
        if not m:
            continue
        pat, s = (m.group(1), m.group(2)), int(m.group(3))
        if pat not in cache:
            img = load_nifti(deriv / pat[0] / pat[1], f"{pat[0]}_{pat[1]}_msk")
            cache[pat] = img.get_fdata(dtype=np.float32) if img is not None else None
        a = cache[pat]
        if a is not None and 0 <= s < a.shape[2]:
            sl = ((a[:, :, s] > 0) * 255).astype(np.uint8)
            area[i] = int((np.asarray(Image.fromarray(sl).resize(TARGET_SIZE, Image.NEAREST)) > 0).sum())
    log(f"ISLES lesion areas: {len(area)} slices, {sum(v is not None for v in cache.values())}/{len(cache)} masks ({time.time() - t0:.0f}s)")
    return area


def size_table(df, probs, area):
    x = df.assign(pred=[IDX_TO_LABEL[i] for i in probs.argmax(1)])
    x = x[x.index.isin(list(area.keys()))].copy()
    x["area"] = x.index.map(area)
    x["hit"] = x["pred"] == "ischemic"
    x["bin"] = pd.cut(x["area"], SIZE_BINS, labels=SIZE_NAMES)
    t = x.groupby("bin", observed=False).agg(n=("hit", "size"), caught=("hit", "sum"))
    t["recall"] = (t["caught"] / t["n"]).round(4)
    return t.reset_index().astype({"bin": str}).to_dict("records")


def per_patient(df, probs, id_col):
    x = df.assign(pred=[IDX_TO_LABEL[i] for i in probs.argmax(1)])
    g = x.groupby(id_col)
    les = g.apply(lambda t: float(((t["label"] == "ischemic") & (t["pred"] == "ischemic")).any())
                  if (t["label"] == "ischemic").any() else np.nan).astype(float)
    fp = g.apply(lambda t: ((t["label"] == "normal") & (t["pred"] != "normal")).sum() / max((t["label"] == "normal").sum(), 1))
    return {"patients": int(g.ngroups), "lesion_patients_detected_ge1_slice": int(les.sum()),
            "lesion_patients": int(les.notna().sum()), "median_normal_slice_fp_rate": float(fp.median())}


def soop_edge_check(soop):
    """Share of SOOP rows in the top/bottom 3 slices of each scan + 12 edge previews."""
    t0 = time.time()
    x = soop.copy()
    x["rank_lo"] = x.groupby("subject")["slice"].rank(method="first") - 1
    x["rank_hi"] = x.groupby("subject")["slice"].rank(method="first", ascending=False) - 1
    edge = (x["rank_lo"] < 3) | (x["rank_hi"] < 3)
    rng = np.random.default_rng(0)
    pick = x[edge & (x["label"] == "normal")].sample(n=min(12, int(edge.sum())), random_state=0)
    fg = []
    for _, r in pick.iterrows():
        d = Image.open(SOOP_INDEX[r["dwi_out_path"]]).convert("L")
        a = Image.open(SOOP_INDEX[r["adc_out_path"]]).convert("L")
        fg.append(float((np.asarray(d) > 10).mean()))
        canvas = Image.new("L", (448, 224))
        canvas.paste(d.resize(TARGET_SIZE), (0, 0))
        canvas.paste(a.resize(TARGET_SIZE), (224, 0))
        canvas.save(PREVIEW_DIR / f"{r['subject']}_slice{int(r['slice']):02d}.png")
    rep = {"rows": int(len(x)), "edge_rows_top_bottom_3": int(edge.sum()), "edge_share": round(float(edge.mean()), 4),
           "edge_rows_by_label": x[edge]["label"].value_counts().to_dict(),
           "preview_dwi_foreground_fraction": [round(v, 3) for v in fg]}
    log(f"SOOP edge-slice check: {rep} ({time.time() - t0:.0f}s)")
    return rep


def main():
    log(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'cpu'}); backbone ConvNeXt-Tiny")
    unpack(AISD_PREP / "aisd_normal_images.zip", EXTRA_ROOT, ["AISD/normal"], NEW_INDEX)
    unpack(RSNA_PREP / "rsna_normal_images.zip", EXTRA_ROOT, ["RSNA_ICH/normal"], NEW_INDEX)
    unpack(SOOP_PREP / "soop_images.zip", SOOP_ROOT, ["SOOP"], SOOP_INDEX)
    unpack(RSNA_EXTRA / "rsna_extra_images.zip", EXTRA_ROOT, ["RSNA_ICH/extra"], NEW_INDEX)
    new_rows = pd.concat([pd.read_csv(need(AISD_PREP / "aisd_normal_rows.csv")),
                          pd.read_csv(need(RSNA_PREP / "rsna_normal_rows.csv"))], ignore_index=True)
    rsna_extra = pd.read_csv(need(RSNA_EXTRA / "rsna_extra_rows.csv"))
    assert (rsna_extra["split"] == "train").all(), "RSNA extra rows must be train-only"
    miss = [p for p in rsna_extra["out_path"] if p not in NEW_INDEX]
    if miss:
        print(f"[ERROR] {len(miss)} RSNA extra images missing, e.g. {miss[:3]}")
        sys.exit(1)
    soop = pd.read_csv(need(SOOP_PREP / "soop_rows.csv"))
    missing = [p for p in pd.concat([soop["dwi_out_path"], soop["adc_out_path"]]) if p not in SOOP_INDEX]
    if missing:
        print(f"[ERROR] {len(missing)} SOOP images missing, e.g. {missing[:3]}")
        sys.exit(1)
    report = {"soop_edge_check": soop_edge_check(soop)}

    # Round 24 data: split v3 WITH CPAISD_v2 (its v3 train/val/test patient assignment; the 71
    # whole-hemisphere masks are already absent from v3 train), MRI_Images excluded,
    # AISD/RSNA normals, RSNA extra (train only), SOOP train + 85% of the old hold-out.
    base, splits = {}, {}
    for s in ("train", "val", "test"):
        b = load_split(s)
        log(f"{s}: excluding {int(b['dataset'].isin(EXCLUDED_DATASETS).sum())} MRI_Images rows; "
            f"CPAISD_v2 rows {int((b['dataset'] == 'CPAISD_v2').sum())} now INCLUDED")
        base[s] = b[~b["dataset"].isin(EXCLUDED_DATASETS)].reset_index(drop=True)
        splits[s] = pd.concat([base[s], new_rows[new_rows["split"] == s].drop(columns=["split"])], ignore_index=True)
    cols = list(splits["train"].columns)
    ext_all = soop[soop["split"] == "external"]
    subs = sorted(ext_all["subject"].unique())
    rng_h = np.random.default_rng(42)
    keep = set(rng_h.choice(subs, size=int(round(SOOP_KEEP_HOLDOUT * len(subs))), replace=False))
    soop_hold = ext_all[ext_all["subject"].isin(keep)].reset_index(drop=True)
    soop_to_train = ext_all[~ext_all["subject"].isin(keep)]
    train = pd.concat([splits["train"], soop[soop["split"] == "train"].reindex(columns=cols),
                       soop_to_train.reindex(columns=cols), rsna_extra.reindex(columns=cols)], ignore_index=True)
    val_all = pd.concat([splits["val"], soop[soop["split"] == "val"].reindex(columns=cols + ["subject", "lesion_px"])],
                        ignore_index=True)
    rsna_ids = pd.concat([d.loc[d["dataset"] == "RSNA_ICH", "out_path"] for d in (train, val_all, splits["test"])])
    report["data"] = {"train_rows": int(len(train)), "val_rows": int(len(val_all)),
                      "train_by_dataset": train["dataset"].value_counts().to_dict(),
                      "train_by_label": train["label"].value_counts().to_dict(),
                      "soop_holdout_patients": len(keep), "soop_holdout_rows": int(len(soop_hold)),
                      "soop_moved_to_train_patients": int(soop_to_train["subject"].nunique()),
                      "rsna_extra_rows": int(len(rsna_extra)), "rsna_rows_all_splits": int(len(rsna_ids))}
    log(f"Data: {report['data']}")

    for r in SUPP_ROOTS:
        SCANS[r] = scan(r, {"Kaggle_CT", "hemorrhagic", "ischemic"})
    main_root, supp = find_images_root(), build_supplementary_index()
    cp_sample = base["train"].loc[base["train"]["dataset"] == "CPAISD_v2", "out_path"].iloc[0]
    cp_index = build_cpaisd_v2_index(Path(cp_sample).name)
    loader = Loader(main_root, supp, cp_index, rsna_dicom=False)  # Round 24 input (old 128 px RSNA images)
    loader_r24 = loader

    # ================================================================================================================
    # FIGURE 3. Everything above this line is the final model's own data pipeline, unchanged. Cases are held-out test
    # rows chosen at random by reference label only (seed 2026), then run through the served prototype code (ONNX).
    # ================================================================================================================
    import os
    import subprocess
    import sys
    rmap = pd.read_csv(need(PHASE2 / "rsna_patient_map.csv"))
    RSNA_MAP.update(dict(zip(rmap["stem"].astype(str), rmap["PatientID"])))
    test = pd.concat([splits["test"], soop_hold], ignore_index=True)
    saved = pd.concat([pd.read_csv(need(SAVED / "predictions_test.csv")), pd.read_csv(need(SAVED / "predictions_soop_external.csv"))],
                      ignore_index=True)
    pc = [f"p_{l}_round30as123" for l in VALID_LABELS]
    test = test.merge(saved[["source_key", "out_path"] + pc], on=["source_key", "out_path"], how="left", validate="many_to_one")
    rng = np.random.default_rng(2026)
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    cases = []
    for ds, lab, mri in (("SICH_CT", "hemorrhagic", False), ("AISD", "ischemic", False), ("AISD", "normal", False),
                         ("ISLES2022_dwi_adc", "ischemic", True), ("SOOP_dwi_adc", "ischemic", True), ("SOOP_dwi_adc", "normal", True)):
        g = test[(test["dataset"] == ds) & (test["label"] == lab)]
        if not mri:
            g = g[g["has_context"].astype(bool)]
        r = g.iloc[int(rng.integers(len(g)))]
        name = f"{SHORT[ds]}_{lab}"
        d = CASE_DIR / name
        d.mkdir()
        files = ({"main": r["dwi_out_path"], "adc": r["adc_out_path"]} if mri else
                 {"prev": r["prev_out_path"], "main": r["out_path"], "next": r["next_out_path"]})
        for k, pth in files.items():
            Image.open(loader._resolve(pth)).convert("L").save(d / f"{k}.png")
        cases.append({"name": name, "dir": str(d), "modality": "MRI" if mri else "CT", "source": SHORT[ds], "label": lab,
                      "offline_probs": [round(float(r[c]), 4) for c in pc]})
    (CASE_DIR / "cases.json").write_text(json.dumps(cases))

    # ---- served prototype code, unchanged, in its own package environment (as on the server)
    srv = Path("/tmp/srv")
    (srv / "models").mkdir(parents=True, exist_ok=True)
    shutil.copy(ONNX_SRC(), srv / "models/neuroscan_r30a_s123.onnx")
    (srv / "inference.py").write_text(INFERENCE_PY)
    pip = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--target", "/tmp/srvpkgs", "onnxruntime==1.20.1",
                          "numpy==2.1.3", "pillow==11.0.0"], capture_output=True, text=True)
    log(f"pip exit {pip.returncode}")
    (srv / "run_cases.py").write_text(RUNNER)
    env = dict(os.environ, PYTHONPATH=f"{srv}:/tmp/srvpkgs", OMP_NUM_THREADS="1")
    p = subprocess.run([sys.executable, str(srv / "run_cases.py"), str(CASE_DIR / "cases.json")], env=env, capture_output=True,
                       text=True, cwd=str(srv))
    log(p.stdout[-3000:])
    if p.returncode != 0:
        log(p.stderr[-3000:])
        raise SystemExit("served pipeline failed")
    res = json.loads((CASE_DIR / "results.json").read_text())
    # v2 (2026-10-17): keep the six cases' input slices and served overlays for the redrawn figure (no other change)
    shutil.copytree(CASE_DIR, OUTPUT_DIR / "fig9_cases", dirs_exist_ok=True)

    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tier_col = {"High alert": "#C0392B", "Moderate alert": "#E67E22", "Normal": "#2980B9", "Uncertain": "#7F8C8D",
                "Unreliable": "#7F8C8D"}
    fig, axes = plt.subplots(2, 6, figsize=(15, 6.2))
    for i, c in enumerate(res):
        ax_in, ax_ov = axes[i // 3, 2 * (i % 3)], axes[i // 3, 2 * (i % 3) + 1]
        ax_in.imshow(Image.open(Path(c["dir"]) / "main.png").convert("L").resize((224, 224)), cmap="gray")
        ax_in.set_title(f"{c['source']} ({c['modality']}{', DWI' if c['modality'] == 'MRI' else ''})\nreference: {c['label']}",
                        fontsize=9)
        ax_ov.imshow(Image.open(Path(c["dir"]) / "overlay.png"))
        pr = c["probs"]
        ax_ov.set_title(f"predicted: {c['prediction']}\nN {pr[0]:.2f} · I {pr[1]:.2f} · H {pr[2]:.2f}", fontsize=9)
        ax_ov.text(0.5, -0.06, c["tier"], transform=ax_ov.transAxes, ha="center", va="top", fontsize=9, color="white",
                   bbox=dict(boxstyle="round,pad=0.3", fc=tier_col[c["tier"]], ec="none"))
        for a in (ax_in, ax_ov):
            a.set_xticks([])
            a.set_yticks([])
    fig.tight_layout(h_pad=2.2)
    fig.savefig(OUTPUT_DIR / "fig3_prototype_output.png", dpi=200, bbox_inches="tight", facecolor="white")
    with open(OUTPUT_DIR / "fig3_cases.json", "w") as f:
        json.dump(res, f, indent=2)
    log("Done.")


SAVED = DS / "atik68/neuroscan-q-round30as123-preds"
PHASE2 = DS / "atik68/neuroscan-q-phase2-split-data"
CASE_DIR = Path("/tmp/fig3_cases")
SHORT = {"ISLES2022_dwi_adc": "ISLES 2022", "SOOP_dwi_adc": "SOOP", "AISD": "AISD", "SICH_CT": "SICH"}


def ONNX_SRC():
    for p in (Path("/kaggle/input/notebooks/atik68/neuroscan-q-onnx-export-check/output/neuroscan_r30a_s123.onnx"),
              Path("/kaggle/input/neuroscan-q-onnx-export-check/output/neuroscan_r30a_s123.onnx")):
        if p.exists():
            return p
    return next(Path("/kaggle/input").rglob("neuroscan_r30a_s123.onnx"))


RUNNER = r'''
import base64, io, json, sys
from pathlib import Path
from PIL import Image
import inference
pipe = inference.get_pipeline()
cases = json.loads(Path(sys.argv[1]).read_text())
def tier(pred, p):
    pn, pi, ph = p
    if pred == "hemorrhagic":
        return "High alert" if ph >= 0.55 else "Moderate alert"
    if pred == "ischemic":
        return "High alert" if pi >= 0.90 else ("Moderate alert" if pi >= 0.70 else "Uncertain")
    return "Normal" if pn >= 0.80 else "Uncertain"
out = []
for c in cases:
    d = Path(c["dir"])
    img = lambda n: Image.open(d / f"{n}.png") if (d / f"{n}.png").exists() else None
    r = pipe.predict(c["modality"], img("main"), prev=img("prev"), nxt=img("next"), adc=img("adc"))
    (d / "overlay.png").write_bytes(base64.b64decode(r["heatmap_png_base64"]))
    p = [r["prob_normal"], r["prob_ischemic"], r["prob_hemorrhagic"]]
    t = "Unreliable" if r["likely_not_a_scan"] else tier(r["prediction"], p)
    out.append(dict(c, prediction=r["prediction"], probs=[round(v, 4) for v in p], tier=t, pathway=r["pathway"],
                    likely_not_a_scan=r["likely_not_a_scan"], warnings=r["warnings"]))
    print(c["name"], r["prediction"], [round(v, 4) for v in p], "offline", c["offline_probs"], t, r["pathway"], flush=True)
(Path(sys.argv[1]).parent / "results.json").write_text(json.dumps(out))
'''


INFERENCE_PY = r'''"""
NeuroScan web demo -- shared inference logic (v2: the evaluated final model and its input pathway).

Model: the paper's final model, R30A s123 -- ConvNeXt-Tiny, one softmax over normal / ischemic / hemorrhagic --
served with ONNX Runtime (models/neuroscan_r30a_s123.onnx; SHA-256 checked at start-up). PyTorch is not needed at
run time: it alone took about 300 MB, and the PyTorch server peaked at 671 MB on Linux, above the 512 MB hosting tier;
with ONNX Runtime the peak is about 300 MB and a request takes about 0.85 s on one CPU thread (kernel
neuroscan-q-onnx-load-memory).
The ONNX graph was exported from the PyTorch weights and checked against them (kaggle kernel
neuroscan-q-onnx-export-check): it returns the logits and, for each class, the Grad-CAM map of ConvNeXt stage 4.
The Grad-CAM channel weights are computed inside the graph in closed form -- the head after stage 4 is global average
pooling, LayerNorm and one linear layer, so the gradient of a logit with respect to the stage-4 activations is the
same at every position and has an exact expression; the kernel checks it against autograd.

Input pathway -- the one the offline evaluation used, chosen per request from what was uploaded:
  * CT, current slice + previous + next  -> channels (previous, current, next)       ["ct_adjacent_slices", evaluated]
  * MRI, DWI + ADC of the same slice      -> channels (DWI, ADC, DWI)                 ["mri_dwi_adc", evaluated]
  * CT, one slice only                    -> channels (slice, slice, slice)          ["single_image", NOT the evaluated
    pathway; the response carries a warning]. Post-hoc experiment C2 (held-out CT test): hemorrhage recall unchanged
    (92.0%), ischemic recall 91.0% -> 68.8%, ischemic AUROC 0.972 -> 0.955.
  * MRI without ADC                       -> refused (400). C2: DWI alone or ADC alone, replicated, detected almost no
    ischemia (recall 0-3%, AUROC 0.32-0.51 on ISLES and SOOP); the model needs both sequences.
Every image is converted to 8-bit grayscale and resized to 224 x 224, exactly as in training.

Decision rule: 4-view test-time augmentation (identity, horizontal flip, +8 deg, -8 deg rotation), softmax averaged
over views, prediction = argmax (plain). This is the rule under which the paper evaluates the alert tiers (post-hoc
analysis B4): it alerts on more hemorrhage slices than the paper's class-weighted rule (92.0% vs 89.0% on the pooled
held-out test), at the cost of more false alerts, which is the safer trade-off for a triage aid. The displayed
probabilities are these same TTA-averaged softmax values (uncalibrated), as in B4.

MRI and hemorrhage: no hemorrhagic MRI source was available for training or evaluation, so an MRI request carries a
warning that hemorrhage on MRI is outside the evaluated scope.

Grad-CAM [Selvaraju et al., 2017]: identity view only, for the predicted class, overlaid on the current CT slice or
the DWI image; suppress below the image's 70th-percentile activation, rescale, bicubic upsample; outline at > 0.6 for
ischemic / hemorrhagic. Post-hoc experiment C3 evaluates this display rule against lesion masks. It shows where the
model's evidence is concentrated, not a lesion boundary.
"""
import base64
import hashlib
import io
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MODELS_DIR = Path(__file__).parent / "models"
MODEL_FILE = MODELS_DIR / "neuroscan_r30a_s123.onnx"
EXPECTED_SHA256 = "4937ab381ee3e7c29d5ab18ea761a8c19ddffb6488210fb6030511a8041502eb"
MODEL_VERSION = "r30a_s123_convnext_tiny_onnx"

TARGET_SIZE = (224, 224)
VALID_LABELS = ("normal", "ischemic", "hemorrhagic")
LABEL_TO_IDX = {"normal": 0, "ischemic": 1, "hemorrhagic": 2}
MODALITIES = ("CT", "MRI")

PATHWAY_CT = "ct_adjacent_slices"
PATHWAY_MRI = "mri_dwi_adc"
PATHWAY_SINGLE = "single_image"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


class InputError(ValueError):
    """The uploaded combination of images cannot be processed (shown to the user as a 400)."""


def _gray(image: Image.Image) -> Image.Image:
    return image.convert("L").resize(TARGET_SIZE)


def build_input(modality: str, main: Image.Image, prev: Image.Image | None = None, nxt: Image.Image | None = None,
                adc: Image.Image | None = None):
    """Returns (merged RGB image, pathway name, base image for the overlay)."""
    if modality not in MODALITIES:
        raise InputError("modality must be CT or MRI.")
    cur = _gray(main)
    if modality == "CT":
        if adc is not None:
            raise InputError("An ADC image applies to MRI only.")
        if (prev is None) != (nxt is None):
            raise InputError("For CT context, upload both the previous and the next slice (or neither).")
        if prev is not None:
            return Image.merge("RGB", (_gray(prev), cur, _gray(nxt))), PATHWAY_CT, cur
    else:
        if prev is not None or nxt is not None:
            raise InputError("Previous/next slices apply to CT only; for MRI upload DWI and ADC.")
        if adc is None:
            raise InputError("MRI needs both images of the same slice: the DWI image and its ADC map. With one "
                             "sequence alone the model does not detect ischemia.")
        return Image.merge("RGB", (cur, _gray(adc), cur)), PATHWAY_MRI, cur
    return Image.merge("RGB", (cur, cur, cur)), PATHWAY_SINGLE, cur


def _looks_like_grayscale_scan(image: Image.Image) -> bool:
    """Input-file check, not an anatomical or diagnostic-quality check: the model was trained only on grayscale
    CT/MRI slices and has no "none of the above" output, so this flags wrong-file uploads before a meaningless result
    reaches an alert. All must hold:
      1. near-grey: mean |R-G|, |G-B|, |R-B| <= 8 (colour photographs fail);
      2. dark surroundings: >= 50% of the outer border ring (6% of each side) is dark (<= 30), as around a head on a
         CT or MRI slice (documents, diagrams, screenshots and most photographs fail);
      3. non-black area (pixels > 10) between 0.5% and 70% of the image: an empty black slice fails, and so do
         photographs that fill the frame.
    The earlier rule ("near-grey and >= 30% of all pixels mid-tone") flagged 78% of genuine held-out scans, because
    the black background around the head counted against it, and accepted 80% of grayscale photographs. Evaluated in
    kernel neuroscan-q-input-check-eval (thresholds set on validation images): genuine non-blank held-out scans from
    all 7 sources flagged 0.26% (n = 3,520; CT slices, DWI and ADC), completely black slices flagged by design, and
    0 of 208 non-scan images accepted (photographs, documents, diagrams, screenshots; this set also informed the 70%
    limit). It cannot tell a brain scan from another dark-background medical image, and a scan cropped so that the
    head fills the frame is flagged."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if ((np.abs(r - g) + np.abs(g - b) + np.abs(r - b)) / 3.0).mean() > 8.0:
        return False
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    h, w = gray.shape
    bh, bw = max(1, int(0.06 * h)), max(1, int(0.06 * w))
    ring = np.ones((h, w), dtype=bool)
    ring[bh:h - bh, bw:w - bw] = False
    if (gray[ring] <= 30).mean() < 0.50:
        return False
    non_black = (gray > 10).mean()
    return bool(0.005 <= non_black <= 0.70)


def _tta_views(img: Image.Image):
    """Identical to torchvision's PIL path: hflip = FLIP_LEFT_RIGHT; rotate(angle, fill=0) = PIL rotate with
    nearest-neighbour resampling, no expansion, black fill."""
    yield img
    yield img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    yield img.rotate(8, resample=Image.Resampling.NEAREST, expand=False, fillcolor=(0, 0, 0))
    yield img.rotate(-8, resample=Image.Resampling.NEAREST, expand=False, fillcolor=(0, 0, 0))


def _to_array(img: Image.Image) -> np.ndarray:
    """torchvision to_tensor + ImageNet normalize, as float32 CHW."""
    a = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return (a - MEAN) / STD


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ThreeClassPipeline:
    """Loads the final model once; call .predict(...) per request."""

    def __init__(self):
        digest = _sha256(MODEL_FILE)
        if digest != EXPECTED_SHA256:
            raise RuntimeError(f"Model file checksum mismatch ({digest[:12]}...); refusing to serve.")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.enable_cpu_mem_arena = False  # lower resident memory between requests on the 512 MB tier
        self.session = ort.InferenceSession(str(MODEL_FILE), sess_options=opts, providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        self.epoch = int(meta["epoch"]) if "epoch" in meta else None
        self.val_acc = float(meta["val_acc"]) if "val_acc" in meta else None
        self.weights_sha256 = digest

    @staticmethod
    def _risk_tier_color(prediction: str, prob_normal: float, prob_ischemic: float, prob_hemorrhagic: float):
        """Same confidence tiers as webapp_php/includes/risk_tier.php (kept in sync by hand), so the heatmap tint
        always agrees with the badge."""
        if prediction == "hemorrhagic":
            if prob_hemorrhagic >= 0.55:
                return (192, 57, 43)
            return (230, 126, 34)
        if prediction == "ischemic":
            if prob_ischemic >= 0.90:
                return (192, 57, 43)
            if prob_ischemic >= 0.70:
                return (230, 126, 34)
            return (127, 140, 141)
        if prediction == "normal" and prob_normal >= 0.80:
            return (41, 128, 185)
        return (127, 140, 141)

    @staticmethod
    def _boundary_mask(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
        eroded = mask.copy()
        eroded[1:, :] &= mask[:-1, :]
        eroded[:-1, :] &= mask[1:, :]
        eroded[:, 1:] &= mask[:, :-1]
        eroded[:, :-1] &= mask[:, 1:]
        boundary = mask & ~eroded
        for _ in range(thickness - 1):
            dilated = boundary.copy()
            dilated[1:, :] |= boundary[:-1, :]
            dilated[:-1, :] |= boundary[1:, :]
            dilated[:, 1:] |= boundary[:, :-1]
            dilated[:, :-1] |= boundary[:, 1:]
            boundary = dilated
        return boundary

    @staticmethod
    def _display_cam(cam7: np.ndarray) -> Image.Image:
        """Prototype display rule on the 7 x 7 Grad-CAM (already ReLU'd): max-normalise, suppress below the 70th
        percentile, rescale, bicubic upsample to 224."""
        cam = cam7 / (cam7.max() + 1e-8)
        threshold = np.percentile(cam, 70)
        cam = np.clip((cam - threshold) / (1 - threshold + 1e-8), 0, 1)
        return Image.fromarray((cam * 255).astype(np.uint8)).resize(TARGET_SIZE, Image.BICUBIC)

    @staticmethod
    def _overlay_heatmap(base_image: Image.Image, cam_img: Image.Image, color, draw_boundary: bool) -> Image.Image:
        base_arr = np.asarray(base_image.convert("RGB"), dtype=np.float32)
        cam_arr = np.asarray(cam_img, dtype=np.float32) / 255.0
        tint = np.asarray(color, dtype=np.float32)
        alpha = (cam_arr * 0.75)[..., None]
        blended = base_arr * (1 - alpha) + tint * alpha
        if draw_boundary:
            boundary = ThreeClassPipeline._boundary_mask(cam_arr > 0.6, thickness=2)
            blended[boundary] = tint
        return Image.fromarray(blended.astype(np.uint8))

    def predict(self, modality: str, image: Image.Image, prev: Image.Image | None = None,
                nxt: Image.Image | None = None, adc: Image.Image | None = None) -> dict:
        merged, pathway, base = build_input(modality, image, prev, nxt, adc)
        x = np.stack([_to_array(v) for v in _tta_views(merged)])
        logits, cams = self.session.run(["logits", "gradcam"], {"image": x})
        probs = _softmax(logits.astype(np.float64)).mean(axis=0)
        idx = int(np.argmax(probs))
        prediction = VALID_LABELS[idx]
        prob_normal, prob_ischemic, prob_hemorrhagic = (float(v) for v in probs)

        uploads = [u for u in (image, prev, nxt, adc) if u is not None]
        try:
            likely_not_a_scan = not all(_looks_like_grayscale_scan(u) for u in uploads)
        except Exception:
            likely_not_a_scan = False

        warnings = []
        if pathway == PATHWAY_SINGLE:
            warnings.append("Single CT slice only: this is not the evaluated input pathway and misses more ischemic "
                            "strokes. Upload the previous and next slices as well.")
        if modality == "MRI":
            warnings.append("MRI: hemorrhage was not represented in training or evaluation; MRI results concern "
                            "normal versus ischemic only.")
        result = {
            "prediction": prediction,
            "prob_normal": prob_normal,
            "prob_ischemic": prob_ischemic,
            "prob_hemorrhagic": prob_hemorrhagic,
            "modality": modality,
            "pathway": pathway,
            "evaluated_pathway": pathway != PATHWAY_SINGLE,
            "model_version": MODEL_VERSION,
            "likely_not_a_scan": likely_not_a_scan,
            "warnings": warnings,
            "decision_note": None,
        }
        try:
            color = self._risk_tier_color(prediction, prob_normal, prob_ischemic, prob_hemorrhagic)
            overlay = self._overlay_heatmap(base, self._display_cam(cams[0, idx]), color,
                                            prediction in ("ischemic", "hemorrhagic"))
            buf = io.BytesIO()
            overlay.save(buf, format="PNG")
            result["heatmap_png_base64"] = base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            pass
        return result


_pipeline: ThreeClassPipeline | None = None


def get_pipeline() -> ThreeClassPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ThreeClassPipeline()
    return _pipeline
'''


RSNA_MAP = {}


if __name__ == "__main__":
    main()
