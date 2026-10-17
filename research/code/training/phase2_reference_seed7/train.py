"""
NeuroScan Phase Q -- Development Phase 2 (pre-registered 2026-10-04), seed 7, office PC.
Phase 2 REFERENCE: ConvNeXt-Tiny exactly as Round 30A, re-trained on the Phase 2 split.
Phase 2 protocol: fresh validation set val2 (12% of former training patients per dataset, never evaluated; from
neuroscan-q-phase2-split), former validation rows moved into training; checkpoint selection and early stopping on
val2 macro one-vs-rest AUROC (single view), patience 15, 45 epochs. Everything else as Round 30A (ConvNeXt-Tiny
ImageNet weights, Adam 1e-4 / wd 1e-4, cosine, batch 64 as 2 x 32 exact micro-batches, augmentation, class weights,
TTA). Predictions: val (= val2) ONLY, as p_*_round30bs7; no test / SOOP hold-out inference and no Round 24 re-scoring in Phase 2 development runs (expert advice 2026-10-04).
"""
import gzip
import json
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

SEED = 7
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
OWN_DIR = NB / "atik68/neuroscan-q-round30b-phase2ref-s7-office"
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

    def __init__(self, main_root, supp_index, cpaisd_index=None, rsna_dicom=True, center=False):
        self.main_root, self.supp_index, self.cpaisd_index = main_root, supp_index, cpaisd_index
        self.center = center
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
        img = self._merged(row)
        return center_brain(img) if self.center else img

    def _merged(self, row):
        if row["dataset"] in MRI_PAIR_DATASETS:
            dwi, adc = self._slice(row["dwi_out_path"]), self._slice(row["adc_out_path"])
            return Image.merge("RGB", (dwi, adc, dwi))
        s = self._slice(row["out_path"])
        if bool(row["has_context"]):
            return Image.merge("RGB", (self._slice(row["prev_out_path"]), s, self._slice(row["next_out_path"])))
        return Image.merge("RGB", (s, s, s))


# ---------------------------------------------------------------- Phase 2 split + checkpoint metric
PHASE2 = DS / "atik68/neuroscan-q-phase2-split-data"
P2_CPAISD_RE = re.compile(r"/([\d.]{10,})/(\d+)/raw\.dcm$")
P2_MRI_RE = re.compile(r"^(sub-[\w]+)/(ses-\w+)/")


def p2_need(p):
    p = Path(p)
    if p.exists():
        return p
    for cand in [INPUT_ROOT / p.parent.name, *INPUT_ROOT.glob(f"*/{p.parent.name}"), *INPUT_ROOT.glob(f"*/*/{p.parent.name}")]:
        if (cand / p.name).exists():
            return cand / p.name
    return need(p)


def patient_key(ds, sk, op, subject, rsna_map):
    """Identical to neuroscan-q-phase2-split/split.py."""
    sk, op = str(sk), str(op)
    if ds == "AISD":
        return f"AISD:{sk.split('/')[2]}"
    if ds == "CPAISD_v2":
        m = P2_CPAISD_RE.search(sk.replace("\\", "/"))
        return f"CPAISD:{m.group(1)}" if m else f"CPAISD_row:{op}"
    if ds == "ISLES2022_dwi_adc":
        m = P2_MRI_RE.match(sk)
        return f"ISLES:{m.group(1)}/{m.group(2)}" if m else f"ISLES_row:{sk}"
    if ds == "SOOP_dwi_adc":
        return f"SOOP:{subject}" if isinstance(subject, str) and subject else f"SOOP_row:{sk}"
    if ds == "RSNA_ICH":
        stem = Path(op).stem
        pid = rsna_map.get(stem)
        return f"RSNA:{pid}" if isinstance(pid, str) and pid else f"RSNA_row:{stem}"
    if ds == "SICH_CT":
        return f"SICH:{sk.replace(chr(92), '/').split('/')[-2]}"
    return f"{ds}_row:{op}"


def apply_phase2(train, val_old, report):
    """val2 = training rows of the pre-registered Phase 2 patients; old validation rows join training."""
    v2 = pd.read_csv(p2_need(PHASE2 / "phase2_val2_patients.csv"))
    rmap = pd.read_csv(p2_need(PHASE2 / "rsna_patient_map.csv"))
    rsna_map = dict(zip(rmap["stem"].astype(str), rmap["PatientID"]))
    subj = train["subject"] if "subject" in train.columns else pd.Series([None] * len(train))
    keys = [patient_key(a, b, c, s, rsna_map) for a, b, c, s in zip(train["dataset"], train["source_key"], train["out_path"], subj)]
    in_v2 = pd.Series(keys).isin(set(v2["patient_key"])).to_numpy()
    val2 = train[in_v2].reset_index(drop=True)
    new_train = pd.concat([train[~in_v2], val_old], ignore_index=True)
    report["phase2"] = {"val2_rows": int(in_v2.sum()), "val2_patients_listed": int(len(v2)),
                        "val2_by_dataset": val2["dataset"].value_counts().to_dict(),
                        "val2_by_label": val2["label"].value_counts().to_dict(),
                        "new_train_rows": int(len(new_train)), "old_val_rows_moved_to_train": int(len(val_old))}
    log(f"PHASE 2 SPLIT: {report['phase2']}")
    return new_train, val2


def macro_auroc(y, p):
    return float(np.mean([roc_auc_score((y == c).astype(int), p[:, c]) for c in range(len(VALID_LABELS))]))


@torch.no_grad()
def evaluate(model, loader, criterion):
    """Single-view validation pass: loss, accuracy and macro one-vs-rest AUROC (the Phase 2 checkpoint metric)."""
    model.eval()
    tot, n, ys, ps = 0.0, 0, [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        tot += criterion(out, y).item() * x.size(0)
        n += y.size(0)
        ys.append(y.cpu().numpy())
        ps.append(torch.softmax(out, 1).cpu().numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return tot / n, float((p.argmax(1) == y).mean()), macro_auroc(y, p)


def center_brain(img):  # not used in R30B
    return img


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
    train = pd.concat([splits["train"], soop[soop["split"] == "train"].reindex(columns=cols + ["subject"]),
                       soop_to_train.reindex(columns=cols + ["subject"]), rsna_extra.reindex(columns=cols)], ignore_index=True)
    val_all = pd.concat([splits["val"], soop[soop["split"] == "val"].reindex(columns=cols + ["subject", "lesion_px"])],
                        ignore_index=True)
    rsna_ids = pd.concat([d.loc[d["dataset"] == "RSNA_ICH", "out_path"] for d in (train, val_all, splits["test"])])
    report["data"] = {"train_rows": int(len(train)), "val_rows": int(len(val_all)),
                      "train_by_dataset": train["dataset"].value_counts().to_dict(),
                      "train_by_label": train["label"].value_counts().to_dict(),
                      "soop_holdout_patients": len(keep), "soop_holdout_rows": int(len(soop_hold)),
                      "soop_moved_to_train_patients": int(soop_to_train["subject"].nunique()),
                      "rsna_extra_rows": int(len(rsna_extra)), "rsna_rows_all_splits": int(len(rsna_ids))}
    log(f"Data (Phase 1 layout): {report['data']}")
    train, val_all = apply_phase2(train, val_all, report)

    for r in SUPP_ROOTS:
        SCANS[r] = scan(r, {"Kaggle_CT", "hemorrhagic", "ischemic"})
    main_root, supp = find_images_root(), build_supplementary_index()
    cp_sample = base["train"].loc[base["train"]["dataset"] == "CPAISD_v2", "out_path"].iloc[0]
    cp_index = build_cpaisd_v2_index(Path(cp_sample).name)
    loader = Loader(main_root, supp, cp_index, rsna_dicom=False)  # Round 24 input (old 128 px RSNA images)
    loader_r24 = loader

    g = torch.Generator()
    g.manual_seed(SEED)
    wk = {"num_workers": NUM_WORKERS, "persistent_workers": NUM_WORKERS > 0, "pin_memory": DEVICE.type == "cuda"}
    if NUM_WORKERS > 0:
        wk["prefetch_factor"] = 4
    train_loader = DataLoader(TrainDataset(train, loader, train_transform), batch_size=BATCH_SIZE, shuffle=True,
                              generator=g, **wk)
    val_loader = DataLoader(TrainDataset(val_all, loader, eval_transform), batch_size=BATCH_SIZE, shuffle=False, **wk)
    log(f"DataLoader workers: {NUM_WORKERS}")

    model = build_model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=LR * 0.01)
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(train).to(DEVICE))

    start_epoch, best_val_auc, since = 0, -1.0, 0
    own = OWN_DIR / "checkpoints/latest.pt"
    if own.exists():
        ck = torch.load(own, map_location=DEVICE)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        scheduler.load_state_dict(ck["scheduler_state"])
        start_epoch, best_val_auc, since = ck["epoch"] + 1, ck["best_val_auc"], ck["epochs_since_improvement"]
        log(f"Resumed at epoch {start_epoch}")

    log(f"Training starts (ConvNeXt-Tiny, micro-batch {MICRO_BATCH} x {BATCH_SIZE // MICRO_BATCH})")
    for epoch in range(start_epoch, TOTAL_EPOCHS):
        t0 = time.time()
        tl, ta = run_epoch(model, train_loader, criterion, optimizer)
        if epoch == start_epoch and DEVICE.type == "cuda":
            log(f"GPU peak memory after first epoch: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        vl, va, vauc = evaluate(model, val_loader, criterion)
        scheduler.step()
        improved = vauc > best_val_auc
        best_val_auc, since = (vauc, 0) if improved else (best_val_auc, since + 1)
        log(f"Epoch {epoch + 1}/{TOTAL_EPOCHS} | train_loss={tl:.4f} train_acc={ta:.4f} | val_loss={vl:.4f} "
            f"val_acc={va:.4f} val_mAUROC={vauc:.4f} | best_mAUROC={best_val_auc:.4f} "
            f"({'improved' if improved else f'no improvement x{since}'}) | {time.time() - t0:.0f}s")
        ck = {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
              "scheduler_state": scheduler.state_dict(), "val_acc": va, "val_auc": vauc, "best_val_auc": best_val_auc,
              "epochs_since_improvement": since}
        torch.save(ck, CKPT_DIR / "latest.pt")
        if improved:
            torch.save(ck, CKPT_DIR / "best.pt")
        if since >= EARLY_STOP_PATIENCE:
            log(f"Early stopping (best_val_mAUROC={best_val_auc:.4f})")
            break

    best = torch.load(CKPT_DIR / "best.pt", map_location=DEVICE)
    model.load_state_dict(best["model_state"])
    both = {"round30bs7": model}
    loaders = {"round30bs7": loader}

    full_test = splits["test"]
    ext = soop_hold

    pv = predict(both, val_all, loaders, "val")
    keep_cols = ["source_key", "dataset", "label", "out_path", "subject", "slice", "lesion_px"]
    for name, df, pr in (("val", val_all, pv),):
        out = df.reindex(columns=keep_cols).copy()
        for k, p in pr.items():
            for c, lab in enumerate(VALID_LABELS):
                out[f"p_{lab}_{k}"] = p[:, c].round(6)
        out.to_csv(OUTPUT_DIR / f"predictions_{name}.csv", index=False)
    log("Saved predictions_val.csv (Phase 2: no test / SOOP hold-out inference)")

    is_cp_v = (val_all["dataset"] == "CPAISD_v2").to_numpy()
    is_soop_v = (val_all["dataset"] == "SOOP_dwi_adc").to_numpy()
    is_isles_v = (val_all["dataset"] == "ISLES2022_dwi_adc").to_numpy()
    is_rsna_v = (val_all["dataset"] == "RSNA_ICH").to_numpy()
    va_area = isles_areas(val_all)

    # TEST FREEZE (2026-09-30): test and SOOP hold-out probabilities are saved above for the single final
    # evaluation, but no test / hold-out metrics are computed or logged in development runs.
    res = {"condition": "r30b_phase2_seed7_office_pc_rtx4060ti_6_workers",
           "checkpoint_epoch": best["epoch"], "checkpoint_val_acc_val2": best["val_acc"], "checkpoint_val_mAUROC_val2": best["val_auc"], "startup": report,
           "VALIDATION (decision set)": {},
           "TEST and SOOP hold-out": "frozen -- no inference in Phase 2 development runs"}
    V = res["VALIDATION (decision set)"]
    for k in both:
        V[f"full_val_incl_cpaisd_{k}"] = score(val_all, pv[k])
        V[f"cpaisd_val_{k}"] = score(val_all[is_cp_v], pv[k][is_cp_v])
        V[f"isles_only_{k}"] = score(val_all[is_isles_v], pv[k][is_isles_v])
        V[f"soop_val_only_{k}"] = score(val_all[is_soop_v], pv[k][is_soop_v])
        V[f"rsna_only_{k}"] = score(val_all[is_rsna_v], pv[k][is_rsna_v])
        V[f"per_source_{k}"] = per_source(val_all, pv[k])
        V[f"isles_size_{k}"] = size_table(val_all, pv[k], va_area)
    res["note"] = ("round30bs7 = R30B (Phase 2), seed 7, office PC; VALIDATION = val2. round24 = the original "
                   "Round 24 is not used in Phase 2.")
    with open(OUTPUT_DIR / "eval_metrics.json", "w") as f:
        json.dump(res, f, indent=2, default=str)
    for k in both:
        for name in (f"full_val_incl_cpaisd_{k}", f"rsna_only_{k}", f"isles_only_{k}", f"cpaisd_val_{k}"):
            r = V[name]
            log(f"VAL {name}: acc={r['accuracy']:.4f} isch R/P={r['ischemic_recall']:.4f}/{r['ischemic_precision']:.4f} "
                f"hem R/P={r['hemorrhagic_recall']:.4f}/{r['hemorrhagic_precision']:.4f} "
                f"isch AUROC={r.get('ischemic_auroc', float('nan')):.4f} hem AUROC={r.get('hemorrhagic_auroc', float('nan')):.4f}")
    log("Done.")


if __name__ == "__main__":
    main()
