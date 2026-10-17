"""
NeuroScan -- post-hoc round 1, C3 re-run for CPAISD only (2026-10-07). Identical to neuroscan-q-c2-c3-gradcam v1
except: (1) C2 is skipped; the locked-rule predictions are taken from the saved final-evaluation probabilities, which
v1 reproduced exactly (decision agreement 1.0); (2) only CPAISD lesion slices; (3) the mask orientation is fixed to
identity. In v1 the automatic orientation rule chose rot270 for CPAISD, which placed 6% of lesion pixels outside the
head (identity: 0%) with a near-identical contrast score; identity is the native DICOM pixel grid of image and mask.
All 8 orientation scores are still computed and logged. Everything else (Grad-CAM layers, measures, deletion test,
randomisation check, bootstrap) is unchanged.
"""
import gzip
import hashlib
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
    # C2 + C3 (post-hoc round 1). Everything above this line is the final model's own data pipeline, unchanged.
    # ================================================================================================================
    rmap = pd.read_csv(need(PHASE2 / "rsna_patient_map.csv"))
    RSNA_MAP.update(dict(zip(rmap["stem"].astype(str), rmap["PatientID"])))
    model = build_model(pretrained=False).to(DEVICE)
    wf = need(WEIGHTS)
    ck = torch.load(wf, map_location=DEVICE)
    model.load_state_dict(ck["model_state"])
    model.eval()
    R = {"note": "C2 (single-image pathway) and C3 (Grad-CAM vs lesion masks) for the final model R30A s123; "
                 "descriptive analyses on held-out rows; the model, its weights and its decision rule are unchanged",
         "weights_file_sha256": sha256(wf), "checkpoint_epoch_index": ck.get("epoch"), "checkpoint_val_acc": ck.get("val_acc")}
    log(f"Final model weights loaded: epoch index {ck.get('epoch')}, val_acc {ck.get('val_acc'):.4f}, sha256 {R['weights_file_sha256'][:12]}")

    test = splits["test"].copy()
    test["pk"] = [patient_key(d, s, o, u) for d, s, o, u in
                  zip(test["dataset"], test["source_key"], test["out_path"], test.get("subject", pd.Series([None] * len(test))))]
    soop_h = soop_hold.copy()
    soop_h["pk"] = [f"SOOP:{s}" for s in soop_h["subject"]]
    is_mri = test["dataset"].isin(MRI_PAIR_DATASETS)
    ct_rows, mri_rows = test[~is_mri].reset_index(drop=True), pd.concat([test[is_mri], soop_h], ignore_index=True)
    mri_rows["cohort"] = np.where(mri_rows["dataset"] == "SOOP_dwi_adc", "C_soop", "A_isles")
    log(f"C2 rows: CT {len(ct_rows)} (has_context {int(ct_rows['has_context'].astype(bool).sum())}), MRI {len(mri_rows)}")

    # ---- C3: Grad-CAM against lesion masks (evaluated full-input pathway, reference class = ischemic) ----
    t0 = time.time()
    aisd_m = aisd_masks()
    log(f"AISD masks: {len(aisd_m)} files ({time.time() - t0:.0f}s)")
    les = test[(test["label"] == "ischemic") & (test["dataset"] == "CPAISD_v2")].copy()
    saved = pd.read_csv(need(SAVED / "predictions_test.csv"))
    full_pred = dict(zip(saved["source_key"].astype(str) + "|" + saved["out_path"].astype(str),
                         decide(saved[[f"p_{l}_round30as123" for l in VALID_LABELS]].to_numpy(float), LOCKED_W)))
    les["pred_locked"] = [full_pred[f"{s}|{o}"] for s, o in zip(les["source_key"].astype(str), les["out_path"].astype(str))]
    masks, isles_cache = [], {}
    for _, r in les.iterrows():
        masks.append(lesion_mask(r, aisd_m, isles_cache))
    les["has_mask"] = [m is not None and m.any() for m in masks]
    les = les.assign(_m=masks)
    R["C3_mask_coverage"] = {ds: [int(g["has_mask"].sum()), int(len(g))] for ds, g in les.groupby("dataset")}
    log(f"C3 lesion slices with a non-empty mask: {R['C3_mask_coverage']}")
    les = les[les["has_mask"]].reset_index(drop=True)

    # orientation check: image-content criterion only (never the model), per dataset, on up to 150 slices
    R["C3_orientation"], orient = {}, {}
    for ds, g in les.groupby("dataset"):
        g = g.sample(n=min(150, len(g)), random_state=0)
        sc = {}
        for t in range(8):
            v = []
            for _, r in g.iterrows():
                img = gray_input(loader, r)
                m = dihedral(r["_m"], t)
                head = img > 10
                if m.sum() == 0 or head.sum() == 0:
                    continue
                inside = (m & head).sum() / m.sum()
                contrast = (img[m].mean() - img[head].mean()) / (img[head].std() + 1e-6)
                v.append((inside, contrast))
            v = np.array(v)
            sc[t] = {"inside_head": float(v[:, 0].mean()), "contrast_z": float(v[:, 1].mean())}
        sign = 1.0 if ds == "ISLES2022_dwi_adc" else -1.0   # DWI lesions hyperintense; NCCT lesions hypodense
        ok = [t for t in sc if sc[t]["inside_head"] >= 0.9 * max(s["inside_head"] for s in sc.values())]
        best = 0  # fixed to identity for this re-run (see docstring); v1 rule would choose:
        R.setdefault("C3_orientation_v1_rule_would_choose", {})[ds] = DIHEDRAL_NAMES[max(ok, key=lambda t: sign * sc[t]["contrast_z"])]
        orient[ds] = best
        R["C3_orientation"][ds] = {"scores_by_transform": sc, "chosen": best, "transform_names": DIHEDRAL_NAMES}
        log(f"C3 orientation {ds}: chosen {DIHEDRAL_NAMES[best]}; " + "; ".join(
            f"{DIHEDRAL_NAMES[t]} in={s['inside_head']:.3f} z={s['contrast_z']:+.3f}" for t, s in sc.items()))
    les["_m"] = [dihedral(m, orient[ds]) for m, ds in zip(les["_m"], les["dataset"])]

    cams = {lay: GradCAM(model, lay) for lay in ("stage4", "stage3")}
    rows = []
    for start in range(0, len(les), 32):
        chunk = les.iloc[start:start + 32]
        x = torch.stack([eval_transform(loader.merged(r)) for _, r in chunk.iterrows()]).to(DEVICE)
        cm = {lay: c(x, [1] * len(chunk)) for lay, c in cams.items()}
        raw4 = cams["stage4"].last_raw
        for i, (_, r) in enumerate(chunk.iterrows()):
            img = gray_input(loader, r)
            rec = {"dataset": r["dataset"], "pk": r["pk"], "correct": int(r["pred_locked"] == 1), "lesion_px_224": int(r["_m"].sum())}
            for lay in cams:
                rec.update({f"{lay}_{k}": v for k, v in localisation(cm[lay][i], r["_m"], img > 10).items()})
            rec.update({f"deployed_display_{k}": v for k, v in deployed_region(raw4[i], r["_m"]).items()})
            rows.append(rec)
        if (start // 32) % 20 == 0:
            log(f"  [C3 cam] {min(start + 32, len(les))}/{len(les)}")
    loc = pd.DataFrame(rows)
    loc.to_csv(OUTPUT_DIR / "c3_cpaisd_localisation_per_slice.csv", index=False)
    R["C3_localisation"] = summarise_loc(loc)
    for k, v in R["C3_localisation"].items():
        log(f"C3 {k}: {json.dumps(jsonable(v))}")

    # faithfulness: deletion of top-CAM vs random vs mask pixels; model-randomisation sanity check
    R["C3_faithfulness"] = faithfulness(model, cams["stage4"], loader, les)
    for k, v in R["C3_faithfulness"].items():
        log(f"C3 faithfulness {k}: {json.dumps(jsonable(v))}")
    R["C3_randomisation_check"] = randomisation_check(loader, les, ck["model_state"])
    log(f"C3 randomisation check: {json.dumps(jsonable(R['C3_randomisation_check']))}")
    save_previews(loader, les, cams["stage4"], orient)

    with open(OUTPUT_DIR / "c3_cpaisd_identity_results.json", "w") as f:
        json.dump(jsonable(R), f, indent=2)
    log("Done.")


# ====================================================================================================================
# C2 / C3 helpers
# ====================================================================================================================
WEIGHTS = DS / "atik68/neuroscan-q-r30a-s123-weights/r30a_s123_weights.pt"
SAVED = DS / "atik68/neuroscan-q-round30as123-preds"
PHASE2 = DS / "atik68/neuroscan-q-phase2-split-data"
CPAISD_RAW = DS / "atik68/neuroscan-cpaisd"
AISD_MASK_ZIP_ID = "1d08fFpEvK4D6YTKfRlNuv_OlIxigZxl6"
LOCKED_W = (0.574, 0.574)
N_BOOT = 2000
FAITH_N = 300
DEL_FRACS = (0.05, 0.10, 0.20)
CPAISD_RE = re.compile(r"/([\d.]{10,})/(\d+)/raw\.dcm$")
MRI_RE = re.compile(r"^(sub-[\w]+)/(ses-\w+)/")
DIHEDRAL_NAMES = ["identity", "rot90", "rot180", "rot270", "hflip", "hflip_rot90", "hflip_rot180", "hflip_rot270"]
KEYS = ["accuracy", "balanced_accuracy", "macro_f1_present_classes", "normal_recall", "ischemic_recall", "ischemic_precision",
        "hemorrhagic_recall", "hemorrhagic_precision", "ischemic_auroc", "hemorrhagic_auroc", "macro_ovr_auroc"]
RSNA_MAP = {}
C3_TMP = Path("/tmp/c3")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, float) and o != o:
        return None
    return o


def patient_key(ds, sk, op, subject):
    sk, op = str(sk), str(op)
    if ds == "AISD":
        return f"AISD:{sk.split('/')[2]}"
    if ds == "CPAISD_v2":
        m = CPAISD_RE.search(sk.replace("\\", "/"))
        return f"CPAISD:{m.group(1)}" if m else f"CPAISD_row:{op}"
    if ds == "ISLES2022_dwi_adc":
        m = MRI_RE.match(sk)
        return f"ISLES:{m.group(1)}/{m.group(2)}" if m else f"ISLES_row:{sk}"
    if ds == "SOOP_dwi_adc":
        if isinstance(subject, str) and subject:
            return f"SOOP:{subject}"
        m = MRI_RE.match(sk)
        return f"SOOP:{m.group(1)}" if m else f"SOOP_row:{sk}"
    if ds == "RSNA_ICH":
        stem = Path(op).stem
        pid = RSNA_MAP.get(stem)
        return f"RSNA:{pid}" if isinstance(pid, str) and pid else f"RSNA_row:{stem}"
    if ds == "SICH_CT":
        return f"SICH:{sk.replace(chr(92), '/').split('/')[-2]}"
    return f"{ds}_row:{op}"


def decide(p, w):
    return (p * np.array([1.0, w[0], w[1]])).argmax(1)


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


def paired_probs(df, pa, pb, wa, wb, seed):
    """a = evaluated pathway, b = single-image pathway; paired patient-cluster bootstrap of (b - a)."""
    y = df["label"].map(LABEL_TO_IDX).to_numpy()
    da, db = decide(pa, wa), decide(pb, wb)
    ma, mb = slice_metrics(y, da, pa), slice_metrics(y, db, pb)
    groups, rng = clusters(df), np.random.default_rng(seed)
    diffs = {k: [] for k in KEYS}
    for _ in range(N_BOOT):
        idx = boot_idx(groups, rng)
        a, b = slice_metrics(y[idx], da[idx], pa[idx]), slice_metrics(y[idx], db[idx], pb[idx])
        for k in KEYS:
            if k in a and k in b and a[k] == a[k] and b[k] == b[k]:
                diffs[k].append(b[k] - a[k])
    out = {"n": int(len(df)), "patients_or_clusters": int(df["pk"].nunique()), "weights_a": list(wa), "weights_b": list(wb),
           "confusion_a": ma["confusion_matrix_rows_true_cols_pred"], "confusion_b": mb["confusion_matrix_rows_true_cols_pred"],
           "decision_agreement": float((da == db).mean())}
    for k in KEYS:
        if k in ma and k in mb and len(diffs[k]) > 50:
            v = np.array(diffs[k])
            out[k] = {"a": ma[k], "b": mb[k], "diff_b_minus_a": mb[k] - ma[k],
                      "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
    return out


def per_source_pair(df, pa, pb, wa, wb):
    out = {}
    for ds, g in df.groupby("dataset"):
        ix = g.index.to_numpy()
        y = g["label"].map(LABEL_TO_IDX).to_numpy()
        a, b = slice_metrics(y, decide(pa[ix], wa), pa[ix]), slice_metrics(y, decide(pb[ix], wb), pb[ix])
        out[ds] = {"n": int(len(g)), **{k: {"a": a[k], "b": b[k]} for k in KEYS if k in a and k in b}}
    return out


def show(tag, r):
    log(f"{tag}: n={r['n']} clusters={r['patients_or_clusters']} agree={r['decision_agreement']:.3f} " + "; ".join(
        f"{k} {r[k]['a']:.4f}->{r[k]['b']:.4f} d={r[k]['diff_b_minus_a']:+.4f} [{r[k]['ci95'][0]:+.4f},{r[k]['ci95'][1]:+.4f}]"
        for k in KEYS if k in r))


def build_input(loader, row, mode):
    """full = the evaluated pathway (CT prev/self/next; MRI DWI/ADC/DWI); single = the deployed prototype's pathway for
    one uploaded image (grayscale replicated); dwi / adc = an MRI upload of one sequence, replicated."""
    if mode == "full":
        return loader.merged(row)
    if mode == "single":
        s = loader._slice(row["out_path"])
    elif mode == "dwi":
        s = loader._slice(row["dwi_out_path"])
    else:
        s = loader._slice(row["adc_out_path"])
    return Image.merge("RGB", (s, s, s))


class ModeDataset(Dataset):
    def __init__(self, df, loader, mode):
        self.df, self.loader, self.mode = df.reset_index(drop=True), loader, mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        img = build_input(self.loader, self.df.iloc[i], self.mode)
        views = (img, transforms.functional.hflip(img), transforms.functional.rotate(img, 8, fill=0),
                 transforms.functional.rotate(img, -8, fill=0))
        return torch.stack([eval_transform(v) for v in views])


@torch.no_grad()
def predict_mode(model, df, loader, mode, tag):
    """Same 4-view TTA as predict(); DataLoader workers only parallelise image reading."""
    dl = DataLoader(ModeDataset(df, loader, mode), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    out, t0 = [], time.time()
    for x in dl:
        b = x.shape[0]
        out.append(torch.softmax(model(x.view(-1, 3, *TARGET_SIZE).to(DEVICE)), 1).view(b, 4, -1).mean(1).cpu().numpy())
    log(f"  [{tag}] {len(df)} rows ({time.time() - t0:.0f}s)")
    return np.concatenate(out)


def aisd_masks():
    ensure_pkg("gdown")
    import gdown
    C3_TMP.mkdir(parents=True, exist_ok=True)
    z = C3_TMP / "mask.zip"
    gdown.download(f"https://drive.google.com/uc?id={AISD_MASK_ZIP_ID}", str(z), quiet=True)
    out = {}
    with zipfile.ZipFile(z) as zf:
        zf.extractall(C3_TMP / "m")
    for f in (C3_TMP / "m").rglob("*.png"):
        if f.stem.isdigit():
            out[(f.parent.name, int(f.stem))] = f
    return out


def to224(a):
    return np.asarray(Image.fromarray((a > 0).astype(np.uint8) * 255).resize(TARGET_SIZE, Image.NEAREST)) > 0


def lesion_mask(r, aisd_m, isles_cache):
    """Binary lesion mask on the 224 x 224 input grid (before the orientation check), or None."""
    ds, sk = r["dataset"], str(r["source_key"])
    if ds == "AISD":
        f = aisd_m.get((sk.split("/")[2], int(re.sub(r"\D", "", Path(sk).stem) or -1)))
        if f is None:
            return None
        a = np.asarray(Image.open(f))
        a = a[..., 0] if a.ndim == 3 else a
        return to224((a == 2) | (a == 3))
    if ds == "CPAISD_v2":
        cands = [Path(sk), Path(sk.replace("/kaggle/input/datasets/atik68/", "/kaggle/input/")),
                 Path(sk.replace("/kaggle/input/", "/kaggle/input/datasets/atik68/", 1))]
        mp = next((c.parent / "mask.npz" for c in cands if (c.parent / "mask.npz").exists()), None)
        if mp is None:
            return None
        with np.load(mp) as d:
            a = np.asarray(d["mask"] if "mask" in d.files else d[d.files[0]])
        if a.ndim == 3 and a.shape[-1] <= 4 and set(np.unique(a)) <= {0, 1}:
            a = a.argmax(-1)
        return to224(a > 0)
    m = ISLES_KEY_RE.match(sk)
    if not m:
        return None
    pat, s = (m.group(1), m.group(2)), int(m.group(3))
    if pat not in isles_cache:
        deriv = scan(ISLES_RAW, {"derivatives"}, max_depth=3, stop_when_found=True)[0]["derivatives"]
        img = load_nifti(deriv[0] / pat[0] / pat[1], f"{pat[0]}_{pat[1]}_msk") if deriv else None
        isles_cache[pat] = img.get_fdata(dtype=np.float32) if img is not None else None
    a = isles_cache[pat]
    if a is None or not 0 <= s < a.shape[2]:
        return None
    return to224(a[:, :, s] > 0)


def dihedral(m, t):
    m = np.fliplr(m) if t >= 4 else m
    return np.ascontiguousarray(np.rot90(m, t % 4))


def gray_input(loader, r):
    """The grayscale image the mask is drawn on: CT centre slice, MRI DWI (224 x 224, float)."""
    p = r["dwi_out_path"] if r["dataset"] in MRI_PAIR_DATASETS else r["out_path"]
    return np.asarray(loader._slice(p), dtype=np.float32)


class GradCAM:
    """Grad-CAM on a ConvNeXt stage output (stage4 = features[7], 7 x 7, the layer a deployment would use;
    stage3 = features[5], 14 x 14). Returns CAMs upsampled bilinearly to 224 and max-normalised; last_raw keeps the
    native-resolution CAM for the deployed display rule."""

    def __init__(self, model, which):
        self.model = model
        layer = model.features[7] if which == "stage4" else model.features[5]
        self.act = self.grad = None
        layer.register_forward_hook(self._hook)
        self.last_raw = None

    def _hook(self, module, inputs, output):
        self.act = output
        if output.requires_grad:
            output.register_hook(lambda g: setattr(self, "grad", g))

    def __call__(self, x, classes):
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            self.model.zero_grad(set_to_none=True)
            logits = self.model(x)
            logits[torch.arange(len(classes)), torch.tensor(classes, device=logits.device)].sum().backward()
        w = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.act).sum(1)).detach()
        cam = cam / (cam.flatten(1).max(1)[0].view(-1, 1, 1) + 1e-8)
        self.last_raw = cam.cpu().numpy()
        up = nn.functional.interpolate(cam[:, None], size=TARGET_SIZE, mode="bilinear", align_corners=False)[:, 0]
        return up.cpu().numpy()


def localisation(cam, m, head):
    """Pointing game (exact; 16-px tolerance), energy inside the mask and its lift over the mask's share of the head,
    area-matched IoU (top-|mask| CAM pixels within the head)."""
    hm = cam * head
    am = np.unravel_index(np.argmax(hm), hm.shape)
    yy, xx = np.nonzero(m)
    dist = np.sqrt(((yy - am[0]) ** 2 + (xx - am[1]) ** 2).min()) if len(yy) else np.inf
    share = m[head].sum() / max(head.sum(), 1)
    energy = float((hm * m).sum() / (hm.sum() + 1e-8))
    k = int(m.sum())
    thr = np.sort(hm[head])[-k] if 0 < k <= head.sum() else np.inf
    top = (hm >= thr) & head
    return {"point_hit": int(bool(m[am])), "point_hit_tol16": int(dist <= 16), "point_chance": float(share),
            "energy_in_mask": energy, "energy_lift": float(energy / share) if share > 0 else np.nan,
            "iou_area_matched": float((top & m).sum() / ((top | m).sum() + 1e-8))}


def deployed_region(raw7, m):
    """The prototype's displayed outline rule applied to a 7 x 7 CAM: suppress below the 70th percentile, rescale,
    bicubic upsample, region = > 0.6."""
    thr = np.percentile(raw7, 70)
    c = np.clip((raw7 - thr) / (1 - thr + 1e-8), 0, 1)
    up = np.asarray(Image.fromarray((c * 255).astype(np.uint8)).resize(TARGET_SIZE, Image.BICUBIC), dtype=np.float32) / 255
    reg = up > 0.6
    return {"region_px": int(reg.sum()), "region_precision_in_mask": float((reg & m).sum() / max(reg.sum(), 1)),
            "mask_recall_by_region": float((reg & m).sum() / max(m.sum(), 1)), "region_touches_mask": int((reg & m).any())}


def cluster_mean_ci(v, pk, seed=0):
    v, pk = np.asarray(v, float), np.asarray(pk)
    ok = ~np.isnan(v)
    v, pk = v[ok], pk[ok]
    if len(v) == 0:
        return None
    u, inv = np.unique(pk, return_inverse=True)
    s, c = np.bincount(inv, v), np.bincount(inv)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(N_BOOT):
        j = rng.integers(0, len(u), len(u))
        bs.append(s[j].sum() / c[j].sum())
    return [float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(len(v)), int(len(u))]


def summarise_loc(loc):
    cols = [c for c in loc.columns if c.startswith(("stage4_", "stage3_", "deployed_display_"))]
    out = {}
    for name, g in [("ALL", loc)] + [(f"{ds}", x) for ds, x in loc.groupby("dataset")] + \
                   [(f"{ds}|{'correct' if c else 'missed'}", x) for (ds, c), x in loc.groupby(["dataset", "correct"])]:
        out[name] = {c: cluster_mean_ci(g[c], g["pk"]) for c in cols}
        out[name]["slices"] = int(len(g))
    for ds, x in loc.groupby("dataset"):
        q = x["lesion_px_224"].quantile([1 / 3, 2 / 3]).to_numpy()
        x = x.assign(size=np.where(x["lesion_px_224"] <= q[0], "small", np.where(x["lesion_px_224"] <= q[1], "medium", "large")))
        for sz, g in x.groupby("size"):
            out[f"{ds}|size_{sz}"] = {c: cluster_mean_ci(g[c], g["pk"]) for c in
                                      ("stage4_point_hit_tol16", "stage4_energy_lift", "stage4_iou_area_matched")}
            out[f"{ds}|size_{sz}"]["slices"] = int(len(g))
            out[f"{ds}|size_{sz}"]["lesion_px_range"] = [int(g["lesion_px_224"].min()), int(g["lesion_px_224"].max())]
    return out


def blur_t(x):
    return transforms.functional.gaussian_blur(x, kernel_size=41, sigma=10.0)


@torch.no_grad()
def p_isch(model, x):
    return torch.softmax(model(x), 1)[:, 1].cpu().numpy()


def faithfulness(model, cam, loader, les):
    """Deletion test on up to FAITH_N slices per dataset: replace the top-k% (of head pixels) by Grad-CAM with a blurred
    copy of the input and compare the drop in p(ischemic) with deleting the same number of random head pixels (mean of
    3 draws) and with deleting the lesion-mask pixels. Identity view, no TTA."""
    rng = np.random.default_rng(1)
    sub = pd.concat([g.sample(n=min(FAITH_N, len(g)), random_state=1) for _, g in les.groupby("dataset")]).reset_index(drop=True)
    recs = []
    for start in range(0, len(sub), 32):
        chunk = sub.iloc[start:start + 32]
        x = torch.stack([eval_transform(loader.merged(r)) for _, r in chunk.iterrows()]).to(DEVICE)
        c = cam(x, [1] * len(chunk))
        xb = blur_t(x)
        p0 = p_isch(model, x)
        heads = [gray_input(loader, r) > 10 for _, r in chunk.iterrows()]
        res = {"p0": p0}

        def run(maskfn, key):
            mm = torch.from_numpy(np.stack([maskfn(i) for i in range(len(chunk))])).to(DEVICE)[:, None].float()
            res[key] = p_isch(model, x * (1 - mm) + xb * mm)

        for f in DEL_FRACS:
            def top(i, f=f):
                h = heads[i]
                k = max(1, int(f * h.sum()))
                v = np.where(h, c[i], -1.0).ravel()
                o = np.zeros(v.size, bool)
                o[np.argpartition(-v, k - 1)[:k]] = True
                return o.reshape(h.shape)

            run(top, f"cam_{f}")
            rs = []
            for d in range(3):
                def rnd(i, f=f):
                    h = heads[i]
                    k = max(1, int(f * h.sum()))
                    idx = np.flatnonzero(h.ravel())
                    o = np.zeros(h.size, bool)
                    o[rng.choice(idx, size=min(k, len(idx)), replace=False)] = True
                    return o.reshape(h.shape)
                run(rnd, "tmp")
                rs.append(res.pop("tmp"))
            res[f"random_{f}"] = np.mean(rs, 0)
        masks_ = list(chunk["_m"])
        run(lambda i: masks_[i], "mask")
        for i, (_, r) in enumerate(chunk.iterrows()):
            recs.append({"dataset": r["dataset"], "pk": r["pk"], "correct": int(r["pred_locked"] == 1),
                         "lesion_share_of_head": float(masks_[i][heads[i]].sum() / max(heads[i].sum(), 1)),
                         **{k: float(v[i]) for k, v in res.items()}})
    df = pd.DataFrame(recs)
    df.to_csv(OUTPUT_DIR / "c3_cpaisd_deletion_per_slice.csv", index=False)
    out = {}
    for name, g in [("ALL", df)] + list(df.groupby("dataset")):
        o = {"slices": int(len(g)), "p0_mean": float(g["p0"].mean())}
        for f in DEL_FRACS:
            o[f"drop_cam_{f}"] = cluster_mean_ci(g["p0"] - g[f"cam_{f}"], g["pk"])
            o[f"drop_random_{f}"] = cluster_mean_ci(g["p0"] - g[f"random_{f}"], g["pk"])
            o[f"drop_cam_minus_random_{f}"] = cluster_mean_ci(g[f"random_{f}"] - g[f"cam_{f}"], g["pk"])
        o["drop_mask_deletion"] = cluster_mean_ci(g["p0"] - g["mask"], g["pk"])
        o["lesion_share_of_head_mean"] = float(g["lesion_share_of_head"].mean())
        out[str(name)] = o
    return out


def randomisation_check(loader, les, state):
    """Cascading-randomisation sanity check (Adebayo et al., 2018): re-initialise the classifier and the last stage,
    recompute stage-4 Grad-CAM for the ischemic class on 120 slices, Spearman correlation with the trained model's CAM."""
    sub = les.sample(n=min(120, len(les)), random_state=2).reset_index(drop=True)
    trained = build_model(pretrained=False).to(DEVICE)
    trained.load_state_dict(state)
    trained.eval()
    rnd = build_model(pretrained=False).to(DEVICE)
    rnd.load_state_dict(state)
    torch.manual_seed(0)
    for mod in list(rnd.features[7].modules()) + list(rnd.classifier.modules()):
        if hasattr(mod, "reset_parameters") and mod is not rnd.features[7]:
            mod.reset_parameters()
    rnd.eval()
    ca, cb = GradCAM(trained, "stage4"), GradCAM(rnd, "stage4")
    rho = []
    for start in range(0, len(sub), 30):
        chunk = sub.iloc[start:start + 30]
        x = torch.stack([eval_transform(loader.merged(r)) for _, r in chunk.iterrows()]).to(DEVICE)
        a, b = ca(x, [1] * len(chunk)), cb(x, [1] * len(chunk))
        for i in range(len(chunk)):
            ra = pd.Series(ca.last_raw[i].ravel()).rank().to_numpy()
            rb = pd.Series(cb.last_raw[i].ravel()).rank().to_numpy()
            rho.append(float(np.corrcoef(ra, rb)[0, 1]) if ra.std() > 0 and rb.std() > 0 else np.nan)
    rho = np.array(rho)
    return {"slices": int(len(rho)), "spearman_median": float(np.nanmedian(rho)), "spearman_mean": float(np.nanmean(rho)),
            "spearman_iqr": [float(np.nanpercentile(rho, 25)), float(np.nanpercentile(rho, 75))]}


def save_previews(loader, les, cam, orient):
    d = OUTPUT_DIR / "c3_previews"
    d.mkdir(exist_ok=True)
    for ds, g in les.groupby("dataset"):
        g = g.sample(n=min(8, len(g)), random_state=3)
        x = torch.stack([eval_transform(loader.merged(r)) for _, r in g.iterrows()]).to(DEVICE)
        c = cam(x, [1] * len(g))
        for i, (_, r) in enumerate(g.iterrows()):
            img = gray_input(loader, r)
            rgb = np.stack([img] * 3, -1)
            heat = c[i][..., None] * np.array([255, 60, 0], np.float32)
            over = np.clip(rgb * 0.6 + heat * 0.4, 0, 255)
            m = r["_m"]
            edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
            over[edge] = [0, 255, 0]
            canvas = np.concatenate([rgb, over], 1).astype(np.uint8)
            tag = "hit" if r["pred_locked"] == 1 else "miss"
            Image.fromarray(canvas).save(d / f"{ds}_{i}_{tag}.png")
    log(f"C3 previews saved ({DIHEDRAL_NAMES[orient.get('AISD', 0)]}, ...)")


if __name__ == "__main__":
    main()
