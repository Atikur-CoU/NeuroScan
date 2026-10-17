"""
NeuroScan -- input-file check evaluation (2026-10-08). CPU; no model. The deployed check flagged every slice of a
genuine full-field SICH CT scan (whole-image mid-tone share < 30% because of the black background), which also
suppresses alerts. This kernel measures, on real scan images exactly as stored in the datasets (validation and held-out
test rows, up to 400 per source; CT slices, MRI DWI and ADC separately) and on non-scan images (colour and grayscale
photographs, documents, diagrams, screenshots), how often the old rule and a redesigned rule (thresholds fixed before
this run; see RULE_TEXT) wrongly flag real scans and wrongly accept non-scans. The data pipeline above main()'s loader
construction is the final model's own script, unchanged.
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
from PIL import Image, ImageDraw
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
    # INPUT-FILE CHECK EVALUATION. Everything above this line is the final model's own data pipeline, unchanged.
    # ================================================================================================================
    R = {"note": "input-file check: old rule vs new rule; thresholds fixed before this run; validation rows may be used "
                 "to adjust, held-out test rows only report", "rules": RULE_TEXT}
    pos = {}
    for split, df in (("val", val_all), ("test", pd.concat([splits["test"], soop_hold], ignore_index=True))):
        rows = []
        for ds, g in df.groupby("dataset"):
            g = g.sample(n=min(N_PER_SOURCE, len(g)), random_state=0)
            for _, r in g.iterrows():
                files = [("dwi", r["dwi_out_path"]), ("adc", r["adc_out_path"])] if ds in MRI_PAIR_DATASETS else [("ct", r["out_path"])]
                for kind, pth in files:
                    try:
                        img = raw_image(loader, r, pth)
                    except Exception as e:  # noqa: BLE001
                        rows.append({"dataset": ds, "kind": kind, "label": r["label"], "error": repr(e)[:80]})
                        continue
                    rows.append({"dataset": ds, "kind": kind, "label": r["label"], **features(img)})
        pos[split] = pd.DataFrame(rows)
        log(f"{split}: {len(pos[split])} real scan images; read errors {int(pos[split].get('error', pd.Series(dtype=str)).notna().sum())}")
    neg = negatives()
    log(f"non-scan images: {len(neg)} ({neg['category'].value_counts().to_dict()})")

    for split, df in pos.items():
        df = df[df.get("error", pd.Series([None] * len(df))).isna()] if "error" in df else df
        out = {}
        for rule in RULES:
            df[rule] = [RULES[rule](f) for f in df.to_dict("records")]
        for (ds, kind), g in df.groupby(["dataset", "kind"]):
            out[f"{ds}|{kind}"] = {"n": int(len(g)), **{f"falsely_flagged_{rule}": float((~g[rule]).mean()) for rule in RULES}}
            log(f"{split} {ds} {kind}: n={len(g)} " + " ".join(f"{rule}={100 * (~g[rule]).mean():.1f}%" for rule in RULES))
        out["ALL"] = {"n": int(len(df)), **{f"falsely_flagged_{rule}": float((~df[rule]).mean()) for rule in RULES}}
        log(f"{split} ALL real scans falsely flagged: " + " ".join(f"{rule}={100 * (~df[rule]).mean():.2f}%" for rule in RULES))
        R[f"real_scans_{split}"] = out
        q = df[["dataset", "kind", "fg_frac", "mid_fg", "mid_all", "border_dark"]].groupby(["dataset", "kind"]).quantile([0.01, 0.05, 0.5])
        R[f"feature_quantiles_{split}"] = {f"{a}|{b}|q{c}": v for (a, b, c), v in q.round(3).to_dict("index").items()}
        df.drop(columns=list(RULES)).to_csv(OUTPUT_DIR / f"guard_features_real_{split}.csv", index=False)
    for rule in RULES:
        neg[rule] = [RULES[rule](f) for f in neg.to_dict("records")]
    R["non_scans"] = {c: {"n": int(len(g)), **{f"wrongly_accepted_{rule}": float(g[rule].mean()) for rule in RULES}}
                      for c, g in neg.groupby("category")}
    for c, v in R["non_scans"].items():
        log(f"non-scan {c}: n={v['n']} " + " ".join(f"{rule} accepted={100 * v['wrongly_accepted_' + rule]:.1f}%" for rule in RULES))
    neg.drop(columns=list(RULES)).to_csv(OUTPUT_DIR / "guard_features_nonscans.csv", index=False)
    with open(OUTPUT_DIR / "input_check_eval.json", "w") as f:
        json.dump(R, f, indent=2, default=float)
    log("Done.")


# ====================================================================================================================
# Input-file check: features, rules, non-scan images
# ====================================================================================================================
N_PER_SOURCE = 400
RULE_TEXT = {
    "old": "near-grey (mean channel difference <= 8) AND >= 30% of ALL pixels mid-tone (40 < v < 220)",
    "new": "near-grey AND head (pixels > 10) covers >= 5% of the image AND >= 30% of head pixels mid-tone AND "
           ">= 50% of the outer border ring (6% of each side) dark (<= 30)",
}


def features(img):
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cd = float(((np.abs(r - g) + np.abs(g - b) + np.abs(r - b)) / 3.0).mean())
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    mid = (gray > 40) & (gray < 220)
    fg = gray > 10
    h, w = gray.shape
    bh, bw = max(1, int(0.06 * h)), max(1, int(0.06 * w))
    ring = np.ones_like(gray, dtype=bool)
    ring[bh:h - bh, bw:w - bw] = False
    return {"channel_diff": cd, "mid_all": float(mid.mean()), "fg_frac": float(fg.mean()),
            "mid_fg": float((mid & fg).sum() / max(fg.sum(), 1)), "border_dark": float((gray[ring] <= 30).mean()),
            "h": int(h), "w": int(w)}


RULES = {
    "old": lambda f: f["channel_diff"] <= 8.0 and f["mid_all"] >= 0.30,
    "new": lambda f: (f["channel_diff"] <= 8.0 and f["fg_frac"] >= 0.05 and f["mid_fg"] >= 0.30
                      and f["border_dark"] >= 0.50),
}


def raw_image(loader, r, pth):
    """The stored image file as a user would upload it (not resized), via the final model's own path resolution."""
    if loader.rsna_dicom and str(pth).startswith("RSNA_ICH/"):
        return Image.open(loader.dcm_index[Path(pth).stem])
    return Image.open(loader._resolve(pth))


def negatives():
    """Non-scan images: colour and grayscale photographs (scikit-image sample data, plus crops and resizes), synthetic
    documents, diagrams and user-interface screenshots."""
    ensure_pkg("skimage", "scikit-image")
    from skimage import data as skd
    rng = np.random.default_rng(7)
    out = []

    def add(cat, name, im):
        out.append({"category": cat, "name": name, **features(im)})

    photos = {}
    for nm in ("astronaut", "coffee", "chelsea", "rocket", "retina", "immunohistochemistry", "colorwheel", "hubble_deep_field",
               "camera", "moon", "page", "text", "coins", "horse", "brick", "grass", "gravel", "clock", "checkerboard", "cell"):
        try:
            a = getattr(skd, nm)()
        except Exception:  # noqa: BLE001
            continue
        a = np.asarray(a)
        if a.dtype == bool:
            a = a.astype(np.uint8) * 255
        if a.dtype != np.uint8:
            a = a.astype(np.float32)
            a = (255 * (a.astype(np.float32) - a.min()) / max(float(a.max() - a.min()), 1e-6)).astype(np.uint8)
        photos[nm] = Image.fromarray(a)
    for nm, im in photos.items():
        cat = "photo_colour" if im.mode in ("RGB", "RGBA") and np.asarray(im.convert("RGB")).std(axis=2).mean() > 3 else "photo_grayscale"
        if nm in ("page", "text"):
            cat = "document_scan"
        add(cat, nm, im)
        for k in range(3):
            W, H = im.size
            cw, ch = int(W * rng.uniform(0.5, 0.9)), int(H * rng.uniform(0.5, 0.9))
            x, y = int(rng.integers(0, W - cw + 1)), int(rng.integers(0, H - ch + 1))
            add(cat, f"{nm}_crop{k}", im.crop((x, y, x + cw, y + ch)).resize((512, 512)))
        add(cat, f"{nm}_gray", im.convert("L")) if cat == "photo_colour" else None
    for k in range(40):  # documents
        im = Image.new("L", (600, 800), int(rng.integers(235, 256)))
        d = ImageDraw.Draw(im)
        y = 40
        while y < 760:
            x0 = 40 + int(rng.integers(0, 20))
            d.text((x0, y), " ".join("".join(chr(int(c)) for c in rng.integers(97, 123, int(rng.integers(3, 9)))) for _ in range(10)),
                   fill=int(rng.integers(0, 60)))
            y += int(rng.integers(14, 24))
        add("document_synthetic", f"doc{k}", im if k % 2 else im.convert("RGB"))
    for k in range(40):  # diagrams / flowcharts
        bg = int(rng.choice([255, 250, 240]))
        im = Image.new("RGB", (800, 600), (bg, bg, bg))
        d = ImageDraw.Draw(im)
        for _ in range(int(rng.integers(4, 10))):
            x, y = int(rng.integers(20, 650)), int(rng.integers(20, 500))
            col = tuple(int(c) for c in rng.integers(0, 120, 3)) if k % 2 else (0, 0, 0)
            d.rectangle([x, y, x + int(rng.integers(60, 150)), y + int(rng.integers(30, 80))], outline=col, width=2)
            d.line([x, y, int(rng.integers(20, 780)), int(rng.integers(20, 580))], fill=col, width=2)
        add("diagram", f"diagram{k}", im)
    for k in range(40):  # UI screenshots: light or dark theme, panels, text
        dark = k % 3 == 0
        base = int(rng.integers(20, 45)) if dark else int(rng.integers(230, 256))
        im = Image.new("RGB", (1280, 800), (base, base, base))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 1280, 60], fill=tuple(int(c) for c in rng.integers(0, 256, 3)))
        for _ in range(int(rng.integers(3, 8))):
            x, y = int(rng.integers(0, 1000)), int(rng.integers(70, 650))
            shade = int(rng.integers(40, 90)) if dark else int(rng.integers(200, 250))
            d.rectangle([x, y, x + int(rng.integers(150, 400)), y + int(rng.integers(80, 200))], fill=(shade, shade, shade))
            d.text((x + 10, y + 10), "Lorem ipsum dolor sit amet", fill=(220, 220, 220) if dark else (30, 30, 30))
        add("screenshot", f"ui{k}", im if k % 2 else im.convert("L"))
    return pd.DataFrame(out)


if __name__ == "__main__":
    main()
