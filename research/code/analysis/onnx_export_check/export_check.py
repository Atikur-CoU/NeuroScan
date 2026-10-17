"""
NeuroScan -- ONNX export and equivalence check for the v2 inference server (2026-10-07). CPU; no data (synthetic
head-like phantoms only). Final model R30A s123, weights unchanged.
1. A closed-form Grad-CAM head (mean-pool -> LayerNorm -> Linear has an exact gradient) is checked against the model's
   logits and against autograd Grad-CAM on ConvNeXt stage 4.
2. Exported to ONNX (opset 17; outputs logits and per-class Grad-CAM), checked with onnx.checker, compared with PyTorch.
3. The server's own inference.py (embedded verbatim below; only the checksum is filled in) runs in a separate process
   with the pinned server packages (onnxruntime 1.20.1, numpy 2.1.3, pillow 11.0.0) and is compared with a PyTorch
   reference of the deployed rule (4-view TTA, mean softmax, plain argmax) on the four input pathways; peak memory and
   time per request are reported. Output: the ONNX file (for the GitHub release) and onnx_export_check.json.
"""
INFERENCE_PY = '"""\nNeuroScan web demo -- shared inference logic (v2: the evaluated final model and its input pathway).\n\nModel: the paper\'s final model, R30A s123 -- ConvNeXt-Tiny, one softmax over normal / ischemic / hemorrhagic --\nserved with ONNX Runtime (models/neuroscan_r30a_s123.onnx; SHA-256 checked at start-up). PyTorch is not needed at\nrun time: it alone took about 300 MB, and the PyTorch server peaked at 671 MB on Linux, above the 512 MB hosting tier.\nThe ONNX graph was exported from the PyTorch weights and checked against them (kaggle kernel\nneuroscan-q-onnx-export-check): it returns the logits and, for each class, the Grad-CAM map of ConvNeXt stage 4.\nThe Grad-CAM channel weights are computed inside the graph in closed form -- the head after stage 4 is global average\npooling, LayerNorm and one linear layer, so the gradient of a logit with respect to the stage-4 activations is the\nsame at every position and has an exact expression; the kernel checks it against autograd.\n\nInput pathway -- the one the offline evaluation used, chosen per request from what was uploaded:\n  * CT, current slice + previous + next  -> channels (previous, current, next)       ["ct_adjacent_slices", evaluated]\n  * MRI, DWI + ADC of the same slice      -> channels (DWI, ADC, DWI)                 ["mri_dwi_adc", evaluated]\n  * anything else (one image only)        -> channels (image, image, image)          ["single_image", NOT evaluated\n    as a deployment pathway; Post-hoc experiment C2 measures how much it loses]. The response says which pathway ran\n    and carries a warning when it is not the evaluated one.\nEvery image is converted to 8-bit grayscale and resized to 224 x 224, exactly as in training.\n\nDecision rule: 4-view test-time augmentation (identity, horizontal flip, +8 deg, -8 deg rotation), softmax averaged\nover views, prediction = argmax (plain). This is the rule under which the paper evaluates the alert tiers (post-hoc\nanalysis B4): it alerts on more hemorrhage slices than the paper\'s class-weighted rule (92.0% vs 89.0% on the pooled\nheld-out test), at the cost of more false alerts, which is the safer trade-off for a triage aid. The displayed\nprobabilities are these same TTA-averaged softmax values (uncalibrated), as in B4.\n\nMRI and hemorrhage: no hemorrhagic MRI source was available for training or evaluation, so an MRI request carries a\nwarning that hemorrhage on MRI is outside the evaluated scope.\n\nGrad-CAM [Selvaraju et al., 2017]: identity view only, for the predicted class, overlaid on the current CT slice or\nthe DWI image; suppress below the image\'s 70th-percentile activation, rescale, bicubic upsample; outline at > 0.6 for\nischemic / hemorrhagic. Post-hoc experiment C3 evaluates this display rule against lesion masks. It shows where the\nmodel\'s evidence is concentrated, not a lesion boundary.\n"""\nimport base64\nimport hashlib\nimport io\nfrom pathlib import Path\n\nimport numpy as np\nimport onnxruntime as ort\nfrom PIL import Image\n\nMODELS_DIR = Path(__file__).parent / "models"\nMODEL_FILE = MODELS_DIR / "neuroscan_r30a_s123.onnx"\nEXPECTED_SHA256 = "TO_BE_FILLED_FROM_EXPORT_KERNEL"\nMODEL_VERSION = "r30a_s123_convnext_tiny_onnx"\n\nTARGET_SIZE = (224, 224)\nVALID_LABELS = ("normal", "ischemic", "hemorrhagic")\nLABEL_TO_IDX = {"normal": 0, "ischemic": 1, "hemorrhagic": 2}\nMODALITIES = ("CT", "MRI")\n\nPATHWAY_CT = "ct_adjacent_slices"\nPATHWAY_MRI = "mri_dwi_adc"\nPATHWAY_SINGLE = "single_image"\n\nMEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]\nSTD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]\n\n\nclass InputError(ValueError):\n    """The uploaded combination of images cannot be processed (shown to the user as a 400)."""\n\n\ndef _gray(image: Image.Image) -> Image.Image:\n    return image.convert("L").resize(TARGET_SIZE)\n\n\ndef build_input(modality: str, main: Image.Image, prev: Image.Image | None = None, nxt: Image.Image | None = None,\n                adc: Image.Image | None = None):\n    """Returns (merged RGB image, pathway name, base image for the overlay)."""\n    if modality not in MODALITIES:\n        raise InputError("modality must be CT or MRI.")\n    cur = _gray(main)\n    if modality == "CT":\n        if adc is not None:\n            raise InputError("An ADC image applies to MRI only.")\n        if (prev is None) != (nxt is None):\n            raise InputError("For CT context, upload both the previous and the next slice (or neither).")\n        if prev is not None:\n            return Image.merge("RGB", (_gray(prev), cur, _gray(nxt))), PATHWAY_CT, cur\n    else:\n        if prev is not None or nxt is not None:\n            raise InputError("Previous/next slices apply to CT only; for MRI upload DWI and ADC.")\n        if adc is not None:\n            return Image.merge("RGB", (cur, _gray(adc), cur)), PATHWAY_MRI, cur\n    return Image.merge("RGB", (cur, cur, cur)), PATHWAY_SINGLE, cur\n\n\ndef _looks_like_grayscale_scan(\n    image: Image.Image, max_mean_channel_diff: float = 8.0, min_midtone_fraction: float = 0.30\n) -> bool:\n    """Input-file check, not an anatomical or diagnostic-quality check: the model was trained only on grayscale\n    CT/MRI slices and has no "none of the above" output, so this flags the commonest wrong-file uploads (colour\n    photos; mostly-white documents or diagrams) before a meaningless result reaches an alert. Both must hold:\n    near-identical R/G/B per pixel, and >= 30% midtone pixels. It will not catch every unrelated image (e.g. a\n    grayscale, midtone-heavy X-ray of something else)."""\n    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)\n    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]\n    channel_diff = (np.abs(r - g) + np.abs(g - b) + np.abs(r - b)) / 3.0\n    if channel_diff.mean() > max_mean_channel_diff:\n        return False\n    gray = np.asarray(image.convert("L"), dtype=np.float32)\n    midtone_fraction = ((gray > 40) & (gray < 220)).mean()\n    return bool(midtone_fraction >= min_midtone_fraction)\n\n\ndef _tta_views(img: Image.Image):\n    """Identical to torchvision\'s PIL path: hflip = FLIP_LEFT_RIGHT; rotate(angle, fill=0) = PIL rotate with\n    nearest-neighbour resampling, no expansion, black fill."""\n    yield img\n    yield img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)\n    yield img.rotate(8, resample=Image.Resampling.NEAREST, expand=False, fillcolor=(0, 0, 0))\n    yield img.rotate(-8, resample=Image.Resampling.NEAREST, expand=False, fillcolor=(0, 0, 0))\n\n\ndef _to_array(img: Image.Image) -> np.ndarray:\n    """torchvision to_tensor + ImageNet normalize, as float32 CHW."""\n    a = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0\n    return (a - MEAN) / STD\n\n\ndef _softmax(z: np.ndarray) -> np.ndarray:\n    e = np.exp(z - z.max(axis=1, keepdims=True))\n    return e / e.sum(axis=1, keepdims=True)\n\n\ndef _sha256(path: Path) -> str:\n    h = hashlib.sha256()\n    with open(path, "rb") as f:\n        for chunk in iter(lambda: f.read(1 << 20), b""):\n            h.update(chunk)\n    return h.hexdigest()\n\n\nclass ThreeClassPipeline:\n    """Loads the final model once; call .predict(...) per request."""\n\n    def __init__(self):\n        digest = _sha256(MODEL_FILE)\n        if digest != EXPECTED_SHA256:\n            raise RuntimeError(f"Model file checksum mismatch ({digest[:12]}...); refusing to serve.")\n        opts = ort.SessionOptions()\n        opts.intra_op_num_threads = 1\n        opts.inter_op_num_threads = 1\n        opts.enable_cpu_mem_arena = False  # lower resident memory between requests on the 512 MB tier\n        self.session = ort.InferenceSession(str(MODEL_FILE), sess_options=opts, providers=["CPUExecutionProvider"])\n        meta = self.session.get_modelmeta().custom_metadata_map\n        self.epoch = int(meta["epoch"]) if "epoch" in meta else None\n        self.val_acc = float(meta["val_acc"]) if "val_acc" in meta else None\n        self.weights_sha256 = digest\n\n    @staticmethod\n    def _risk_tier_color(prediction: str, prob_normal: float, prob_ischemic: float, prob_hemorrhagic: float):\n        """Same confidence tiers as webapp_php/includes/risk_tier.php (kept in sync by hand), so the heatmap tint\n        always agrees with the badge."""\n        if prediction == "hemorrhagic":\n            if prob_hemorrhagic >= 0.55:\n                return (192, 57, 43)\n            return (230, 126, 34)\n        if prediction == "ischemic":\n            if prob_ischemic >= 0.90:\n                return (192, 57, 43)\n            if prob_ischemic >= 0.70:\n                return (230, 126, 34)\n            return (127, 140, 141)\n        if prediction == "normal" and prob_normal >= 0.80:\n            return (41, 128, 185)\n        return (127, 140, 141)\n\n    @staticmethod\n    def _boundary_mask(mask: np.ndarray, thickness: int = 2) -> np.ndarray:\n        eroded = mask.copy()\n        eroded[1:, :] &= mask[:-1, :]\n        eroded[:-1, :] &= mask[1:, :]\n        eroded[:, 1:] &= mask[:, :-1]\n        eroded[:, :-1] &= mask[:, 1:]\n        boundary = mask & ~eroded\n        for _ in range(thickness - 1):\n            dilated = boundary.copy()\n            dilated[1:, :] |= boundary[:-1, :]\n            dilated[:-1, :] |= boundary[1:, :]\n            dilated[:, 1:] |= boundary[:, :-1]\n            dilated[:, :-1] |= boundary[:, 1:]\n            boundary = dilated\n        return boundary\n\n    @staticmethod\n    def _display_cam(cam7: np.ndarray) -> Image.Image:\n        """Prototype display rule on the 7 x 7 Grad-CAM (already ReLU\'d): max-normalise, suppress below the 70th\n        percentile, rescale, bicubic upsample to 224."""\n        cam = cam7 / (cam7.max() + 1e-8)\n        threshold = np.percentile(cam, 70)\n        cam = np.clip((cam - threshold) / (1 - threshold + 1e-8), 0, 1)\n        return Image.fromarray((cam * 255).astype(np.uint8)).resize(TARGET_SIZE, Image.BICUBIC)\n\n    @staticmethod\n    def _overlay_heatmap(base_image: Image.Image, cam_img: Image.Image, color, draw_boundary: bool) -> Image.Image:\n        base_arr = np.asarray(base_image.convert("RGB"), dtype=np.float32)\n        cam_arr = np.asarray(cam_img, dtype=np.float32) / 255.0\n        tint = np.asarray(color, dtype=np.float32)\n        alpha = (cam_arr * 0.75)[..., None]\n        blended = base_arr * (1 - alpha) + tint * alpha\n        if draw_boundary:\n            boundary = ThreeClassPipeline._boundary_mask(cam_arr > 0.6, thickness=2)\n            blended[boundary] = tint\n        return Image.fromarray(blended.astype(np.uint8))\n\n    def predict(self, modality: str, image: Image.Image, prev: Image.Image | None = None,\n                nxt: Image.Image | None = None, adc: Image.Image | None = None) -> dict:\n        merged, pathway, base = build_input(modality, image, prev, nxt, adc)\n        x = np.stack([_to_array(v) for v in _tta_views(merged)])\n        logits, cams = self.session.run(["logits", "gradcam"], {"image": x})\n        probs = _softmax(logits.astype(np.float64)).mean(axis=0)\n        idx = int(np.argmax(probs))\n        prediction = VALID_LABELS[idx]\n        prob_normal, prob_ischemic, prob_hemorrhagic = (float(v) for v in probs)\n\n        uploads = [u for u in (image, prev, nxt, adc) if u is not None]\n        try:\n            likely_not_a_scan = not all(_looks_like_grayscale_scan(u) for u in uploads)\n        except Exception:\n            likely_not_a_scan = False\n\n        warnings = []\n        if pathway == PATHWAY_SINGLE:\n            warnings.append("Single image only: this is not the evaluated input pathway. For CT upload the previous "\n                            "and next slices; for MRI upload the matching ADC image.")\n        if modality == "MRI":\n            warnings.append("MRI: hemorrhage was not represented in training or evaluation; MRI results concern "\n                            "normal versus ischemic only.")\n        result = {\n            "prediction": prediction,\n            "prob_normal": prob_normal,\n            "prob_ischemic": prob_ischemic,\n            "prob_hemorrhagic": prob_hemorrhagic,\n            "modality": modality,\n            "pathway": pathway,\n            "evaluated_pathway": pathway != PATHWAY_SINGLE,\n            "model_version": MODEL_VERSION,\n            "likely_not_a_scan": likely_not_a_scan,\n            "warnings": warnings,\n            "decision_note": None,\n        }\n        try:\n            color = self._risk_tier_color(prediction, prob_normal, prob_ischemic, prob_hemorrhagic)\n            overlay = self._overlay_heatmap(base, self._display_cam(cams[0, idx]), color,\n                                            prediction in ("ischemic", "hemorrhagic"))\n            buf = io.BytesIO()\n            overlay.save(buf, format="PNG")\n            result["heatmap_png_base64"] = base64.b64encode(buf.getvalue()).decode("ascii")\n        except Exception:\n            pass\n        return result\n\n\n_pipeline: ThreeClassPipeline | None = None\n\n\ndef get_pipeline() -> ThreeClassPipeline:\n    global _pipeline\n    if _pipeline is None:\n        _pipeline = ThreeClassPipeline()\n    return _pipeline\n'

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime"], check=True)
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFilter
from torchvision import models, transforms
from torchvision.transforms import functional as TF

T0 = time.time()
OUT = Path("/kaggle/working/output")
OUT.mkdir(parents=True, exist_ok=True)
SRV = Path("/kaggle/working/srv")
(SRV / "models").mkdir(parents=True, exist_ok=True)
W = next(p for p in (Path("/kaggle/input/datasets/atik68/neuroscan-q-r30a-s123-weights/r30a_s123_weights.pt"),
                     Path("/kaggle/input/neuroscan-q-r30a-s123-weights/r30a_s123_weights.pt")) if p.exists())
ONNX_PATH = OUT / "neuroscan_r30a_s123.onnx"
R = {}


def log(m):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {m}", flush=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


ck = torch.load(W, map_location="cpu", weights_only=False)
model = models.convnext_tiny(weights=None)
model.classifier[2] = nn.Linear(768, 3)
model.load_state_dict(ck["model_state"])
model.eval()
R["weights_sha256"] = sha256(W)


class CamHead(nn.Module):
    """logits + per-class Grad-CAM of stage 4 (ReLU(sum_k w_ck A_k)), with w_ck = spatial mean of d logit_c / d A_k
    in closed form: the head is mean-pool -> LayerNorm2d -> Linear, so d logit_c / d A[k,i,j] = g_ck / (H W) where
    g_c = d logit_c / d pooled = (1/s) (v_c - mean(v_c) - xhat * mean(v_c * xhat)), v_c = W_c * gamma."""

    def __init__(self, m):
        super().__init__()
        self.features, self.ln, self.fc = m.features, m.classifier[0], m.classifier[2]

    def forward(self, x):
        a = self.features(x)                                   # B, 768, 7, 7
        pooled = a.mean(dim=(2, 3))                            # B, 768
        mu = pooled.mean(dim=1, keepdim=True)
        var = pooled.var(dim=1, unbiased=False, keepdim=True)
        s = torch.sqrt(var + self.ln.eps)
        xhat = (pooled - mu) / s
        logits = self.fc(xhat * self.ln.weight + self.ln.bias)
        v = self.fc.weight * self.ln.weight                    # 3, 768
        vb = v.unsqueeze(0)                                    # 1, 3, 768
        xh = xhat.unsqueeze(1)                                 # B, 1, 768
        g = (vb - vb.mean(dim=2, keepdim=True) - xh * (vb * xh).mean(dim=2, keepdim=True)) / s.unsqueeze(1)
        w = g / (a.shape[2] * a.shape[3])                      # B, 3, 768
        cams = torch.relu(torch.einsum("bck,bkij->bcij", w, a))
        return logits, cams


head = CamHead(model).eval()


def phantom(seed):
    """Deterministic head-like test image (ellipse, texture, one lesion-like blob), 512 x 512 grayscale."""
    rng = np.random.default_rng(seed)
    img = Image.new("L", (512, 512), 0)
    d = ImageDraw.Draw(img)
    d.ellipse([60, 40, 452, 472], fill=int(rng.integers(150, 200)))
    d.ellipse([80, 60, 432, 452], fill=int(rng.integers(90, 130)))
    cx, cy, r = rng.integers(160, 352), rng.integers(140, 372), rng.integers(20, 60)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=int(rng.integers(40, 220)))
    a = np.asarray(img, dtype=np.float32) + rng.normal(0, 8, (512, 512))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))


NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def ref_views(merged):
    """The training/evaluation TTA exactly as in the final model's script (torchvision PIL ops)."""
    vs = (merged, TF.hflip(merged), TF.rotate(merged, 8, fill=0), TF.rotate(merged, -8, fill=0))
    return torch.stack([NORMALIZE(TF.to_tensor(v)) for v in vs])


# ---- 1. closed-form head vs the model and vs autograd Grad-CAM
xs = []
for k in range(6):
    g = phantom(k).convert("L").resize((224, 224))
    xs.append(ref_views(Image.merge("RGB", (g, g, g)))[0])
x = torch.stack(xs)
with torch.no_grad():
    lm = model(x)
    lh, ch = head(x)
R["head_vs_model_logits_maxabs"] = float((lm - lh).abs().max())
acts = {}
def _cap(m, i, o):
    acts["a"] = o
    o.register_hook(lambda gr: acts.__setitem__("g", gr))


hk = model.features[7].register_forward_hook(_cap)
diffs = []
for c in range(3):
    xx = x.clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    model(xx)[:, c].sum().backward()
    wa = acts["g"].mean(dim=(2, 3), keepdim=True)
    ca = torch.relu((wa * acts["a"]).sum(1)).detach()
    na = ca / (ca.flatten(1).max(1)[0].view(-1, 1, 1) + 1e-8)
    nh = ch[:, c] / (ch[:, c].flatten(1).max(1)[0].view(-1, 1, 1) + 1e-8)
    diffs.append(float((na - nh).abs().max()))
hk.remove()
R["closed_form_vs_autograd_gradcam_normalised_maxabs_per_class"] = diffs
log(f"head vs model logits {R['head_vs_model_logits_maxabs']:.2e}; Grad-CAM closed form vs autograd {diffs}")
assert R["head_vs_model_logits_maxabs"] < 1e-4 and max(diffs) < 1e-3

# ---- 2. export
torch.onnx.export(head, x[:1], str(ONNX_PATH), input_names=["image"], output_names=["logits", "gradcam"],
                  dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}, "gradcam": {0: "batch"}},
                  opset_version=17, do_constant_folding=True, dynamo=False)
m = onnx.load(str(ONNX_PATH))
for k, v in {"epoch": str(ck.get("epoch")), "val_acc": str(ck.get("val_acc")), "model": "R30A s123 ConvNeXt-Tiny",
             "weights_sha256": R["weights_sha256"]}.items():
    m.metadata_props.add(key=k, value=v)
onnx.checker.check_model(m)
onnx.save(m, str(ONNX_PATH))
R["onnx_sha256"] = sha256(ONNX_PATH)
R["onnx_mb"] = round(ONNX_PATH.stat().st_size / 1e6, 2)
log(f"ONNX exported: {R['onnx_mb']} MB sha256 {R['onnx_sha256']}")
sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
lo, co = sess.run(["logits", "gradcam"], {"image": x.numpy()})
R["onnx_vs_torch_logits_maxabs"] = float(np.abs(lo - lh.numpy()).max())
R["onnx_vs_torch_gradcam_rel_maxabs"] = float(np.abs(co - ch.numpy()).max() / (np.abs(ch.numpy()).max() + 1e-12))
log(f"ONNX vs torch: logits {R['onnx_vs_torch_logits_maxabs']:.2e}, gradcam rel {R['onnx_vs_torch_gradcam_rel_maxabs']:.2e}")

# ---- 3. reference results with the PyTorch model (A2 rule) for the server comparison
cases = {"ct_single": ("CT", [0]), "ct_context": ("CT", [1, 2, 3]), "mri_dwi_adc": ("MRI", [4, None, None, 5]),
         "mri_single": ("MRI", [6])}
imgdir = SRV / "cases"
imgdir.mkdir(exist_ok=True)
for i in range(7):
    phantom(100 + i).save(imgdir / f"{i}.png")


def ref_predict(modality, idx):
    ims = [None if i is None else Image.open(imgdir / f"{i}.png") for i in idx] + [None] * (4 - len(idx))
    main, prev, nxt, adc = ims
    g = lambda im: im.convert("L").resize((224, 224))
    cur = g(main)
    if modality == "CT" and prev is not None:
        merged = Image.merge("RGB", (g(prev), cur, g(nxt)))
    elif modality == "MRI" and adc is not None:
        merged = Image.merge("RGB", (cur, g(adc), cur))
    else:
        merged = Image.merge("RGB", (cur, cur, cur))
    with torch.no_grad():
        p = torch.softmax(model(ref_views(merged)), 1).mean(0).numpy()
    c = int(p.argmax())
    return {"probs": p.tolist(), "prediction": c}


REF = {k: ref_predict(*v) for k, v in cases.items()}
json.dump({"cases": cases, "ref": REF}, open(SRV / "ref.json", "w"))

# ---- 4. the server's own inference.py with the pinned server packages, in a separate process
inf_src = INFERENCE_PY.replace('EXPECTED_SHA256 = "TO_BE_FILLED_FROM_EXPORT_KERNEL"', f'EXPECTED_SHA256 = "{R["onnx_sha256"]}"')
(SRV / "inference.py").write_text(inf_src)
os.symlink(ONNX_PATH, SRV / "models/neuroscan_r30a_s123.onnx")
pins = ["onnxruntime==1.20.1", "numpy==2.1.3", "pillow==11.0.0"]
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--target", "/tmp/srvpkgs", *pins], capture_output=True, text=True)
log(f"pip {pins}: exit {r.returncode} {r.stderr[-300:]}")
R["server_packages"] = pins if r.returncode == 0 else "pinned install failed; Kaggle packages used"
probe = r'''
import json, resource, sys, time
sys.path.insert(0, "/kaggle/working/srv")
def rss():
    for l in open("/proc/self/status"):
        if l.startswith("VmRSS"):
            return int(l.split()[1]) // 1024
out = {"rss_start_mb": rss()}
t = time.time()
import numpy as np, onnxruntime, PIL
from PIL import Image
import inference as inf
out["versions"] = {"onnxruntime": onnxruntime.__version__, "numpy": np.__version__, "pillow": PIL.__version__}
p = inf.get_pipeline()
out["load_s"], out["rss_after_load_mb"] = round(time.time() - t, 2), rss()
d = json.load(open("/kaggle/working/srv/ref.json"))
res, times = {}, {}
for k, (mod, idx) in d["cases"].items():
    ims = [None if i is None else Image.open(f"/kaggle/working/srv/cases/{i}.png") for i in idx] + [None] * (4 - len(idx))
    ts = []
    for _ in range(5):
        t = time.time(); r = p.predict(mod, *ims); ts.append(time.time() - t)
    res[k] = {"probs": [r["prob_normal"], r["prob_ischemic"], r["prob_hemorrhagic"]], "prediction": inf.VALID_LABELS.index(r["prediction"]),
              "pathway": r["pathway"], "heatmap": "heatmap_png_base64" in r, "warnings": len(r["warnings"])}
    times[k] = [round(float(np.median(ts)), 3), round(max(ts), 3)]
out["results"], out["seconds_median_max"] = res, times
out["rss_after_requests_mb"] = rss()
out["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
print("RESULT " + json.dumps(out), flush=True)
'''
(SRV / "probe.py").write_text(probe)
env = dict(os.environ, PYTHONPATH="/tmp/srvpkgs" if r.returncode == 0 else "", OMP_NUM_THREADS="1")
pr = subprocess.run([sys.executable, str(SRV / "probe.py")], env=env, capture_output=True, text=True)
print(pr.stderr[-2000:])
line = next((l for l in pr.stdout.splitlines() if l.startswith("RESULT ")), None)
assert line, pr.stdout[-2000:]
S = json.loads(line[7:])
R["server_run"] = S
cmp = {}
for k, ref in REF.items():
    got = S["results"][k]
    cmp[k] = {"max_abs_prob_diff": float(np.abs(np.array(got["probs"]) - np.array(ref["probs"])).max()),
              "same_prediction": got["prediction"] == ref["prediction"], "pathway": got["pathway"], "heatmap": got["heatmap"]}
R["server_vs_pytorch_reference"] = cmp
log(f"server (pinned packages) vs PyTorch reference: {json.dumps(cmp)}")
log(f"server memory/time: start {S['rss_start_mb']} MB, after load {S['rss_after_load_mb']} MB, after requests "
    f"{S['rss_after_requests_mb']} MB, peak {S['peak_rss_mb']} MB; seconds {S['seconds_median_max']}; versions {S['versions']}")
assert all(v["same_prediction"] and v["max_abs_prob_diff"] < 1e-4 for v in cmp.values())
json.dump(R, open(OUT / "onnx_export_check.json", "w"), indent=2)
log("Done.")
