"""
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
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MODELS_DIR = Path(__file__).parent / "models"
# Served copy of the model, chosen by NEUROSCAN_MODEL_VARIANT (default fp32 = the evaluated model). int8_dynamic = the
# same model with INT8 dynamic quantization of the backbone's MatMul weights; the classifier head and the in-graph
# Grad-CAM stay FP32 (kernel neuroscan-p2-compression-gradcam: 33.8 MB instead of 111.4 MB, 99.71% of held-out
# predictions unchanged, every change a near-tie). Each copy is checked against its own SHA-256 at start-up and reports
# its own model_version, so stored predictions say which copy made them.
MODEL_VARIANTS = {
    "fp32": ("neuroscan_r30a_s123.onnx", "4937ab381ee3e7c29d5ab18ea761a8c19ddffb6488210fb6030511a8041502eb",
             "r30a_s123_convnext_tiny_onnx"),
    "int8_dynamic": ("neuroscan_r30a_s123_int8_dynamic.onnx",
                     "ad2f6e0ee22b28ec98295b742075a2168e480801073a8df3f1222df37c74d99f",
                     "r30a_s123_convnext_tiny_onnx_int8_dynamic"),
}
MODEL_VARIANT = os.environ.get("NEUROSCAN_MODEL_VARIANT", "fp32").strip().lower()
if MODEL_VARIANT not in MODEL_VARIANTS:
    raise RuntimeError(f"NEUROSCAN_MODEL_VARIANT must be one of {sorted(MODEL_VARIANTS)}, not {MODEL_VARIANT!r}.")
_MODEL_NAME, EXPECTED_SHA256, MODEL_VERSION = MODEL_VARIANTS[MODEL_VARIANT]
MODEL_FILE = MODELS_DIR / _MODEL_NAME

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


def process_memory_mb() -> dict | None:
    """Current (VmRSS) and peak (VmHWM) resident memory of this process, from /proc on Linux; None elsewhere."""
    try:
        out = {}
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(("VmRSS", "VmHWM")):
                    out["rss" if line.startswith("VmRSS") else "peak"] = int(line.split()[1]) // 1024
        return out or None
    except OSError:
        return None


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
        t0 = time.perf_counter()
        merged, pathway, base = build_input(modality, image, prev, nxt, adc)
        x = np.stack([_to_array(v) for v in _tta_views(merged)])
        t1 = time.perf_counter()
        logits, cams = self.session.run(["logits", "gradcam"], {"image": x})
        t2 = time.perf_counter()
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
        # Hemorrhage on MRI is outside the evaluated scope (no hemorrhagic MRI source): the prediction is returned
        # unchanged but marked as not validated; the platform shows it as a "not validated" tier with no alert.
        unsupported = modality == "MRI" and prediction == "hemorrhagic"
        if unsupported:
            warnings.append("Hemorrhagic prediction on MRI: not a validated finding, because the model was trained and "
                            "evaluated on hemorrhage on CT only.")
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
            "decision_note": "not_validated: hemorrhage on MRI" if unsupported else None,
        }
        try:
            color = ((127, 140, 141) if unsupported
                     else self._risk_tier_color(prediction, prob_normal, prob_ischemic, prob_hemorrhagic))
            overlay = self._overlay_heatmap(base, self._display_cam(cams[0, idx]), color,
                                            prediction in ("ischemic", "hemorrhagic"))
            buf = io.BytesIO()
            overlay.save(buf, format="PNG")
            result["heatmap_png_base64"] = base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            pass
        t3 = time.perf_counter()
        # Server-side time per stage (paper 2, E1): separates the model's time from the network and the platform.
        result["timing_ms"] = {"preprocess": round(1000 * (t1 - t0), 1), "model": round(1000 * (t2 - t1), 1),
                               "postprocess": round(1000 * (t3 - t2), 1), "total": round(1000 * (t3 - t0), 1)}
        return result


_pipeline: ThreeClassPipeline | None = None


def get_pipeline() -> ThreeClassPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ThreeClassPipeline()
    return _pipeline
