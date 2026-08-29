"""
NeuroScan web demo -- shared inference logic.

Single ResNet18 model, three classes (normal / ischemic / hemorrhagic),
one softmax over all three at once. Replaces the earlier chained
binary(Phase D) -> subtype(Phase F) design and the subtype-only
confidence-threshold heuristic, both retired after real-world testing
showed they could produce confident, contradictory answers on the same
image -- each ran its own separate softmax over only its own classes, so
"normal" and "hemorrhagic" were never actually competing against each
other for the same probability mass. See PROCESS.md Sec 20 and
QA_LOG.md Sec 17 for the full failure analysis.

Checkpoint: Phase O (PROCESS.md Sec 33), fine-tuned from Phase N with
ISLES2022 ischemic slices paired as (DWI, ADC, DWI) -- matching how
DWI/ADC are read together in real radiology practice -- while every
other row still uses the (prev, self, next) multi-slice merge. A single
uploaded image has no known prev/next slice or DWI/ADC pairing, so this
pipeline always takes the graceful-degrade path used at training time
for slices without neighbor context: channels = (self, self, self).

Inference also runs 4-view test-time augmentation (identity, horizontal
flip, +8 deg rotation, -8 deg rotation) on the merged image and averages
the softmax across views -- the same tta_views() pattern proven in
kaggle_kernels/phase_o_tta_eval/eval.py. Per PROCESS.md Sec 33.9, TTA on
top of Phase O was the best configuration found in the project.

models/threeclass_best.pt: epoch 18, val_acc 0.9675. Test accuracy
0.9736 (0.9707 single-pass), per-class test F1: ischemic 94.98%,
hemorrhagic 99.63%. The prior Phase G checkpoint is kept at
models/threeclass_best_phase_g_backup.pt for rollback.

Each prediction also includes a Grad-CAM [Selvaraju et al., 2017]
heatmap -- a red-tinted overlay showing which regions of the image most
influenced the predicted class -- as a base64-encoded PNG. Computed on
the identity (untransformed) view only, for the class TTA actually
predicted: the other 3 TTA views are flipped/rotated, and inverse-
transforming a heatmap back to align with the original image is
complexity this visualization doesn't need. A single uploaded image has
no prev/next or DWI/ADC context (see above), so there is exactly one
image to overlay the heatmap onto -- no ambiguity about which channel/
slice to show, unlike the multi-slice training pipeline.
"""
import base64
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

MODELS_DIR = Path(__file__).parent / "models"

TARGET_SIZE = (224, 224)
VALID_LABELS = ("normal", "ischemic", "hemorrhagic")
LABEL_TO_IDX = {"normal": 0, "ischemic": 1, "hemorrhagic": 2}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def _build_merged_image(image: Image.Image) -> Image.Image:
    """Single-image degrade-gracefully path: no known prev/next slice or
    DWI/ADC pairing, so channels = (self, self, self) -- matching the
    has_context=False branch used at training/eval time."""
    self_img = image.convert("L").resize(TARGET_SIZE)
    return Image.merge("RGB", (self_img, self_img, self_img))


def _tta_views(img: Image.Image):
    yield img
    yield transforms.functional.hflip(img)
    yield transforms.functional.rotate(img, 8, fill=0)
    yield transforms.functional.rotate(img, -8, fill=0)


def _to_tensor(img: Image.Image) -> torch.Tensor:
    return NORMALIZE(transforms.functional.to_tensor(img))


def _build_resnet18(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class ThreeClassPipeline:
    """Loads the single 3-class checkpoint once; call .predict(pil_image)
    per request. One softmax over normal/ischemic/hemorrhagic -- the
    three classes always compete for the same probability mass, so
    there's exactly one coherent answer per image, not two independently
    -trained models that can disagree."""

    def __init__(self):
        self.model = _build_resnet18(len(VALID_LABELS)).to(DEVICE)
        ckpt = torch.load(MODELS_DIR / "threeclass_best.pt", map_location=DEVICE)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.epoch = ckpt.get("epoch")
        self.val_acc = ckpt.get("val_acc")

        # Grad-CAM hooks on the final conv block (layer4), the standard
        # target for ResNet -- captures this instance's activations/
        # gradients each time it fires. Safe as plain instance attributes
        # (not thread-local) because this pipeline is a process-wide
        # singleton served by a single worker (see app.py/server_app.py).
        self._gradcam_activations = None
        self._gradcam_gradients = None
        self.model.layer4.register_forward_hook(self._save_activations)
        self.model.layer4.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inputs, output):
        self._gradcam_activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradcam_gradients = grad_output[0]

    def _grad_cam_overlay(self, identity_view: Image.Image, class_idx: int) -> Image.Image:
        """Forward+backward pass on the single identity view only (see
        module docstring), producing a red-tinted heatmap overlay for the
        given class. Runs under its own enable_grad() since predict()
        otherwise executes fully under no_grad()."""
        x = _to_tensor(identity_view).unsqueeze(0).to(DEVICE)
        with torch.enable_grad():
            x.requires_grad_(True)
            self.model.zero_grad(set_to_none=True)
            logits = self.model(x)
            logits[0, class_idx].backward()

        activations = self._gradcam_activations[0]  # (C, H, W)
        gradients = self._gradcam_gradients[0]  # (C, H, W)
        weights = gradients.mean(dim=(1, 2))  # (C,)
        cam = torch.relu((weights[:, None, None] * activations).sum(dim=0))  # (H, W)
        cam = cam / (cam.max() + 1e-8)

        cam_img = Image.fromarray((cam.detach().cpu().numpy() * 255).astype(np.uint8))
        cam_img = cam_img.resize(TARGET_SIZE, Image.BILINEAR)
        return self._overlay_heatmap(identity_view, cam_img)

    @staticmethod
    def _overlay_heatmap(base_image: Image.Image, cam_img: Image.Image) -> Image.Image:
        base_arr = np.asarray(base_image.convert("RGB"), dtype=np.float32)
        cam_arr = np.asarray(cam_img, dtype=np.float32) / 255.0  # (H, W) in [0, 1]
        red = np.zeros_like(base_arr)
        red[..., 0] = 255.0
        alpha = (cam_arr * 0.6)[..., None]
        blended = base_arr * (1 - alpha) + red * alpha
        return Image.fromarray(blended.astype(np.uint8))

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        merged = _build_merged_image(image)
        views = list(_tta_views(merged))
        x = torch.stack([_to_tensor(v) for v in views]).to(DEVICE)

        probs = torch.softmax(self.model(x), dim=1).mean(dim=0)
        prob_normal = probs[LABEL_TO_IDX["normal"]].item()
        prob_ischemic = probs[LABEL_TO_IDX["ischemic"]].item()
        prob_hemorrhagic = probs[LABEL_TO_IDX["hemorrhagic"]].item()

        idx_to_label = {v: k for k, v in LABEL_TO_IDX.items()}
        prediction = idx_to_label[probs.argmax().item()]

        result = {
            "prediction": prediction,
            "prob_normal": prob_normal,
            "prob_ischemic": prob_ischemic,
            "prob_hemorrhagic": prob_hemorrhagic,
        }

        # Best-effort: a heatmap failure should never take down a
        # prediction that otherwise succeeded.
        try:
            overlay = self._grad_cam_overlay(views[0], LABEL_TO_IDX[prediction])
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
