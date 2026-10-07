"""
NeuroScan inference API -- Render-hosted version.

Headless copy of webapp/app.py's /api/predict and /api/health routes,
backed by the same inference.ThreeClassPipeline (inference.py in this
directory is an unmodified copy of webapp/inference.py). No static UI
here -- this is called only by the PHP site's includes/inference_client.php
and, indirectly, the Android app.

Adds an X-API-Key check since this service is reachable at a public URL
around the clock: anyone with the URL could otherwise call your model.

v2 (final model): /api/predict takes `modality` (CT or MRI; default CT for older clients) and, besides `file`
(the current CT slice, or the MRI DWI image), optional `prev_file` + `next_file` (CT adjacent slices) or `adc_file`
(MRI, required: MRI without ADC is refused with 400). CT with `file` alone runs the single-image fallback and the
response says so (see inference.py).
The expected key is read from the NEUROSCAN_API_KEY environment variable
(set in the Render dashboard) rather than hardcoded, matching the
project's standing rule of never committing real secrets to any repo.
"""
import io
import os

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from inference import InputError, get_pipeline

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
API_KEY = os.environ.get("NEUROSCAN_API_KEY", "")

app = FastAPI(title="NeuroScan Inference API")


@app.on_event("startup")
def load_models():
    if not API_KEY:
        raise RuntimeError(
            "NEUROSCAN_API_KEY is not set. Configure it in the Render dashboard "
            "(Environment tab) before this service can start."
        )
    get_pipeline()


def _check_api_key(x_api_key: str = Header(default="")):
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key.")


@app.get("/api/health")
def health(x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    pipeline = get_pipeline()
    return {
        "status": "ok",
        "checkpoint": {"epoch": pipeline.epoch, "val_acc": pipeline.val_acc, "model_version": "r30a_s123_convnext_tiny_onnx",
                       "weights_sha256": pipeline.weights_sha256},
    }


async def _read_image(upload: UploadFile | None, what: str) -> Image.Image | None:
    if upload is None:
        return None
    if upload.content_type is None or not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"{what} must be an image.")
    raw = await upload.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail=f"{what} is too large (10 MB limit).")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail=f"Could not read {what.lower()} as an image.")
    return image


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    prev_file: UploadFile | None = File(default=None),
    next_file: UploadFile | None = File(default=None),
    adc_file: UploadFile | None = File(default=None),
    modality: str = Form(default="CT"),
    x_api_key: str = Header(default=""),
):
    _check_api_key(x_api_key)
    modality = modality.strip().upper()
    image = await _read_image(file, "The image")
    prev = await _read_image(prev_file, "The previous slice")
    nxt = await _read_image(next_file, "The next slice")
    adc = await _read_image(adc_file, "The ADC image")

    pipeline = get_pipeline()
    try:
        result = pipeline.predict(modality, image, prev, nxt, adc)
    except InputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    result["filename"] = file.filename
    return result
