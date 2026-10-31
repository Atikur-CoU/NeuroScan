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
import hmac
import io
import os
import time

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from inference import MODEL_VARIANT, MODEL_VERSION, InputError, get_pipeline, process_memory_mb

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
# Same limit as the web app: a small, highly compressed file can otherwise decode into a huge pixel buffer,
# and the free tier has 512 MB. The size is read from the header, before decoding.
MAX_IMAGE_SIDE = 4096
API_KEY = os.environ.get("NEUROSCAN_API_KEY", "")
STARTED = time.time()

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
    if not API_KEY or not hmac.compare_digest(x_api_key.encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key.")


@app.get("/api/health")
def health(x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    pipeline = get_pipeline()
    return {
        "status": "ok",
        "checkpoint": {"epoch": pipeline.epoch, "val_acc": pipeline.val_acc, "model_version": MODEL_VERSION,
                       "model_variant": MODEL_VARIANT, "weights_sha256": pipeline.weights_sha256},
        # Paper 2, E1: process memory (MB; current and peak since start) and seconds since start-up.
        "memory_mb": process_memory_mb(),
        "uptime_s": round(time.time() - STARTED, 1),
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
        if max(image.size) > MAX_IMAGE_SIDE:
            raise HTTPException(status_code=400, detail=f"{what} is too large ({MAX_IMAGE_SIDE} px per side limit).")
        image.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        # Not an image, or a damaged one (e.g. truncated): a client error, not a server failure.
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
    t0 = time.perf_counter()
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
    # Whole request inside the server, upload reading and image decoding included (the rest is network and queueing).
    result.setdefault("timing_ms", {})["request"] = round(1000 * (time.perf_counter() - t0), 1)
    return result
