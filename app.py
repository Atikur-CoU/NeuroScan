"""
NeuroScan inference API -- Render-hosted version.

Headless copy of webapp/app.py's /api/predict and /api/health routes,
backed by the same inference.ThreeClassPipeline (inference.py in this
directory is an unmodified copy of webapp/inference.py). No static UI
here -- this is called only by the PHP site's includes/inference_client.php
and, indirectly, the Android app.

Adds an X-API-Key check since this service is reachable at a public URL
around the clock: anyone with the URL could otherwise call your model.
The expected key is read from the NEUROSCAN_API_KEY environment variable
(set in the Render dashboard) rather than hardcoded, matching the
project's standing rule of never committing real secrets to any repo.
"""
import io
import os

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from inference import get_pipeline

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
        "checkpoint": {"epoch": pipeline.epoch, "val_acc": pipeline.val_acc},
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)

    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="Image too large (10 MB limit).")

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not read this file as an image.")

    pipeline = get_pipeline()
    result = pipeline.predict(image)
    result["filename"] = file.filename
    return result
