#!/bin/sh
# Downloads the model checkpoint from a GitHub Release asset on this same
# repo, at container startup -- not baked in at build time, so runtime
# environment variables (universally supported everywhere) are all this
# needs, no build-arg plumbing required.
#
# Uses a GitHub Release instead of Kaggle: the exact same credentials
# that work perfectly when tested directly against Kaggle's API return a
# 403 specifically from Render's network (most likely Kaggle blocking or
# restricting requests from cloud-hosting IP ranges for its newer-style
# API tokens) -- and we already know Render can reach GitHub reliably,
# since that's how it pulls this very repo to build from.
set -e
# v2: final model R30A s123 (ConvNeXt-Tiny exported to ONNX, ~111 MB); inference.py verifies its SHA-256.

if [ -z "$GITHUB_TOKEN" ]; then
    echo "FATAL: GITHUB_TOKEN must be set." >&2
    exit 1
fi

REPO="Atikur-CoU/NeuroScan"
RELEASE_TAG="checkpoint-v2-convnext"
ASSET_NAME="neuroscan_r30a_s123.onnx"

mkdir -p /app/models
if [ ! -f /app/models/neuroscan_r30a_s123.onnx ]; then
    echo "Looking up release asset..."
    ASSET_URL=$(curl -s \
        -H "Authorization: Bearer $GITHUB_TOKEN" \
        -H "Accept: application/vnd.github+json" \
        "https://api.github.com/repos/$REPO/releases/tags/$RELEASE_TAG" \
        | python3 -c "
import sys, json
data = json.load(sys.stdin)
matches = [a['url'] for a in data.get('assets', []) if a['name'] == '$ASSET_NAME']
if not matches:
    print('ASSET_NOT_FOUND', file=sys.stderr)
    sys.exit(1)
print(matches[0])
")

    echo "Downloading checkpoint from GitHub Release..."
    curl -sL \
        -H "Authorization: Bearer $GITHUB_TOKEN" \
        -H "Accept: application/octet-stream" \
        -o /app/models/neuroscan_r30a_s123.onnx \
        "$ASSET_URL"

    SIZE=$(wc -c < /app/models/neuroscan_r30a_s123.onnx)
    echo "Downloaded $SIZE bytes."
fi

echo "Starting server..."
exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}"
