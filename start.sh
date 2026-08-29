#!/bin/sh
# Downloads the model checkpoint at container startup (not build time) --
# runtime environment variables work identically on every host, unlike
# Docker build-time ARGs, whose availability depends on platform-specific
# behavior that isn't worth depending on. Costs a few extra seconds on
# each cold start; the free tier already cold-starts on idle wake, so
# this doesn't introduce a new class of delay, just extends the existing one.
set -e

if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
    echo "FATAL: KAGGLE_USERNAME and KAGGLE_KEY must be set." >&2
    exit 1
fi

# Diagnostic only -- lengths and first/last char, never the full value.
# Expected: username length 6 (atik68), key length 37 (KGAT_ + 32 hex chars).
echo "KAGGLE_USERNAME: length=${#KAGGLE_USERNAME} first=${KAGGLE_USERNAME%"${KAGGLE_USERNAME#?}"} last=${KAGGLE_USERNAME#"${KAGGLE_USERNAME%?}"}"
echo "KAGGLE_KEY: length=${#KAGGLE_KEY} first=${KAGGLE_KEY%"${KAGGLE_KEY#?}"} last=${KAGGLE_KEY#"${KAGGLE_KEY%?}"}"

mkdir -p /root/.kaggle
printf '{"username":"%s","key":"%s"}' "$KAGGLE_USERNAME" "$KAGGLE_KEY" > /root/.kaggle/kaggle.json
chmod 600 /root/.kaggle/kaggle.json

mkdir -p /app/models
if [ ! -f /app/models/threeclass_best.pt ]; then
    echo "Downloading checkpoint from Kaggle..."
    kaggle datasets download -d atik68/neuroscan-phase-o-checkpoints-final -f best.pt -p /app/models --force --unzip
    mv /app/models/best.pt /app/models/threeclass_best.pt
fi

echo "Starting server..."
exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}"
