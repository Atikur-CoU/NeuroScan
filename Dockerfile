# --- Stage 1: fetch the model checkpoint from Kaggle -----------------------
# Isolated in its own stage so the Kaggle credentials used here never end up
# in the final image's layers -- only the downloaded checkpoint file is
# copied forward into the runtime stage below.
FROM python:3.11-slim AS fetcher

ARG KAGGLE_USERNAME
ARG KAGGLE_KEY

RUN pip install --no-cache-dir kaggle==1.6.17

RUN mkdir -p /root/.kaggle \
    && printf '{"username":"%s","key":"%s"}' "$KAGGLE_USERNAME" "$KAGGLE_KEY" > /root/.kaggle/kaggle.json \
    && chmod 600 /root/.kaggle/kaggle.json \
    && kaggle datasets download -d atik68/neuroscan-phase-o-checkpoints-final -f best.pt -p /checkpoint --force --unzip \
    && ls -la /checkpoint

# --- Stage 2: runtime --------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu

COPY app.py inference.py ./
COPY --from=fetcher /checkpoint/best.pt ./models/threeclass_best.pt

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
