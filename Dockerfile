FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock .
# v2: ONNX Runtime instead of PyTorch (PyTorch alone needed ~300 MB; the 512 MB tier did not fit).
# Installs from the lock file (every package pinned and hash-checked), so a rebuild gets the same environment.
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

COPY app.py inference.py start.sh ./
RUN chmod +x start.sh

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["./start.sh"]
