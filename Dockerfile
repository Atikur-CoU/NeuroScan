FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# v2: ONNX Runtime instead of PyTorch (PyTorch alone needed ~300 MB; the 512 MB tier did not fit).
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py inference.py start.sh ./
RUN chmod +x start.sh

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["./start.sh"]
