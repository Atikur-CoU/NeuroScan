FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu

COPY app.py inference.py start.sh ./
RUN chmod +x start.sh

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["./start.sh"]
