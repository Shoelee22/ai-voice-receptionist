FROM python:3.11-slim

# ffmpeg for browser-audio -> 16kHz WAV conversion; curl for first-run model fetch
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py widget.html entrypoint.sh README.md ./
RUN chmod +x entrypoint.sh

EXPOSE 8765

ENTRYPOINT ["./entrypoint.sh"]
