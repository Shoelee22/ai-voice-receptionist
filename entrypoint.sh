#!/bin/bash
# First-run model fetch: downloads STT + TTS models into /app/models if absent.
# Models persist in the 'voice-models' Docker volume across restarts.
set -e

mkdir -p /app/models/whisper-tiny /app/models/piper

WHISPER_BASE="https://huggingface.co/Systran/faster-whisper-tiny/resolve/main"
for f in config.json model.bin tokenizer.json vocabulary.txt; do
  if [ ! -f "/app/models/whisper-tiny/$f" ]; then
    echo "downloading whisper-tiny/$f ..."
    curl -sSL -o "/app/models/whisper-tiny/$f" "$WHISPER_BASE/$f"
  fi
done

PIPER_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium"
if [ ! -f "/app/models/piper/en_US-amy-medium.onnx" ]; then
  echo "downloading piper en_US-amy-medium ..."
  curl -sSL -o "/app/models/piper/en_US-amy-medium.onnx" "$PIPER_BASE/en_US-amy-medium.onnx"
  curl -sSL -o "/app/models/piper/en_US-amy-medium.onnx.json" "$PIPER_BASE/en_US-amy-medium.onnx.json"
fi

echo "models ready, starting voice server ..."
exec uvicorn server:app --host 0.0.0.0 --port 8765
