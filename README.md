# Pixalified Voice Receptionist (in-house, zero cost)

Clean-room build. No cloud APIs, no API keys, no Jarvis code anywhere.
Browser demo: press 🎤, speak, Maya transcribes, thinks, and answers out loud.

## Run with Docker (recommended)

```bash
# 1. Start the voice server (models auto-download on first run, ~140 MB)
docker compose up --build

# 2. Point it at an LLM — any OpenAI-compatible endpoint:
#    - llama-server on your machine: default just works
#      (http://host.docker.internal:11435/v1)
#    - Ollama:   LLM_API_URL=http://host.docker.internal:11434/v1 docker compose up
#    - Or run the LLM in Docker too (needs a GGUF file):
#      LLM_GGUF_PATH=/path/to/qwen2.5-1.5b-q4_k_m.gguf \
#      LLM_API_URL=http://llm:8080/v1 \
#      docker compose --profile llm up --build
```

Then open http://127.0.0.1:8765/ in a browser, allow the mic, press 🎤.

## Architecture

```
browser mic (webm/opus)
  -> ffmpeg -> 16kHz mono wav
  -> faster-whisper tiny (int8, CPU)          [STT]
  -> qwen2.5:1.5b-instruct Q4_K_M via llama-server, OpenAI-compatible API  [brain]
  -> optional JSON tool call (capture_lead / book_appointment / handoff_to_owner)
  -> Piper en_US-amy-medium                   [TTS]
  -> wav back to browser
```

The brain runs on **llama-server** (`http://127.0.0.1:11435/v1`, OpenAI-compatible),
not Ollama: Ollama's bundled runner delivered 0.1–0.2 tok/s on this 2-vCPU VM
(unusable), while llama-server gives 2–6 tok/s on the identical GGUF weights.
Point `LLM_API_URL` at any OpenAI-compatible endpoint (including Ollama on a
bigger box) to swap it.

## Run

Terminal 1 — LLM brain:
```
~/llamacpp/build/bin/llama-server \
  -m ~/ollama/models/blobs/sha256-183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4 \
  -t 2 -c 1024 --port 11435
```

Terminal 2 — voice server (API :8765, widget at http://127.0.0.1:8765/):
```
cd ~/workspace/pixalified-voice-agent && ./venv/bin/python server.py
```

Then open http://127.0.0.1:8765/ in a browser, allow the mic, press 🎤.

Needs ~2.5 GB free RAM (LLM ~1.1 GB + STT/TTS ~0.7 GB). Below ~2 GB free the
OS OOM-kills the servers — that happened once during testing.

## Files

- `server.py` — FastAPI app: `POST /api/turn` (audio in → transcript, reply,
  tool result, reply audio, per-stage latencies), `GET /api/health`, `GET /` widget.
- `widget.html` — mic-button demo page.
- `e2e_test.py` — end-to-end test (synthesizes a caller with Piper, drives
  `/api/turn`, asserts transcript/reply/tool/audio and the lead JSON).
- `test_audio_loopback.py` — STT/TTS loopback latency check.
- `test_caller.wav` — sample caller audio.
- `models/piper/en_US-amy-medium.onnx` — TTS voice.
- `models/whisper-tiny/` — faster-whisper model files.
- `leads.json` / `appointments.json` — tool output (created on first tool call).

## Measured latencies (2 vCPU, no GPU)

| Stage | Measurement |
|---|---|
| STT (12.2 s audio) | 1271 ms (RTF ≈ 0.10) |
| LLM (trimmed prompt) | 8579 ms total (prompt 12.5 tok/s, decode ~2–6 tok/s by context length) |
| TTS (~9 s reply) | 2960 ms (RTF ≈ 0.33) |
| **Full turn** | **≈ 13 s** |

A long system prompt tripled LLM time (32 s) — prompt length dominates via
KV-cache size, so the prompt is kept tight.

## Honest tradeoffs

- **Speed**: ~13 s/turn on this VM. Usable for a click-to-demo, not for live
  phone calls. On a 4–8 vCPU box this stack should do ~3–5 s/turn. Streaming
  (tokens → sentence-by-sentence TTS → chunked audio) would cut perceived
  latency a lot — not implemented yet.
- **Brain quality**: 1.5B is the biggest model that fits this VM's RAM
  (3B would not load). It follows the tool JSON protocol and keeps replies
  short, but it is noticeably less sharp than a 7B+ model. It once invented
  a "$500" price with the loose prompt — the shipped prompt forbids inventing
  prices and it obeyed in testing.
- **STT accuracy**: tiny model got "Pixalified" as "Pixalfide" once and nails
  clear speech otherwise, including spoken digit strings → "9876543210".
- **Not yet**: no public URL (runs on localhost only — deployment is a later
  step), no real phone/SIP path, no streaming, single-turn context only
  (no conversation memory across turns).
