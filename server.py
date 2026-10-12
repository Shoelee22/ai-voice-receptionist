"""
Pixalified in-house AI voice receptionist (clean-room build, no external services).

Pipeline per turn:
  browser mic (webm) -> ffmpeg -> wav 16k mono -> faster-whisper (STT)
  -> local LLM (llama-server, OpenAI-compatible) -> optional tool call
  -> Piper (TTS) -> wav back to browser

Run:
  1. Start the LLM:  ~/llamacpp/build/bin/llama-server -m <gguf> -t 2 -c 1024 --port 11435
     (or set LLM_API_URL to any OpenAI-compatible endpoint)
  2. ./venv/bin/python server.py   (serves API on :8765 and the widget at /)
"""
import base64
import io
import json
import os

# Sandbox NO_PROXY contains bracketed IPv6 entries that crash httpx's
# proxy-pattern parser. Sanitize before httpx is imported/used.
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

import re
import subprocess
import tempfile
import time
import wave
from datetime import datetime, timezone

import httpx
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse

BASE = os.path.dirname(os.path.abspath(__file__))
LLM_API_URL = os.environ.get("LLM_API_URL", "http://127.0.0.1:11435/v1").rstrip("/")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5-1.5b")
PIPER_MODEL = os.path.join(BASE, "models", "piper", "en_US-amy-medium.onnx")
WHISPER_MODEL = os.path.join(BASE, "models", "whisper-tiny")
LEADS_FILE = os.path.join(BASE, "leads.json")
APPTS_FILE = os.path.join(BASE, "appointments.json")

SYSTEM_PROMPT = """You are Maya, AI receptionist for Pixalified (websites, automation, CRM for small businesses; a VelocitiQ Technologies venture).
Reply in under 35 words, warm and conversational. Never invent prices or offers; if asked, say pricing depends on requirements and offer an owner callback.
Tools: to use one, output ONLY JSON like {"tool":"capture_lead","args":{"name":"...","phone":"...","interest":"..."}}.
Tools: capture_lead(name,phone,interest), book_appointment(name,phone,when,topic), handoff_to_owner(reason).
Ask for name and phone before capture_lead/book_appointment. If they want a human, use handoff_to_owner."""

app = FastAPI(title="Pixalified Voice Receptionist")

_whisper = None
_piper = None


def get_whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _whisper


def get_piper():
    global _piper
    if _piper is None:
        from piper import PiperVoice
        _piper = PiperVoice.load(PIPER_MODEL)
    return _piper


def convert_to_wav16k(raw: bytes) -> bytes:
    """Convert arbitrary browser audio (webm/opus) to 16kHz mono 16-bit WAV."""
    with tempfile.NamedTemporaryFile(suffix=".in", delete=False) as f:
        f.write(raw)
        in_path = f.name
    out_path = in_path + ".wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", in_path,
             "-ar", "16000", "-ac", "1", "-f", "wav", out_path],
            check=True, timeout=30,
        )
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        for p in (in_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def transcribe(wav_bytes: bytes) -> str:
    model = get_whisper()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        path = f.name
    try:
        segments, _info = model.transcribe(path, beam_size=1, language="en")
        return " ".join(s.text.strip() for s in segments).strip()
    finally:
        os.unlink(path)


def synthesize(text: str) -> bytes:
    voice = get_piper()
    chunks = [(c.audio_float_array * 32767).astype(np.int16)
              for c in voice.synthesize(text)]
    pcm = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(voice.config.sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def llm_reply(messages: list, max_tokens: int = 80) -> str:
    r = httpx.post(
        f"{LLM_API_URL}/chat/completions",
        json={"model": LLM_MODEL, "messages": messages,
              "max_tokens": max_tokens, "temperature": 0.7},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def parse_tool_call(text: str):
    """Return (tool_name, args) if the model emitted a JSON tool call, else None."""
    t = text.strip()
    # tolerate the model wrapping JSON in code fences
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and "tool" in obj and isinstance(obj.get("args"), dict):
        return obj["tool"], obj["args"]
    return None


def append_json(path: str, record: dict):
    rows = []
    if os.path.exists(path):
        try:
            with open(path) as f:
                rows = json.load(f)
        except (json.JSONDecodeError, OSError):
            rows = []
    rows.append(record)
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)


def run_tool(name: str, args: dict, transcript: str):
    ts = datetime.now(timezone.utc).isoformat()
    if name == "capture_lead":
        append_json(LEADS_FILE, {"ts": ts, "name": args.get("name", ""),
                                 "phone": args.get("phone", ""),
                                 "interest": args.get("interest", ""),
                                 "transcript": transcript})
        who = args.get("name") or "there"
        return {"ok": True,
                "say": f"Thanks {who}! I've saved your details and noted you're interested in {args.get('interest', 'our services')}. Someone from Pixalified will call you back shortly."}
    if name == "book_appointment":
        append_json(APPTS_FILE, {"ts": ts, "name": args.get("name", ""),
                                 "phone": args.get("phone", ""),
                                 "when": args.get("when", ""),
                                 "topic": args.get("topic", ""),
                                 "transcript": transcript})
        return {"ok": True,
                "say": f"Done! I've booked {args.get('when', 'a slot')} for you to discuss {args.get('topic', 'your requirements')}. We'll confirm on your phone shortly."}
    if name == "handoff_to_owner":
        return {"ok": True,
                "say": "Of course. I've flagged this for the owner and they will call you back shortly. Anything else I can help with?"}
    return {"ok": False, "say": "Sorry, I couldn't complete that request."}


@app.post("/api/turn")
async def voice_turn(audio: UploadFile = File(...)):
    """One full voice turn: audio in -> transcript -> reply (+ tool) -> audio out."""
    t_all = time.time()
    raw = await audio.read()

    t0 = time.time()
    wav = convert_to_wav16k(raw)
    t_convert = time.time() - t0

    t0 = time.time()
    transcript = transcribe(wav)
    t_stt = time.time() - t0
    if not transcript:
        transcript = "(no speech detected)"

    t0 = time.time()
    raw_reply = llm_reply([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": transcript},
    ])
    t_llm = time.time() - t0

    tool = parse_tool_call(raw_reply)
    if tool:
        name, targs = tool
        result = run_tool(name, targs, transcript)
        reply_text = result["say"]
        tool_info = {"name": name, "args": targs, "ok": result["ok"]}
    else:
        reply_text = raw_reply
        tool_info = None

    t0 = time.time()
    reply_wav = synthesize(reply_text)
    t_tts = time.time() - t0

    return JSONResponse({
        "transcript": transcript,
        "reply_text": reply_text,
        "tool": tool_info,
        "audio_b64": base64.b64encode(reply_wav).decode(),
        "latency_ms": {
            "convert": round(t_convert * 1000),
            "stt": round(t_stt * 1000),
            "llm": round(t_llm * 1000),
            "tts": round(t_tts * 1000),
            "total": round((time.time() - t_all) * 1000),
        },
    })


@app.get("/api/health")
def health():
    return {"ok": True, "llm": LLM_API_URL, "model": LLM_MODEL,
            "whisper": os.path.isdir(WHISPER_MODEL),
            "piper": os.path.exists(PIPER_MODEL)}


@app.get("/")
def widget():
    return FileResponse(os.path.join(BASE, "widget.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
