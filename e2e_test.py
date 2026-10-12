"""End-to-end test: fake caller audio in -> transcript -> LLM (+tool) -> audio out.
Asserts the lead lands in leads.json and the reply audio is a real WAV."""
import base64
import io
import json
import os
import wave

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
# Sandbox NO_PROXY contains bracketed IPv6 entries that crash httpx; sanitize.
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
import sys
sys.path.insert(0, BASE)
os.environ["LLM_API_URL"] = "http://127.0.0.1:11435/v1"

# 1. Synthesize a caller utterance with Piper (acts as the "user's mic")
from piper import PiperVoice
voice = PiperVoice.load(os.path.join(BASE, "models", "piper", "en_US-amy-medium.onnx"))
caller_text = ("Hello, my name is Raj. I own a gym and I need a new website. "
               "My phone number is nine eight seven six five four three two one zero. "
               "Please have someone call me back.")
chunks = [(c.audio_float_array * 32767).astype(np.int16) for c in voice.synthesize(caller_text)]
pcm = np.concatenate(chunks)
buf = io.BytesIO()
with wave.open(buf, "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(voice.config.sample_rate)
    wf.writeframes(pcm.tobytes())
caller_wav = buf.getvalue()
print(f"caller audio: {len(pcm)/voice.config.sample_rate:.1f}s", flush=True)

# 2. Drive the real HTTP endpoint
import httpx
r = httpx.post("http://127.0.0.1:8765/api/turn",
               files={"audio": ("caller.wav", caller_wav, "audio/wav")},
               timeout=600)
r.raise_for_status()
d = r.json()

print("transcript:", d["transcript"], flush=True)
print("reply:     ", d["reply_text"], flush=True)
print("tool:      ", json.dumps(d["tool"]), flush=True)
print("latency_ms:", json.dumps(d["latency_ms"]), flush=True)

# 3. Assertions
assert d["transcript"] and len(d["transcript"]) > 10, "empty transcript"
assert d["reply_text"] and len(d["reply_text"]) > 5, "empty reply"
audio = base64.b64decode(d["audio_b64"])
assert audio[:4] == b"RIFF" and len(audio) > 10000, "reply audio is not a real WAV"
print(f"reply audio: {len(audio)/1024:.0f} KB wav", flush=True)

with open(os.path.join(BASE, "leads.json")) as f:
    leads = json.load(f)
assert leads, "leads.json empty - capture_lead did not fire"
last = leads[-1]
print("lead saved:", json.dumps(last), flush=True)
assert last.get("phone"), "lead has no phone number"

print("E2E PASS", flush=True)
