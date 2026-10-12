"""Loopback test: Piper TTS -> wav -> faster-whisper STT. Measures both latencies."""
import time, wave, sys, os
# Sandbox NO_PROXY contains bracketed IPv6 which breaks httpx; sanitize.
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

t0 = time.time()
from piper import PiperVoice
voice = PiperVoice.load("/home/hatch/workspace/pixalified-voice-agent/models/piper/en_US-amy-medium.onnx")
t_load = time.time() - t0
print(f"piper load: {t_load*1000:.0f} ms", flush=True)

text = "Hello! Thank you for calling Pixalified. We build websites, automation, and CRM systems for growing businesses. How can I help you today?"
t0 = time.time()
import numpy as np
chunks = []
for chunk in voice.synthesize(text):
    chunks.append((chunk.audio_float_array * 32767).astype(np.int16))
pcm = np.concatenate(chunks)
sr = voice.config.sample_rate
with wave.open("/tmp/piper_test.wav", "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
    wf.writeframes(pcm.tobytes())
t_tts = time.time() - t0
dur = len(pcm) / sr
print(f"piper synth: {t_tts*1000:.0f} ms for {dur:.1f}s audio (RTF={t_tts/dur:.2f})", flush=True)

t0 = time.time()
from faster_whisper import WhisperModel
model = WhisperModel("/home/hatch/workspace/pixalified-voice-agent/models/whisper-tiny", device="cpu", compute_type="int8")
t_wload = time.time() - t0
print(f"whisper load: {t_wload*1000:.0f} ms", flush=True)

t0 = time.time()
segments, info = model.transcribe("/tmp/piper_test.wav", beam_size=1)
transcript = " ".join(s.text.strip() for s in segments)
t_stt = time.time() - t0
print(f"whisper transcribe: {t_stt*1000:.0f} ms for {dur:.1f}s audio (RTF={t_stt/dur:.2f})", flush=True)
print(f"transcript: {transcript}", flush=True)
print(f"detected language: {info.language} (prob {info.language_probability:.2f})", flush=True)
