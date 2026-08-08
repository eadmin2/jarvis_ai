#!/usr/bin/env python3
"""Jarvis GPU STT server — run on any machine with an NVIDIA GPU.

Gives the voice pipeline big-model transcription (~0.2s for a sentence on a
modern GPU) while the host machine keeps a small local model as fallback.

Setup (Windows or Linux, NVIDIA driver installed):
    python -m venv .venv
    .venv/Scripts/pip install faster-whisper fastapi uvicorn nvidia-cublas-cu12 nvidia-cudnn-cu12
    set JARVIS_STT_TOKEN=<same value as JARVIS_HUD_TOKEN on the server>
    .venv/Scripts/python stt_server.py

Then point the voice server's `stt.remote.url` at http://this-machine:8768/stt

API:
    POST /stt    body: raw int16 16 kHz mono PCM, header X-Jarvis-Token
                 -> {"text": "...", "confidence": 0..1, "language": "..."}
    GET  /health -> {"status":"ok", ...}
"""
import os
import pathlib
import sys
import time

# Make the pip-installed NVIDIA cuBLAS/cuDNN DLLs loadable for ctranslate2.
# NOTE: use sys.prefix — site.getsitepackages() misses the venv on Windows.
_nv = pathlib.Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
for sub in ("cublas/bin", "cudnn/bin", "cuda_nvrtc/bin"):
    p = _nv / sub
    if p.exists():
        os.add_dll_directory(str(p))
        os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")

import numpy as np  # noqa: E402 - CUDA DLL paths must be installed first
import uvicorn  # noqa: E402
from fastapi import FastAPI, Request, Response  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402

MODEL_NAME = os.environ.get("JARVIS_STT_MODEL", "large-v3-turbo")
TOKEN = os.environ.get("JARVIS_STT_TOKEN", "")
PORT = int(os.environ.get("JARVIS_STT_PORT", "8768"))

print(f"Loading {MODEL_NAME} on CUDA...", flush=True)
t0 = time.time()
model = WhisperModel(MODEL_NAME, device="cuda", compute_type="float16")
print(f"Model ready in {time.time()-t0:.1f}s", flush=True)

app = FastAPI(title="Jarvis GPU STT")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME, "device": "cuda"}


@app.post("/stt")
async def stt(request: Request):
    if TOKEN and request.headers.get("x-jarvis-token") != TOKEN:
        return Response(status_code=401, content="auth required")
    body = await request.body()
    if len(body) < 3200:  # <0.1s of audio
        return {"text": ""}
    audio = np.frombuffer(body[: len(body) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0
    t0 = time.time()
    try:
        configured_language = os.environ.get("JARVIS_STT_LANGUAGE") or None
        segment_iter, info = model.transcribe(
            audio, language=configured_language, beam_size=1, vad_filter=False,
        )
        segments = list(segment_iter)
        text = " ".join(s.text.strip() for s in segments).strip()
        weights = [max(0.01, float(s.end - s.start)) for s in segments]
        if weights:
            scores = [
                float(np.exp(min(0.0, s.avg_logprob))) * (1.0 - float(s.no_speech_prob))
                for s in segments
            ]
            confidence = float(np.average(scores, weights=weights))
        else:
            confidence = 0.0
        language = getattr(info, "language", configured_language)
    except Exception as exc:  # silence/garbage audio -> empty transcript, not a 500
        print(f"transcribe error -> empty: {exc}", flush=True)
        text = ""
        confidence = 0.0
        language = configured_language
    print(f"{len(audio)/16000:.1f}s audio -> {time.time()-t0:.2f}s : {text[:80]}", flush=True)
    return {"text": text, "confidence": round(confidence, 4), "language": language}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
