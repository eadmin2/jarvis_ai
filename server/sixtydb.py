"""60db REST synthesis for the browser's mono PCM16/16 kHz contract."""
import base64
import io
import json
import math
import os
import wave

import requests

MAX_BYTES = 32 * 1024 * 1024


def _metadata(record, depth=0):
    if depth > 4 or not isinstance(record, dict):
        raise RuntimeError("60db returned invalid audio metadata")
    if record.get("success") is False or record.get("error") or record.get("type") == "error":
        raise RuntimeError("60db reported a synthesis failure")
    for key in ("encoding", "audio_encoding", "output_format"):
        if key in record and str(record[key]).lower() not in {"linear16", "pcm", "pcm16", "wav"}:
            raise RuntimeError("60db returned compressed audio instead of PCM16")
    for key, expected in (("sample_rate", 16000), ("sample_rate_hertz", 16000), ("channels", 1), ("bit_depth", 16)):
        if key in record and record[key] != expected:
            raise RuntimeError("60db returned incompatible PCM metadata")
    if "audio_config" in record:
        _metadata(record["audio_config"], depth + 1)


def _decode(record, depth=0):
    if depth > 4:
        raise RuntimeError("60db audio envelope is too deeply nested")
    _metadata(record)
    result = record.get("result", record.get("backendResponse", record))
    _metadata(result)
    value = result.get("audioContent", result.get("audio_base64"))
    if value is None:
        return b""
    if not isinstance(value, str):
        raise RuntimeError("60db audio must be base64 text")
    audio = base64.b64decode(value, validate=True)
    if audio.startswith(b"{"):
        audio = _decode(json.loads(audio), depth + 1)
    return _pcm(audio)


def _pcm(audio):
    if audio.startswith(b"RIFF"):
        with wave.open(io.BytesIO(audio), "rb") as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, 16000, "NONE"):
                raise RuntimeError("60db WAV must be mono PCM16 at 16000 Hz")
            frames = wav.getnframes()
            audio = wav.readframes(frames)
            if len(audio) != frames * 2:
                raise RuntimeError("60db returned truncated WAV audio")
    elif audio.startswith((b"ID3", b"OggS", b"fLaC")):
        raise RuntimeError("60db returned compressed audio instead of PCM16")
    return audio


def tts_chunks(text, voice, stopped=None):
    """Yield aligned PCM; cooperative cancellation is bounded by the read timeout."""
    key = os.environ.get("SIXTYDB_API_KEY", "").strip()
    voice_id = str(voice.get("voice_id") or "").strip()
    speed = float(voice.get("speed", 1))
    if not key or not voice_id:
        raise RuntimeError("60db requires SIXTYDB_API_KEY and a workspace voice_id")
    if not isinstance(text, str) or not text.strip() or len(text) > 5000:
        raise RuntimeError("60db text must contain 1 to 5000 characters")
    if not math.isfinite(speed) or not 0.5 <= speed <= 2:
        raise RuntimeError("60db speed must be between 0.5 and 2")
    if voice.get("output_format", "pcm_16000") != "pcm_16000":
        raise RuntimeError("60db output_format must be pcm_16000 for browser playback")
    if stopped is not None and stopped.is_set():
        return
    payload = {"text": text, "voice_id": voice_id, "speed": speed,
               "audio_config": {"audio_encoding": "LINEAR16", "sample_rate_hertz": 16000},
               "output_format": "wav", "timestamp_type": "NONE"}
    if voice.get("model"):
        payload["model_id"] = voice["model"]
    with requests.post("https://api.60db.ai/tts-synthesize", json=payload,
                       headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                       stream=True, timeout=(10, 30), allow_redirects=False) as response:
        if response.status_code != 200:
            raise RuntimeError(f"60db TTS failed (HTTP {response.status_code})")
        _metadata({key: int(response.headers[header])
                   for key, header in (("sample_rate", "X-Sample-Rate"), ("channels", "X-Channels"), ("bit_depth", "X-Bit-Depth"))
                   if header in response.headers})
        kind = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
        streaming = kind in {"application/x-ndjson", "application/ndjson", "text/plain"}
        if not streaming and kind not in {"application/json", "audio/wav", "audio/x-wav", "audio/pcm", "application/octet-stream"}:
            raise RuntimeError("60db returned an unsupported content type")
        pending, pcm = bytearray(), bytearray()
        total, produced = 0, False
        for chunk in response.iter_content(chunk_size=4096):
            if stopped is not None and stopped.is_set():
                return
            total += len(chunk)
            if total > MAX_BYTES:
                raise RuntimeError("60db response exceeds 32 MiB")
            pending.extend(chunk)
            if streaming:
                while b"\n" in pending:
                    line, _, remainder = pending.partition(b"\n")
                    pending = bytearray(remainder)
                    if line.strip():
                        pcm.extend(_decode(json.loads(line)))
                size = len(pcm) - len(pcm) % 2
                if size:
                    produced = True
                    yield bytes(pcm[:size])
                    del pcm[:size]
        if stopped is not None and stopped.is_set():
            return
        if streaming:
            if pending.strip():
                pcm.extend(_decode(json.loads(pending)))
        elif kind == "application/json":
            pcm.extend(_decode(json.loads(pending)))
        else:
            pcm.extend(_pcm(bytes(pending)))
        if len(pcm) % 2:
            raise RuntimeError("60db returned an incomplete PCM16 sample")
        for offset in range(0, len(pcm), 4096):
            if stopped is not None and stopped.is_set():
                return
            produced = True
            yield bytes(pcm[offset:offset + 4096])
        if not produced:
            raise RuntimeError("60db returned no audio")
