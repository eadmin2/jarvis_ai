#!/usr/bin/env python3
"""Hermes LAN voice pipeline server (v3 — sessions, stop, approvals, partials).

WebSocket protocol (client → server):
  {"type":"start", "sample_rate":16000, "format":"pcm_s16le", "channels":1,
   "conversation": "jarvis-main"?}          begin a turn (mid-turn = barge-in)
  <binary int16 16 kHz mono PCM chunks>
  {"type":"stop"}                            end of speech, process turn
  {"type":"stop_run"}                        halt the running agent turn
  {"type":"approval_decision", "run_id":..., "approval_id":..., "decision":"allow"|"deny"}

Server → client JSON events:
  status, transcript, partial_transcript, agent_status{thinking|tool_use|speaking},
  run_started{run_id}, approval_request{...}, error, done{timing}
plus binary 16 kHz mono int16 PCM TTS audio.

Brain: Hermes Agent API server via the Sessions API (/api/sessions/{id}/chat/stream),
which provides persistent conversation memory, run ids (stoppable), tool events,
and approval events. Falls back to direct Anthropic ("basic mode") if unreachable.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Iterator

import requests
import uvicorn
import yaml
import numpy as np
from anthropic import Anthropic
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from RealtimeSTT import AudioToTextRecorder

from brain import GbrainContextAdapter, HermesAdapter
from voice import (
    CancellationToken,
    PronunciationDictionary,
    TTSMetrics,
    TTSProvider,
    create_tts_provider,
)

try:
    import psutil
except ImportError:  # machines panel degrades gracefully
    psutil = None

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = Path(
    os.environ.get("JARVIS_CONFIG_PATH", ROOT / "config" / "server.yaml")
).expanduser()
LOG_PATH = ROOT / "logs" / "latency.jsonl"
STATE_PATH = ROOT / "logs" / "hermes_sessions.json"
USAGE_PATH = ROOT / "logs" / "usage_stats.json"
_USAGE_LOCK = threading.Lock()


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def record_usage(llm_in: int = 0, llm_out: int = 0, turns: int = 0, tts_chars: int = 0) -> None:
    """Accumulate token/character usage into logs/usage_stats.json (total + per-day)."""
    with _USAGE_LOCK:
        try:
            data = json.loads(USAGE_PATH.read_text(encoding="utf-8"))
        except Exception:
            data = {"total": {}, "days": {}}
        day = data["days"].setdefault(_today(), {})
        for bucket in (data["total"], day):
            bucket["llm_in"] = bucket.get("llm_in", 0) + llm_in
            bucket["llm_out"] = bucket.get("llm_out", 0) + llm_out
            bucket["turns"] = bucket.get("turns", 0) + turns
            bucket["tts_chars"] = bucket.get("tts_chars", 0) + tts_chars
        # keep last 60 days
        for k in sorted(data["days"])[:-60]:
            del data["days"][k]
        USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        USAGE_PATH.write_text(json.dumps(data), encoding="utf-8")


def read_usage() -> dict:
    with _USAGE_LOCK:
        try:
            data = json.loads(USAGE_PATH.read_text(encoding="utf-8"))
        except Exception:
            data = {"total": {}, "days": {}}
    return {"total": data.get("total", {}), "today": data.get("days", {}).get(_today(), {})}


def record_brain_usage(payload: dict) -> None:
    usage = payload.get("usage") or {}
    if not isinstance(usage, dict) or not usage:
        return
    record_usage(
        llm_in=int(usage.get("input_tokens") or 0),
        llm_out=int(usage.get("output_tokens") or 0),
        turns=1,
    )
_ENV_OVERRIDE = os.environ.get("JARVIS_ENV_PATH")
ENV_PATHS = (
    [Path(_ENV_OVERRIDE).expanduser()]
    if _ENV_OVERRIDE
    else [Path.home() / ".hermes" / ".env", ROOT / ".env"]
)
SENTENCE_RE = re.compile(r"(.+?[.!?])(?=\s|$)", re.DOTALL)
CLAUSE_RE = re.compile(r"(.{45,}?[,;:])(?=\s|$)", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
CODEBLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
# Secret-shaped strings are never sent to cloud TTS (privacy filter):
SECRET_RES = [
    re.compile(r"\b(?:api[_-]?key|secret|password|passwd|token|bearer|authorization)\b\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"\b(?:sk|pk|key|tok|ghp|xox[abp])[-_][A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"\b[A-Za-z0-9+/_\-]{36,}\b"),          # long opaque blobs (keys, JWT segments)
    re.compile(r"-----BEGIN [A-Z ]+-----.*?-----END [A-Z ]+-----", re.DOTALL),
]


def load_env() -> None:
    for path in ENV_PATHS:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@dataclass
class TurnTiming:
    turn_id: int
    audio_start_monotonic: float | None = None
    end_of_speech_monotonic: float | None = None
    stt_start_monotonic: float | None = None
    stt_final_monotonic: float | None = None
    llm_start_monotonic: float | None = None
    llm_first_token_monotonic: float | None = None
    first_sentence_monotonic: float | None = None
    tts_request_start_monotonic: float | None = None
    first_tts_audio_byte_monotonic: float | None = None
    total_done_monotonic: float | None = None
    transcript: str = ""
    response_text: str = ""
    stt_model: str = ""
    llm_provider: str = ""
    llm_model: str = ""
    tts_model: str = ""
    tts_provider: str = ""
    voice_id: str = ""
    stt_language: str = ""
    stt_confidence: float | None = None
    tts_attempted_providers: list[str] = field(default_factory=list)
    tts_fallback_active: bool = False
    run_id: str = ""
    interrupted: bool = False
    tools_used: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    gbrain_context_used: bool = False

    def summary(self) -> dict:
        eos = self.end_of_speech_monotonic
        return {
            "turn_id": self.turn_id,
            "transcript": self.transcript,
            "response_text": self.response_text,
            "stt_model": self.stt_model,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "tts_model": self.tts_model,
            "tts_provider": self.tts_provider,
            "voice_id": self.voice_id,
            "stt_language": self.stt_language,
            "stt_confidence": self.stt_confidence,
            "tts_attempted_providers": self.tts_attempted_providers,
            "tts_fallback_active": self.tts_fallback_active,
            "run_id": self.run_id,
            "interrupted": self.interrupted,
            "tools_used": self.tools_used,
            "stt_finalize_seconds": self._delta(self.stt_start_monotonic, self.stt_final_monotonic),
            "llm_time_to_first_token_seconds": self._delta(self.llm_start_monotonic, self.llm_first_token_monotonic),
            "time_to_first_tts_audio_byte_seconds": self._delta(self.tts_request_start_monotonic, self.first_tts_audio_byte_monotonic),
            "end_of_speech_to_first_audio_seconds": self._delta(eos, self.first_tts_audio_byte_monotonic),
            "total_turn_seconds": self._delta(eos, self.total_done_monotonic),
            "errors": self.errors,
            "gbrain_context_used": self.gbrain_context_used,
        }

    @staticmethod
    def _delta(start: float | None, end: float | None) -> float | None:
        if start is None or end is None:
            return None
        return round(end - start, 4)


# ===================================================================== Hermes


class HermesAPI:
    """Thin client for the Hermes Agent API server (sessions, runs, approvals)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg.get("hermes") or {}

    @property
    def base(self) -> str:
        return (self.cfg.get("base_url") or "http://127.0.0.1:8642").rstrip("/")

    def headers(self) -> dict:
        key = os.environ.get(self.cfg.get("api_key_env", "API_SERVER_KEY"), "")
        if not key:
            raise RuntimeError("Hermes API key not found in environment")
        h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        if self.cfg.get("session_key"):
            h["X-Hermes-Session-Key"] = self.cfg["session_key"]
        return h

    # ---- persistent named sessions ----
    def _load_state(self) -> dict:
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_state(self, state: dict) -> None:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state), encoding="utf-8")

    def get_session_id(self, name: str, force_new: bool = False) -> str:
        state = self._load_state()
        sid = state.get(name)
        if sid and not force_new:
            return sid
        r = requests.post(f"{self.base}/api/sessions", headers=self.headers(),
                          json={"title": name}, timeout=15)
        r.raise_for_status()
        data = r.json()
        sid = (data.get("session") or data).get("id")
        state[name] = sid
        self._save_state(state)
        print(f"Created Hermes session '{name}' -> {sid}", flush=True)
        return sid

    def stop_run(self, run_id: str) -> dict:
        r = requests.post(f"{self.base}/v1/runs/{run_id}/stop", headers=self.headers(), timeout=15)
        return {"status_code": r.status_code, "body": r.text[:300]}

    def post_approval(self, run_id: str, body: dict) -> dict:
        r = requests.post(f"{self.base}/v1/runs/{run_id}/approval", headers=self.headers(),
                          json=body, timeout=15)
        return {"status_code": r.status_code, "body": r.text[:300]}

    def chat_stream_events(self, session_id: str, input_text: str, timeout: float) -> Iterator[tuple[str, str]]:
        """Yield ("run"|"text"|"tool"|"approval"|"final", value) from a session turn."""
        resp = requests.post(
            f"{self.base}/api/sessions/{session_id}/chat/stream",
            headers={**self.headers(), "Accept": "text/event-stream"},
            json={"input": input_text}, stream=True, timeout=(10, timeout),
        )
        if resp.status_code >= 400:
            resp.close()
            raise RuntimeError(f"Hermes session chat HTTP {resp.status_code}: {resp.text[:300]}")
        resp.encoding = "utf-8"  # SSE has no charset header; requests would assume latin-1 (mojibake)
        try:
            yield from self._parse_sse(resp)
        finally:
            resp.close()  # leaked FDs killed the server once (launchd limit is tiny)

    @staticmethod
    def _parse_sse(resp) -> Iterator[tuple[str, str]]:
        event_name = ""
        for raw in resp.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            if raw.startswith("event: "):
                event_name = raw[7:].strip()
                continue
            if not raw.startswith("data: "):
                continue
            data_text = raw[6:].strip()
            try:
                data = json.loads(data_text)
            except json.JSONDecodeError:
                continue
            ev = event_name or data.get("event", "")
            if ev == "run.started":
                yield ("run", data.get("run_id") or "")
            elif ev == "assistant.delta":
                d = data.get("delta") or ""
                if d:
                    yield ("text", d)
            elif ev == "tool.started":
                name = data.get("tool_name") or "tool"
                if name.startswith("_"):
                    continue  # internal pseudo-tools like _thinking
                yield ("tool", json.dumps({"name": name, "preview": (data.get("preview") or "")[:200]}))
            elif "approval" in ev:
                yield ("approval", json.dumps(data)[:2000])
            elif ev == "assistant.completed":
                yield ("final", json.dumps({
                    "content": data.get("content") or "",
                    "interrupted": bool(data.get("interrupted")),
                }))
            elif ev in ("run.failed", "error"):
                raise RuntimeError(f"Hermes stream error: {data_text[:300]}")
            elif ev == "run.completed":
                usage = data.get("usage") or {}
                if usage:
                    record_usage(
                        llm_in=int(usage.get("input_tokens") or 0),
                        llm_out=int(usage.get("output_tokens") or 0),
                        turns=1,
                    )
            elif ev == "done":
                pass  # stream closes after this


# ==================================================================== Pipeline


class VoicePipelineServer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.turn_counter = 0
        brain_provider = str(
            (cfg.get("brain") or {}).get("provider")
            or cfg.get("llm", {}).get("provider", "hermes")
        ).lower()
        if brain_provider == "hermes":
            self.brain = HermesAdapter(cfg)
        else:
            self.brain = None
        self.gbrain_context = GbrainContextAdapter(cfg)
        # Compatibility for stop/approval call sites while the external
        # WebSocket protocol remains provider-neutral.
        self.hermes = self.brain
        self.stt_lock = asyncio.Lock()
        self._tts_provider: TTSProvider | None = None
        self.pronunciations = PronunciationDictionary(
            (cfg.get("voice") or {}).get("pronunciations") or {}
        )
        stt_language = cfg["stt"].get("language", "en")
        self.recorder = AudioToTextRecorder(
            model=cfg["stt"]["model"],
            use_microphone=False,
            spinner=False,
            device=cfg["stt"].get("device", "cpu"),
            compute_type=cfg["stt"].get("compute_type", "int8"),
            sample_rate=int(cfg["stt"].get("sample_rate", 16000)),
            language=stt_language,
            beam_size=1,
            faster_whisper_vad_filter=False,
            no_log_file=True,
        )

    def next_turn_id(self) -> int:
        self.turn_counter += 1
        return self.turn_counter

    async def transcribe(self, audio: bytes, timing: TurnTiming | None = None) -> str:
        if timing:
            timing.stt_start_monotonic = time.perf_counter()
        # 1) GPU worker (if configured and reachable) — big model, ~0.3s
        remote = self.cfg["stt"].get("remote") or {}
        if remote.get("url"):
            result = await asyncio.to_thread(self._remote_stt, audio, remote)
            if result is not None:
                text, confidence, language = result
                if timing:
                    timing.stt_model = f"remote:{remote.get('name', 'gpu')}"
                    timing.stt_confidence = confidence
                    if language:
                        timing.stt_language = language
                    timing.stt_final_monotonic = time.perf_counter()
                return text
        # 2) local Whisper fallback
        sample_rate = int(self.cfg["stt"].get("sample_rate", 16000))
        samples = (np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0).copy()
        try:
            async with self.stt_lock:
                self.recorder.feed_audio(samples, original_sample_rate=sample_rate)
                text = await asyncio.to_thread(self.recorder.perform_final_transcription, samples, True)
                self.recorder.clear_audio_queue()
        except Exception as exc:
            # near-silent audio can make whisper raise ("No clip timestamps found");
            # treat as empty transcript instead of failing the turn
            print(f"local STT error treated as empty transcript: {exc}", flush=True)
            text = ""
        if timing:
            timing.stt_final_monotonic = time.perf_counter()
        return (text or "").strip()

    def _remote_stt(
        self, audio: bytes, remote: dict,
    ) -> tuple[str, float | None, str | None] | None:
        """POST raw PCM to the GPU STT worker. None = unavailable (use fallback)."""
        headers = {"Content-Type": "application/octet-stream"}
        token = os.environ.get(remote.get("token_env", "JARVIS_HUD_TOKEN"), "")
        if token:
            headers["X-Jarvis-Token"] = token
        try:
            r = requests.post(remote["url"], data=audio, headers=headers,
                              timeout=float(remote.get("timeout", 6)))
            if r.ok:
                payload = r.json()
                confidence = payload.get("confidence")
                return (
                    (payload.get("text") or "").strip(),
                    float(confidence) if confidence is not None else None,
                    payload.get("language"),
                )
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------ LLM

    def stream_llm_events_sync(
        self,
        transcript: str,
        timing: TurnTiming,
        conversation: str,
        cancellation: CancellationToken,
    ) -> Iterator[tuple[str, str]]:
        llm = self.cfg["llm"]
        provider = str(
            (self.cfg.get("brain") or {}).get("provider") or llm["provider"]
        ).lower()
        timing.llm_start_monotonic = time.perf_counter()
        if provider == "hermes":
            try:
                brain_cfg = self.cfg.get(provider) or {}
                assert self.brain is not None
                if self.gbrain_context.enabled:
                    try:
                        context = self.gbrain_context.query(transcript, cancellation)
                        if context:
                            transcript = self.gbrain_context.augment(transcript, context)
                            timing.gbrain_context_used = True
                    except Exception as exc:
                        timing.errors.append(f"gbrain_context: {exc}")
                        print(f"Gbrain context unavailable ({type(exc).__name__}: {exc}); continuing", flush=True)
                session_id = self.brain.get_session_id(conversation)
                gen = self._brain_turn(
                    session_id,
                    transcript,
                    timing,
                    brain_cfg,
                    conversation,
                    cancellation,
                )
                first = next(gen)
            except StopIteration:
                return
            except Exception as exc:
                fb = brain_cfg.get("fallback_provider", "anthropic")
                print(f"{provider} unavailable ({type(exc).__name__}: {exc}); fallback={fb}", flush=True)
                timing.errors.append(f"{provider}_fallback: {exc}")
                if not fb:
                    raise
                yield ("text", "Agent backend offline. Running in basic mode. ")
                provider = fb
            else:
                yield first
                yield from gen
                return
        timing.llm_provider = provider
        timing.llm_model = llm["model"]
        if provider == "anthropic":
            key = os.environ.get(llm.get("api_key_env", "ANTHROPIC_API_KEY"))
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY not found")
            client = Anthropic(api_key=key)
            with client.messages.stream(
                model=llm["model"],
                max_tokens=int(llm.get("max_tokens", 220)),
                temperature=float(llm.get("temperature", 0.3)),
                system=self.cfg["persona"]["system_prompt"],
                messages=[{"role": "user", "content": transcript}],
            ) as stream:
                close_stream = getattr(stream, "close", None)
                unsubscribe = (
                    cancellation.register(close_stream)
                    if callable(close_stream)
                    else lambda: None
                )
                try:
                    for text in stream.text_stream:
                        if cancellation.cancelled:
                            break
                        if text and timing.llm_first_token_monotonic is None:
                            timing.llm_first_token_monotonic = time.perf_counter()
                        yield ("text", text)
                finally:
                    unsubscribe()
        else:
            raise RuntimeError(f"Unsupported LLM provider: {provider}")

    def _brain_turn(
        self,
        session_id: str,
        transcript: str,
        timing: TurnTiming,
        brain_cfg: dict,
        conversation: str,
        cancellation: CancellationToken,
    ) -> Iterator[tuple[str, str]]:
        assert self.brain is not None
        timing.llm_provider = self.brain.provider
        timing.llm_model = str(brain_cfg.get("model") or f"{self.brain.provider}-agent")
        timeout = float(brain_cfg.get("timeout", 240))
        try:
            it = self.brain.chat_stream_events(
                session_id, transcript, timeout, cancellation,
            )
            for kind, value in it:
                if kind == "text" and timing.llm_first_token_monotonic is None:
                    timing.llm_first_token_monotonic = time.perf_counter()
                if kind == "final":
                    record_brain_usage(json.loads(value))
                yield (kind, value)
        except RuntimeError as exc:
            # stale session id (e.g. Hermes DB reset) -> recreate once
            if "404" in str(exc):
                session_id = self.brain.get_session_id(conversation, force_new=True)
                for kind, value in self.brain.chat_stream_events(
                    session_id, transcript, timeout, cancellation,
                ):
                    if kind == "text" and timing.llm_first_token_monotonic is None:
                        timing.llm_first_token_monotonic = time.perf_counter()
                    if kind == "final":
                        record_brain_usage(json.loads(value))
                    yield (kind, value)
            else:
                raise

    # ------------------------------------------------------------------ TTS

    def _get_tts_provider(self) -> TTSProvider:
        if self._tts_provider is None:
            self._tts_provider = create_tts_provider(
                self.cfg["voice"], metrics_hook=self._record_tts_metrics,
            )
        return self._tts_provider

    @staticmethod
    def _record_tts_metrics(metrics: TTSMetrics) -> None:
        """Provider-neutral local usage accounting."""
        if not metrics.cancelled and metrics.error is None:
            record_usage(tts_chars=metrics.characters)

    def tts_chunks_sync(
        self,
        text: str,
        timing: TurnTiming,
        cancellation: CancellationToken | None = None,
    ) -> Iterator[bytes]:
        provider = self._get_tts_provider()
        timing.tts_provider = provider.name
        timing.tts_model = provider.model
        timing.voice_id = provider.voice_id
        timing.tts_request_start_monotonic = timing.tts_request_start_monotonic or time.perf_counter()
        for chunk in provider.stream(text, cancellation):
            served_provider = getattr(provider, "served_provider", None)
            attempted = list(getattr(provider, "attempted_providers", ()) or ())
            if served_provider:
                timing.tts_provider = served_provider
                timing.tts_attempted_providers = attempted
                timing.tts_fallback_active = len(attempted) > 1
                selected = next(
                    (
                        candidate
                        for candidate in (
                            getattr(provider, "primary", None),
                            getattr(provider, "fallback", None),
                        )
                        if candidate is not None and candidate.name == served_provider
                    ),
                    None,
                )
                if selected is not None:
                    timing.tts_model = selected.model
                    timing.voice_id = selected.voice_id
            if timing.first_tts_audio_byte_monotonic is None:
                timing.first_tts_audio_byte_monotonic = time.perf_counter()
            yield chunk

    # ------------------------------------------------------------- Turn flow

    async def stream_response_audio(
        self,
        ws: WebSocket,
        transcript: str,
        timing: TurnTiming,
        conn: "ConnState",
        cancellation: CancellationToken,
    ) -> None:
        pending = ""
        full_response: list[str] = []
        spoken = False
        await ws.send_json({"type": "agent_status", "state": "thinking"})

        q: asyncio.Queue = asyncio.Queue()

        async def forward() -> None:
            try:
                async for item in self._async_llm_events(
                    transcript, timing, conn.conversation, cancellation,
                ):
                    await q.put(item)
                await q.put(None)
            except Exception as exc:
                await q.put(exc)

        forward_task = asyncio.create_task(forward())
        try:
            while True:
                item = await q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                kind, value = item
                if kind == "run":
                    timing.run_id = value
                    conn.current_run_id = value
                    await ws.send_json({"type": "run_started", "run_id": value})
                    continue
                if kind == "tool":
                    info = json.loads(value)
                    timing.tools_used.append(info.get("name", "tool"))
                    await ws.send_json({"type": "agent_status", "state": "tool_use",
                                        "tool": info.get("name"), "preview": info.get("preview", "")})
                    continue
                if kind == "approval":
                    info = json.loads(value)
                    conn.pending_approval = {
                        "run_id": conn.current_run_id,
                        "approval_id": info.get("approval_id") or info.get("id"),
                    }
                    await ws.send_json({"type": "approval_request", "data": info,
                                        "run_id": conn.current_run_id})
                    self._arm_approval_timeout(ws, conn)
                    if (self.cfg.get("voice") or {}).get("speak_approvals", True):
                        await ws.send_json({"type": "agent_status", "state": "speaking"})
                        await self._speak_segment(
                            ws,
                            "Approval required. Review the request on screen.",
                            timing,
                            conn,
                            cancellation,
                        )
                    continue
                if kind == "final":
                    info = json.loads(value)
                    timing.interrupted = info.get("interrupted", False)
                    continue
                # kind == "text"
                full_response.append(value)
                pending += value
                sentences, pending = self._extract_complete_sentences(pending)
                for sentence in sentences:
                    clean = self._clean_for_tts(sentence)
                    if not clean:
                        continue
                    if timing.first_sentence_monotonic is None:
                        timing.first_sentence_monotonic = time.perf_counter()
                    if not spoken:
                        await ws.send_json({"type": "agent_status", "state": "speaking"})
                        spoken = True
                    await self._speak_segment(
                        ws, clean, timing, conn, cancellation,
                    )
            tail = self._clean_for_tts(pending.strip())
            if tail:
                await self._speak_segment(
                    ws, tail, timing, conn, cancellation,
                )
        finally:
            if not forward_task.done():
                forward_task.cancel()
        timing.response_text = "".join(full_response).strip()

    async def _async_llm_events(
        self,
        transcript: str,
        timing: TurnTiming,
        conversation: str,
        cancellation: CancellationToken,
    ) -> AsyncIterator[tuple[str, str]]:
        q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def worker() -> None:
            try:
                for item in self.stream_llm_events_sync(
                    transcript, timing, conversation, cancellation,
                ):
                    loop.call_soon_threadsafe(q.put_nowait, item)
                loop.call_soon_threadsafe(q.put_nowait, None)
            except Exception as exc:
                loop.call_soon_threadsafe(q.put_nowait, exc)

        worker_task = asyncio.create_task(asyncio.to_thread(worker))
        while True:
            item = await q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
        await worker_task

    async def _send_tts_sentence(
        self,
        ws: WebSocket,
        sentence: str,
        timing: TurnTiming,
        cancellation: CancellationToken,
    ) -> None:
        if not sentence:
            return
        q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def worker() -> None:
            try:
                for chunk in self.tts_chunks_sync(sentence, timing, cancellation):
                    loop.call_soon_threadsafe(q.put_nowait, chunk)
                loop.call_soon_threadsafe(q.put_nowait, None)
            except Exception as exc:
                loop.call_soon_threadsafe(q.put_nowait, exc)

        worker_task = asyncio.create_task(asyncio.to_thread(worker))
        while True:
            item = await q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            await ws.send_bytes(item)
        await worker_task

    async def _speak_segment(
        self,
        ws: WebSocket,
        sentence: str,
        timing: TurnTiming,
        conn: "ConnState",
        cancellation: CancellationToken,
    ) -> None:
        """Frame raw audio with text metadata so the HUD can acknowledge playback."""
        conn.segment_counter += 1
        segment_id = f"{timing.turn_id}:{conn.segment_counter}"
        conn.pending_segments[segment_id] = sentence
        conn.spoken_sentences.append(sentence)
        await ws.send_json({
            "type": "speech_segment",
            "segment_id": segment_id,
            "text": sentence,
        })
        await self._send_tts_sentence(ws, sentence, timing, cancellation)
        if not cancellation.cancelled:
            await ws.send_json({
                "type": "speech_segment_end",
                "segment_id": segment_id,
            })

    def _arm_approval_timeout(self, ws: WebSocket, conn: "ConnState") -> None:
        """Default-deny an unattended founder-local approval request."""
        if conn.approval_timeout_task and not conn.approval_timeout_task.done():
            conn.approval_timeout_task.cancel()
        pending = dict(conn.pending_approval or {})
        timeout = float(
            (self.cfg.get("security") or {}).get("approval_timeout_seconds", 60)
        )

        async def deny_later() -> None:
            try:
                await asyncio.sleep(max(1, timeout))
                if conn.pending_approval != pending or self.brain is None:
                    return
                run_id = str(pending.get("run_id") or "")
                approval_id = str(pending.get("approval_id") or "")
                if run_id:
                    await asyncio.to_thread(
                        self.brain.submit_approval,
                        run_id,
                        approval_id,
                        "deny",
                    )
                conn.pending_approval = None
                await ws.send_json({
                    "type": "status",
                    "message": "Approval timed out and was denied.",
                })
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                try:
                    await ws.send_json({
                        "type": "status",
                        "message": f"Approval timeout failed safely: {exc}",
                    })
                except Exception:
                    pass

        conn.approval_timeout_task = asyncio.create_task(deny_later())

    @staticmethod
    def _extract_complete_sentences(text: str) -> tuple[list[str], str]:
        segments: list[str] = []
        remaining = text
        while remaining:
            sentence = SENTENCE_RE.search(remaining)
            clause = CLAUSE_RE.search(remaining)
            candidates = [match for match in (sentence, clause) if match is not None]
            if not candidates:
                break
            match = min(candidates, key=lambda item: item.end())
            segment = match.group(1).strip()
            if segment:
                segments.append(segment)
            remaining = remaining[match.end():]
        return segments, remaining

    def _clean_for_tts(self, text: str) -> str:
        if not text:
            return ""
        text = THINK_RE.sub("", text)
        for pattern in SECRET_RES:                  # privacy: never speak secrets
            text = pattern.sub(" redacted ", text)
        text = CODEBLOCK_RE.sub(" code omitted. ", text)
        text = re.sub(r"`([^`]*)`", r"\1", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"^[\s>*#-]+", "", text)
        text = re.sub(r"[*_#]{1,3}([^*_#]+)[*_#]{1,3}", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()
        return self.pronunciations.normalize(text)

    def log_turn(self, timing: TurnTiming) -> None:
        timing.total_done_monotonic = timing.total_done_monotonic or time.perf_counter()
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        summary = timing.summary()
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        print("TURN TIMING", json.dumps(summary, ensure_ascii=False), flush=True)


load_env()
CFG = load_config()
HERMES = HermesAPI(CFG)   # lightweight API client - independent of the STT pipeline
BRAIN_PROVIDER = str(
    (CFG.get("brain") or {}).get("provider")
    or CFG.get("llm", {}).get("provider", "hermes")
).lower()
BRAIN = HermesAdapter(CFG)
PIPELINE: VoicePipelineServer | None = None


_PIPELINE_LOCK = threading.Lock()


def get_pipeline() -> VoicePipelineServer:
    """Lock prevents the four uvicorn listeners' startup hooks from racing
    into concurrent recorder inits (which crashed three of the four lifespans
    and silently killed the TLS ports)."""
    global PIPELINE
    if PIPELINE is None:
        with _PIPELINE_LOCK:
            if PIPELINE is None:
                PIPELINE = VoicePipelineServer(CFG)
    return PIPELINE


async def warm_pipeline() -> None:
    """Warm the local Whisper fallback in the BACKGROUND, exactly once (this
    hook fires once per uvicorn listener — there are four), and never let a
    warm failure take a listener down."""
    global _WARM_STARTED
    if _WARM_STARTED:
        return
    _WARM_STARTED = True

    async def warm() -> None:
        try:
            await asyncio.to_thread(get_pipeline)
            print("STT pipeline warmed.", flush=True)
        except Exception as exc:
            print(f"STT warm failed (remote STT still available): {exc}", flush=True)

    asyncio.get_running_loop().create_task(warm())


_WARM_STARTED = False


@asynccontextmanager
async def app_lifespan(_app: FastAPI):
    await warm_pipeline()
    yield


app = FastAPI(title="Hermes Voice Pipeline", lifespan=app_lifespan)


# ------------------------------------------------------------------ Auth

ALLOWED_ORIGIN_HOSTS = {"jarvis.local", "jarvis", "localhost", "127.0.0.1"}
ALLOWED_ORIGIN_HOSTS |= set((CFG.get("security") or {}).get("extra_origin_hosts") or [])


def hud_token() -> str | None:
    env_name = (CFG.get("security") or {}).get("hud_token_env", "JARVIS_HUD_TOKEN")
    return os.environ.get(env_name) or None


def _request_authed(request: Request) -> bool:
    token = hud_token()
    if not token:
        return True
    supplied = request.headers.get("x-jarvis-token") or request.cookies.get("jarvis_token")
    return supplied == token


@app.middleware("http")
async def api_auth_middleware(request: Request, call_next):
    if request.url.path.startswith("/api/") and not _request_authed(request):
        return Response(status_code=401, content="jarvis auth required")
    return await call_next(request)


def _ws_allowed(ws: WebSocket) -> bool:
    """Browsers send Origin (+cookie); native clients (PTT, tests) send neither."""
    origin = ws.headers.get("origin")
    if not origin:
        return True  # non-browser client on the LAN (Python PTT, e2e tests)
    from urllib.parse import urlparse
    host = (urlparse(origin).hostname or "").lower()
    if host not in ALLOWED_ORIGIN_HOSTS:
        return False
    token = hud_token()
    if not token:
        return True
    return ws.cookies.get("jarvis_token") == token or ws.query_params.get("token") == token


# --------------------------------------------------------------- HUD + proxy

HUD_DIR = ROOT / "hud"
ALLOWED_GET_PATHS = {
    "/health", "/health/detailed", "/v1/capabilities",
    "/v1/skills", "/v1/toolsets", "/api/jobs", "/api/sessions",
}


def _proxy_allowed(method: str, path: str) -> bool:
    if method == "GET":
        return path in ALLOWED_GET_PATHS or (
            path.startswith("/api/sessions/") and path.endswith("/messages")
        )
    if method == "POST":
        return path == "/v1/responses"
    return False


@app.api_route("/api/hermes/{path:path}", methods=["GET", "POST"])
async def hermes_proxy(path: str, request: Request) -> Response:
    target = "/" + path
    if not _proxy_allowed(request.method, target):
        return Response(status_code=403, content="path not allowed")
    hermes = HERMES
    body = await request.body()
    params = dict(request.query_params)

    def do_request() -> requests.Response:
        return requests.request(
            request.method, hermes.base + target, params=params,
            headers=hermes.headers(), data=body if body else None, timeout=300,
        )

    resp = await asyncio.to_thread(do_request)
    return Response(content=resp.content, status_code=resp.status_code,
                    media_type=resp.headers.get("Content-Type", "application/json"))


@app.post("/api/chat")
async def hud_chat(request: Request) -> JSONResponse:
    """Typed chat from the HUD — same selected brain session as voice."""
    body = await request.json()
    text = (body.get("input") or "").strip()
    selected_brain_cfg = CFG.get(BRAIN_PROVIDER) or {}
    conversation = body.get("conversation") or selected_brain_cfg.get("conversation", "jarvis-main")
    if not text:
        return JSONResponse({"error": "empty input"}, status_code=400)
    out: dict = {"text": "", "tools": [], "run_id": None}

    def run_sync() -> None:
        timeout = float(selected_brain_cfg.get("timeout", 240))

        def consume(sid: str) -> list[str]:
            parts: list[str] = []
            for kind, value in BRAIN.chat_stream_events(sid, text, timeout):
                if kind == "text":
                    parts.append(value)
                elif kind == "tool":
                    out["tools"].append(json.loads(value))
                elif kind == "run":
                    out["run_id"] = value
                elif kind == "final":
                    info = json.loads(value)
                    record_brain_usage(info)
                    if info.get("content"):
                        parts = [info["content"]]
            return parts

        try:
            parts = consume(BRAIN.get_session_id(conversation))
        except RuntimeError as exc:
            if "404" not in str(exc):
                raise
            # stale session id (e.g. profile switch / DB reset) -> recreate once
            out["tools"].clear()
            parts = consume(BRAIN.get_session_id(conversation, force_new=True))
        out["text"] = "".join(parts).strip()

    try:
        await asyncio.to_thread(run_sync)
        return JSONResponse(out)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)


@app.get("/api/brain/capabilities")
async def brain_capabilities() -> JSONResponse:
    """Founder-local capability snapshot for the selected brain adapter."""
    try:
        if BRAIN.provider == "hermes":
            health = await asyncio.to_thread(
                requests.get,
                f"{HERMES.base}/health",
                headers=HERMES.headers(),
                timeout=5,
            )
            health.raise_for_status()
        caps = await asyncio.to_thread(BRAIN.discover_capabilities)
        return JSONResponse({
            "provider": BRAIN.provider,
            "online": True,
            "capabilities": {
                "streaming": caps.streaming,
                "sessions": caps.sessions,
                "stop": caps.stop,
                "approvals": caps.approvals,
                "memory": caps.memory,
                "extra": sorted(caps.extra),
            },
        })
    except Exception as exc:
        return JSONResponse(
            {"provider": BRAIN.provider, "online": False, "error": str(exc)},
            status_code=503,
        )


_ELEVEN_CACHE: dict = {"ts": 0.0, "data": None}


@app.get("/api/usage")
async def usage() -> JSONResponse:
    """Provider-neutral local usage plus optional ElevenLabs quota."""
    u = read_usage()
    cost_cfg = CFG.get("usage") or {}
    cin = float(cost_cfg.get("llm_cost_per_mtok_input", 0) or 0)
    cout = float(cost_cfg.get("llm_cost_per_mtok_output", 0) or 0)

    def est(b: dict) -> float | None:
        if not (cin or cout):
            return None
        return round(b.get("llm_in", 0) / 1e6 * cin + b.get("llm_out", 0) / 1e6 * cout, 4)

    configured_voice = str((CFG.get("voice") or {}).get("provider", "elevenlabs"))
    out = {
        "llm": {
            "today": u["today"], "total": u["total"],
            "today_cost": est(u["today"]), "total_cost": est(u["total"]),
        },
        "voice_provider": configured_voice,
        "voice_usage": {
            "provider": configured_voice,
            "used": u["today"].get("tts_chars", 0),
            "unit": "characters",
        },
        "elevenlabs": None,
    }
    # ElevenLabs subscription — NEVER blocks the response: serve the cache and
    # refresh it in the background when stale.
    now = time.time()
    if (_ELEVEN_CACHE["data"] is None or now - _ELEVEN_CACHE["ts"] > 300) and not _ELEVEN_CACHE.get("refreshing"):
        key = os.environ.get("ELEVENLABS_API_KEY") or os.environ.get("ELEVEN_API_KEY") or os.environ.get("XI_API_KEY")
        if key:
            _ELEVEN_CACHE["refreshing"] = True

            def fetch() -> dict | None:
                try:
                    r = requests.get("https://api.elevenlabs.io/v1/user/subscription",
                                     headers={"xi-api-key": key}, timeout=10)
                    if r.ok:
                        j = r.json()
                        return {
                            "used": j.get("character_count"),
                            "limit": j.get("character_limit"),
                            "remaining": (j.get("character_limit") or 0) - (j.get("character_count") or 0),
                            "tier": j.get("tier"),
                            "resets_unix": j.get("next_character_count_reset_unix"),
                        }
                except Exception:
                    pass
                return None

            async def refresh() -> None:
                try:
                    data = await asyncio.to_thread(fetch)
                    if data is not None:
                        _ELEVEN_CACHE.update(ts=time.time(), data=data)
                finally:
                    _ELEVEN_CACHE["refreshing"] = False

            asyncio.get_running_loop().create_task(refresh())
    out["elevenlabs"] = _ELEVEN_CACHE["data"]
    if configured_voice == "elevenlabs" and _ELEVEN_CACHE["data"]:
        out["voice_usage"] = {
            "provider": "elevenlabs",
            **_ELEVEN_CACHE["data"],
            "unit": "characters",
        }
    return JSONResponse(out)


WS_CLIENTS: set = set()


@app.post("/api/summon")
async def summon(request: Request) -> JSONResponse:
    """Broadcast a holographic media panel to all connected HUD clients.

    Body: {"media": "video"|"iframe"|"image", "src": "...", "title": "...",
           "position": "center"|"left"|"right"}  or  {"action": "dismiss"}
    Hermes can call this (curl with X-Jarvis-Token) to display media on the HUD.
    """
    body = await request.json()
    if body.get("action") == "dismiss":
        payload = {"type": "dismiss_panels"}
    else:
        payload = {"type": "summon_panel",
                   "media": body.get("media") or body.get("type") or "iframe",
                   "src": body.get("src", ""),
                   "title": body.get("title", "INCOMING FEED"),
                   "position": body.get("position", "center")}
    sent = 0
    for client in list(WS_CLIENTS):
        try:
            await client.send_json(payload)
            sent += 1
        except Exception:
            WS_CLIENTS.discard(client)
    return JSONResponse({"sent_to": sent})


_WORKER_CACHE: dict = {"ts": 0.0, "data": [], "refreshing": False}


@app.get("/api/machines")
async def machines() -> JSONResponse:
    """Local (Mac) stats + configured remote workers.

    Worker polls can take seconds when a worker is offline, so they run in a
    background refresh; the endpoint always answers instantly from cache.
    """
    result: list[dict] = []
    mac: dict = {"name": "MAC MINI · HERMES", "online": True}
    if psutil:
        mac.update({
            "cpu": psutil.cpu_percent(interval=0.1),
            "mem": psutil.virtual_memory().percent,
            "disk": psutil.disk_usage(str(ROOT)).percent,
        })
    result.append(mac)

    def poll_worker(w: dict) -> dict:
        info = {"name": w.get("name", w.get("host", "worker")), "online": False}
        url = w.get("stats_url")
        if url:
            try:
                r = requests.get(url, timeout=2)
                if r.ok:
                    info.update(r.json())
                    info["online"] = True
                    return info
            except Exception:
                pass
        import socket
        try:
            with socket.create_connection((w.get("host"), int(w.get("ping_port", 445))), timeout=1.5):
                info["online"] = True
                info["note"] = "online (no stats agent)"
        except Exception:
            pass
        return info

    workers = CFG.get("machines") or []
    now = time.time()
    if workers and now - _WORKER_CACHE["ts"] > 10 and not _WORKER_CACHE["refreshing"]:
        _WORKER_CACHE["refreshing"] = True

        async def refresh() -> None:
            try:
                data = [await asyncio.to_thread(poll_worker, w) for w in workers]
                _WORKER_CACHE.update(ts=time.time(), data=data)
            finally:
                _WORKER_CACHE["refreshing"] = False

        asyncio.get_running_loop().create_task(refresh())
    result.extend(_WORKER_CACHE["data"] or
                  [{"name": w.get("name", "worker"), "online": False, "note": "checking..."} for w in workers])
    return JSONResponse({"machines": result})


@app.get("/")
async def root() -> RedirectResponse:
    return RedirectResponse("/hud/")


if HUD_DIR.exists():
    app.mount("/hud", StaticFiles(directory=str(HUD_DIR), html=True), name="hud")


# ----------------------------------------------- Hermes dashboard TLS proxy
# The HUD (https) cannot iframe the plain-http dashboard (mixed content), so
# this second app reverse-proxies the entire dashboard over TLS, stripping
# frame-blocking headers. Served on its own port (see server.dashboard_proxy).

dash_app = FastAPI(title="Hermes Dashboard TLS Proxy")
_STRIP_HEADERS = {"x-frame-options", "content-security-policy", "content-length",
                  "transfer-encoding", "connection", "content-encoding"}


@dash_app.middleware("http")
async def dash_auth_middleware(request: Request, call_next):
    if not _request_authed(request):
        return Response(status_code=401, content="jarvis auth required")
    return await call_next(request)


def _dash_target() -> str:
    return ((CFG.get("server") or {}).get("dashboard_proxy") or {}).get(
        "target", "http://127.0.0.1:9119").rstrip("/")


@dash_app.websocket("/{path:path}")
async def dash_ws_proxy(ws: WebSocket, path: str) -> None:
    import websockets as wslib
    token = hud_token()
    if token and ws.cookies.get("jarvis_token") != token:
        await ws.close(code=4401)
        return
    await ws.accept()
    target = _dash_target().replace("http://", "ws://").replace("https://", "wss://")
    uri = f"{target}/{path}" + (f"?{ws.url.query}" if ws.url.query else "")
    try:
        async with wslib.connect(uri, max_size=None) as backend:
            async def client_to_backend() -> None:
                while True:
                    m = await ws.receive()
                    if m.get("text") is not None:
                        await backend.send(m["text"])
                    elif m.get("bytes") is not None:
                        await backend.send(m["bytes"])
                    elif m.get("type") == "websocket.disconnect":
                        break

            async def backend_to_client() -> None:
                async for m in backend:
                    if isinstance(m, str):
                        await ws.send_text(m)
                    else:
                        await ws.send_bytes(m)

            done, pending_t = await asyncio.wait(
                [asyncio.create_task(client_to_backend()),
                 asyncio.create_task(backend_to_client())],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending_t:
                t.cancel()
    except Exception:
        pass


@dash_app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def dash_http_proxy(path: str, request: Request) -> Response:
    body = await request.body()
    fwd_headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in ("host", "accept-encoding", "connection")}

    def do_request() -> requests.Response:
        return requests.request(
            request.method, f"{_dash_target()}/{path}",
            params=dict(request.query_params), headers=fwd_headers,
            data=body if body else None, timeout=60, allow_redirects=False,
        )

    resp = await asyncio.to_thread(do_request)
    out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in _STRIP_HEADERS}
    return Response(content=resp.content, status_code=resp.status_code, headers=out_headers)


# ------------------------------------------------------------------ WebSocket


@dataclass
class ConnState:
    audio_chunks: list = field(default_factory=list)
    recording: bool = False
    timing: TurnTiming | None = None
    turn_task: asyncio.Task | None = None
    current_run_id: str | None = None
    conversation: str = "jarvis-main"
    spoken_sentences: list = field(default_factory=list)
    interrupt_note: str | None = None
    partial_task: asyncio.Task | None = None
    last_partial_bytes: int = 0
    cancellation: CancellationToken | None = None
    pending_segments: dict[str, str] = field(default_factory=dict)
    last_played_sentence: str | None = None
    segment_counter: int = 0
    pending_approval: dict | None = None
    approval_timeout_task: asyncio.Task | None = None
    review_id: str | None = None
    review_future: asyncio.Future | None = None


async def _run_turn(ws: WebSocket, pipeline: VoicePipelineServer, conn: ConnState) -> None:
    timing = conn.timing
    assert timing is not None
    cancellation = conn.cancellation or CancellationToken()
    audio = b"".join(conn.audio_chunks)
    conn.audio_chunks = []
    try:
        transcript = await pipeline.transcribe(audio, timing)
        timing.transcript = transcript
        transcript_event: dict = {"type": "transcript", "text": transcript}
        review_required = False
        if timing.stt_confidence is not None:
            transcript_event["confidence"] = timing.stt_confidence
            review_required = timing.stt_confidence < float(
                (pipeline.cfg.get("stt") or {}).get("review_below_confidence", 0.72)
            )
            transcript_event["review_required"] = review_required
        if review_required:
            review_id = f"{timing.turn_id}:review"
            transcript_event["review_id"] = review_id
            conn.review_id = review_id
            conn.review_future = asyncio.get_running_loop().create_future()
        await ws.send_json(transcript_event)
        if review_required and conn.review_future is not None:
            try:
                reviewed = await asyncio.wait_for(
                    asyncio.shield(conn.review_future),
                    timeout=float(
                        (pipeline.cfg.get("stt") or {}).get(
                            "review_timeout_seconds", 20,
                        )
                    ),
                )
                if isinstance(reviewed, str) and reviewed.strip():
                    transcript = reviewed.strip()
                    timing.transcript = transcript
                    await ws.send_json({
                        "type": "transcript",
                        "text": transcript,
                        "review_applied": True,
                    })
            except asyncio.TimeoutError:
                await ws.send_json({
                    "type": "status",
                    "message": "Transcript review timed out; using the original.",
                })
            finally:
                conn.review_future = None
                conn.review_id = None
        if not transcript:
            await ws.send_json({"type": "error", "message": "No transcript detected."})
        else:
            if conn.interrupt_note:
                transcript_sent = (
                    f"[note: your previous spoken reply was cut off by the user after you said: "
                    f"\"{conn.interrupt_note}\"]\n{transcript}"
                )
                conn.interrupt_note = None
            else:
                transcript_sent = transcript
            conn.spoken_sentences = []
            await pipeline.stream_response_audio(ws, transcript_sent, timing, conn, cancellation)
            timing.total_done_monotonic = time.perf_counter()
            await ws.send_json({"type": "done", "turn_id": timing.turn_id, "timing": timing.summary()})
    except asyncio.CancelledError:
        cancellation.cancel()
        timing.interrupted = True
        timing.errors.append("turn cancelled (barge-in or stop)")
        raise
    except Exception as exc:
        timing.errors.append(f"{type(exc).__name__}: {exc}")
        try:
            await ws.send_json({"type": "agent_status", "state": "error"})
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        timing.total_done_monotonic = timing.total_done_monotonic or time.perf_counter()
        pipeline.log_turn(timing)
        conn.timing = None
        conn.current_run_id = None
        if conn.cancellation is cancellation:
            conn.cancellation = None


async def _cancel_active_turn(ws: WebSocket, pipeline: VoicePipelineServer, conn: ConnState,
                              stop_remote: bool = True) -> None:
    run_id = conn.current_run_id  # capture BEFORE cancel: turn cleanup clears it
    turn_was_active = conn.turn_task is not None and not conn.turn_task.done()
    if turn_was_active:
        if conn.cancellation is not None:
            conn.cancellation.cancel()
        if conn.last_played_sentence:
            conn.interrupt_note = conn.last_played_sentence
        conn.turn_task.cancel()
        try:
            await conn.turn_task
        except (asyncio.CancelledError, Exception):
            pass
    if conn.approval_timeout_task and not conn.approval_timeout_task.done():
        conn.approval_timeout_task.cancel()
    conn.pending_approval = None
    if conn.review_future and not conn.review_future.done():
        conn.review_future.cancel()
    conn.review_future = None
    conn.review_id = None
    if stop_remote and run_id and turn_was_active:
        conn.current_run_id = None
        try:
            assert pipeline.brain is not None
            res = await asyncio.to_thread(pipeline.brain.stop_run, run_id)
            # 404 = session runs not in the runs registry on this Hermes build;
            # dropping the SSE stream (above) still cuts the turn off.
            msg = "Run halted." if res["status_code"] in (200, 202, 404) else f"Stop returned {res['status_code']}."
            await ws.send_json({"type": "status", "message": msg})
        except Exception as exc:
            await ws.send_json({"type": "status", "message": f"Stop failed: {exc}"})
    if turn_was_active:
        await ws.send_json({"type": "agent_status", "state": "interrupted"})


def _maybe_schedule_partial(ws: WebSocket, pipeline: VoicePipelineServer, conn: ConnState) -> None:
    stt_cfg = CFG.get("stt") or {}
    if not stt_cfg.get("partials", True) or not conn.recording:
        return
    if conn.partial_task and not conn.partial_task.done():
        return
    buf = b"".join(conn.audio_chunks)
    min_new = int(16000 * 2 * float(stt_cfg.get("partial_interval", 1.2)))
    if len(buf) < 16000 or len(buf) - conn.last_partial_bytes < min_new or len(buf) > 16000 * 2 * 30:
        return
    conn.last_partial_bytes = len(buf)

    async def run() -> None:
        try:
            text = await pipeline.transcribe(buf)
            if text and conn.recording:
                await ws.send_json({"type": "partial_transcript", "text": text})
        except Exception:
            pass

    conn.partial_task = asyncio.create_task(run())


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    if not _ws_allowed(ws):
        await ws.close(code=4401)
        return
    await ws.accept()
    WS_CLIENTS.add(ws)
    pipeline = get_pipeline()
    conn = ConnState(
        conversation=(CFG.get(BRAIN_PROVIDER) or {}).get("conversation", "jarvis-main")
    )
    voice_cfg = CFG.get("voice") or {}
    brain_cfg = CFG.get(BRAIN_PROVIDER) or {}
    stt_cfg = CFG.get("stt") or {}
    await ws.send_json({
        "type": "status",
        "message": f"{BRAIN_PROVIDER.title()} voice server connected.",
        "models": {
            "brain": brain_cfg.get("model") or f"{BRAIN_PROVIDER}-agent",
            "stt": stt_cfg.get("model", "unknown"),
            "tts_provider": voice_cfg.get("provider", "elevenlabs"),
            "tts": voice_cfg.get("model", "unknown"),
            "language": stt_cfg.get("language", "en"),
            "fallback": brain_cfg.get("fallback_provider") or "disabled",
        },
    })
    try:
        while True:
            message = await ws.receive()
            if "text" in message and message["text"] is not None:
                event = json.loads(message["text"])
                etype = event.get("type")
                if etype == "start":
                    await _cancel_active_turn(ws, pipeline, conn)  # barge-in
                    if event.get("conversation"):
                        conn.conversation = str(event["conversation"])
                    conn.audio_chunks = []
                    conn.last_partial_bytes = 0
                    conn.pending_segments.clear()
                    conn.last_played_sentence = None
                    conn.segment_counter = 0
                    conn.recording = True
                    conn.timing = TurnTiming(turn_id=pipeline.next_turn_id())
                    conn.cancellation = CancellationToken()
                    conn.timing.audio_start_monotonic = time.perf_counter()
                    conn.timing.stt_model = CFG["stt"]["model"]
                    conn.timing.stt_language = str(CFG["stt"].get("language", "en"))
                    await ws.send_json({"type": "status", "message": f"Turn {conn.timing.turn_id} recording started."})
                elif etype == "stop":
                    if conn.timing is None:
                        await ws.send_json({"type": "error", "message": "Received stop before start."})
                        continue
                    conn.recording = False
                    conn.timing.end_of_speech_monotonic = time.perf_counter()
                    await ws.send_json({"type": "agent_status", "state": "transcribing"})
                    conn.turn_task = asyncio.create_task(_run_turn(ws, pipeline, conn))
                elif etype == "stop_run":
                    await _cancel_active_turn(ws, pipeline, conn)
                    await ws.send_json({"type": "agent_status", "state": "stopped"})
                elif etype == "approval_decision":
                    run_id = event.get("run_id") or conn.current_run_id
                    if not run_id:
                        await ws.send_json({"type": "error", "message": "No run for approval."})
                        continue
                    decision = event.get("decision", "deny")
                    body = {
                        "decision": decision,
                        "approved": decision == "allow",
                        "approval_id": event.get("approval_id"),
                    }
                    assert pipeline.brain is not None
                    res = await asyncio.to_thread(pipeline.brain.post_approval, run_id, body)
                    if conn.approval_timeout_task and not conn.approval_timeout_task.done():
                        conn.approval_timeout_task.cancel()
                    conn.pending_approval = None
                    await ws.send_json({"type": "status", "message": f"Approval sent ({res['status_code']})."})
                elif etype == "speech_played":
                    segment_id = str(event.get("segment_id") or "")
                    played = conn.pending_segments.pop(segment_id, None)
                    if played:
                        conn.last_played_sentence = played
                elif etype == "transcript_review":
                    if (
                        conn.review_future is not None
                        and not conn.review_future.done()
                        and event.get("review_id") == conn.review_id
                    ):
                        conn.review_future.set_result(str(event.get("text") or ""))
                else:
                    await ws.send_json({"type": "error", "message": f"Unknown event type: {etype}"})
            elif "bytes" in message and message["bytes"] is not None:
                if conn.recording:
                    conn.audio_chunks.append(message["bytes"])
                    _maybe_schedule_partial(ws, pipeline, conn)
    except WebSocketDisconnect:
        if conn.turn_task and not conn.turn_task.done():
            conn.turn_task.cancel()
        if conn.approval_timeout_task and not conn.approval_timeout_task.done():
            conn.approval_timeout_task.cancel()
        if conn.review_future and not conn.review_future.done():
            conn.review_future.cancel()
        print("Client disconnected", flush=True)
    finally:
        WS_CLIENTS.discard(ws)


def main() -> int:
    server = CFG["server"]
    host = server.get("host", "0.0.0.0")
    port = int(server.get("port", 8765))
    tls_ports = server.get("tls_ports") or ([server["tls_port"]] if server.get("tls_port") else [])
    cert = server.get("tls_cert")
    key = server.get("tls_key")
    print(f"Starting Hermes voice server on ws://{host}:{port}/ws", flush=True)
    if tls_ports and cert and key and (ROOT / cert).exists() and (ROOT / key).exists():
        servers = [uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="info"))]
        for tp in tls_ports:
            print(f"HUD available on https://{host}:{tp}/hud/", flush=True)
            servers.append(uvicorn.Server(uvicorn.Config(
                app, host=host, port=int(tp), log_level="info",
                ssl_certfile=str(ROOT / cert), ssl_keyfile=str(ROOT / key),
            )))

        dp = server.get("dashboard_proxy") or {}
        if dp.get("port"):
            print(f"Dashboard proxy on https://{host}:{dp['port']}/", flush=True)
            servers.append(uvicorn.Server(uvicorn.Config(
                dash_app, host=host, port=int(dp["port"]), log_level="warning",
                ssl_certfile=str(ROOT / cert), ssl_keyfile=str(ROOT / key),
            )))

        async def serve_all() -> None:
            await asyncio.gather(*[s.serve() for s in servers])

        asyncio.run(serve_all())
    else:
        uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
