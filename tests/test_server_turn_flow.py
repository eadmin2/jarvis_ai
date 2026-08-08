from __future__ import annotations

import asyncio
import importlib.util
import os
from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = ROOT / "server"
os.environ.setdefault(
    "JARVIS_CONFIG_PATH", str(SERVER_DIR / "config" / "server.example.yaml")
)
sys.path.insert(0, str(SERVER_DIR))

# The turn orchestration tests use fakes and must not load a Whisper model or
# construct a cloud SDK client during module import.
anthropic_stub = types.ModuleType("anthropic")
anthropic_stub.Anthropic = object
sys.modules.setdefault("anthropic", anthropic_stub)
stt_stub = types.ModuleType("RealtimeSTT")
stt_stub.AudioToTextRecorder = object
sys.modules.setdefault("RealtimeSTT", stt_stub)

SPEC = importlib.util.spec_from_file_location("jarvis_server_test", SERVER_DIR / "server.py")
assert SPEC and SPEC.loader
SERVER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SERVER
SPEC.loader.exec_module(SERVER)


class FakeWebSocket:
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def send_json(self, event: dict) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_low_confidence_transcript_waits_for_founder_correction() -> None:
    class Pipeline:
        cfg = {"stt": {"review_below_confidence": 0.72, "review_timeout_seconds": 2}}

        def __init__(self) -> None:
            self.received = None

        async def transcribe(self, _audio, timing):
            timing.stt_confidence = 0.41
            return "wrong transcript"

        async def stream_response_audio(self, _ws, transcript, _timing, _conn, _cancel):
            self.received = transcript

        def log_turn(self, _timing):
            pass

    pipeline = Pipeline()
    ws = FakeWebSocket()
    conn = SERVER.ConnState(
        audio_chunks=[b"\0\0"],
        timing=SERVER.TurnTiming(turn_id=7),
        cancellation=SERVER.CancellationToken(),
    )
    task = asyncio.create_task(SERVER._run_turn(ws, pipeline, conn))
    for _ in range(20):
        if conn.review_future is not None:
            break
        await asyncio.sleep(0)
    assert conn.review_future is not None
    review_event = next(event for event in ws.events if event["type"] == "transcript")
    assert review_event["review_required"] is True
    assert review_event["review_id"] == "7:review"
    conn.review_future.set_result("corrected transcript")
    await task

    assert pipeline.received == "corrected transcript"
    assert any(event.get("review_applied") for event in ws.events)


@pytest.mark.asyncio
async def test_speech_segment_frames_audio_for_playback_acknowledgement() -> None:
    pipeline = SERVER.VoicePipelineServer.__new__(SERVER.VoicePipelineServer)

    async def no_audio(_ws, _sentence, _timing, _cancellation):
        return None

    pipeline._send_tts_sentence = no_audio
    ws = FakeWebSocket()
    conn = SERVER.ConnState()
    timing = SERVER.TurnTiming(turn_id=3)
    token = SERVER.CancellationToken()

    await pipeline._speak_segment(ws, "Systems online.", timing, conn, token)

    assert [event["type"] for event in ws.events] == [
        "speech_segment",
        "speech_segment_end",
    ]
    assert ws.events[0]["segment_id"] == "3:1"
    assert conn.pending_segments == {"3:1": "Systems online."}


@pytest.mark.asyncio
async def test_interruption_note_uses_acknowledged_audio_not_queued_text() -> None:
    class Brain:
        def stop_run(self, _run_id):
            return {"status_code": 202}

    pipeline = types.SimpleNamespace(brain=Brain())
    ws = FakeWebSocket()

    async def active_turn():
        await asyncio.Future()

    conn = SERVER.ConnState(
        turn_task=asyncio.create_task(active_turn()),
        current_run_id=None,
        spoken_sentences=["queued but unheard"],
        last_played_sentence="actually heard",
        cancellation=SERVER.CancellationToken(),
    )

    await SERVER._cancel_active_turn(ws, pipeline, conn, stop_remote=False)

    assert conn.interrupt_note == "actually heard"
    assert conn.cancellation.cancelled is True


def test_clause_chunking_and_pronunciation_are_low_latency_and_configurable() -> None:
    segments, remainder = SERVER.VoicePipelineServer._extract_complete_sentences(
        "This is a deliberately long clause that should start speaking now, while generation continues."
    )
    assert segments == [
        "This is a deliberately long clause that should start speaking now,",
        "while generation continues.",
    ]
    assert remainder == ""

    pipeline = SERVER.VoicePipelineServer.__new__(SERVER.VoicePipelineServer)
    pipeline.pronunciations = SERVER.PronunciationDictionary(
        {"FAOS": "F A O S", "Gbrain": "G brain"}
    )
    assert pipeline._clean_for_tts("FAOS uses Gbrain.") == "F A O S uses G brain."


def test_gbrain_retrieval_enriches_hermes_without_replacing_the_agent() -> None:
    class Brain:
        provider = "hermes"

        def __init__(self):
            self.received = ""

        def get_session_id(self, _conversation):
            return "session-1"

        def chat_stream_events(self, _session_id, text, _timeout, _cancellation):
            self.received = text
            yield "run", "run-1"
            yield "text", "answer"

    class Context:
        enabled = True

        def query(self, text, _cancellation):
            assert text == "what changed?"
            return "compiled founder context"

        @staticmethod
        def augment(text, context):
            return SERVER.GbrainContextAdapter.augment(text, context)

    pipeline = SERVER.VoicePipelineServer.__new__(SERVER.VoicePipelineServer)
    pipeline.cfg = {
        "brain": {"provider": "hermes"},
        "llm": {"provider": "hermes"},
        "hermes": {"timeout": 1, "fallback_provider": ""},
    }
    pipeline.brain = Brain()
    pipeline.gbrain_context = Context()
    timing = SERVER.TurnTiming(turn_id=9)

    events = list(
        pipeline.stream_llm_events_sync(
            "what changed?", timing, "jarvis-main", SERVER.CancellationToken()
        )
    )

    assert events == [("run", "run-1"), ("text", "answer")]
    assert pipeline.brain.provider == "hermes"
    assert "compiled founder context" in pipeline.brain.received
    assert pipeline.brain.received.endswith("Founder request:\nwhat changed?")
    assert timing.gbrain_context_used is True
