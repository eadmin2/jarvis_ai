"""Exercise live dispatch and interruption without loading a local STT model."""
import asyncio
import importlib
import os
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from test_sixtydb import PCM, Response, VOICE


def load_server():
    # Only the hardware/model import is stubbed; exercise the real voice server.
    stt = types.ModuleType("RealtimeSTT")
    stt.AudioToTextRecorder = None
    exists, read_text = Path.exists, Path.read_text
    example = Path("config/server.example.yaml").read_text()
    def config_text(path, *args, **kwargs):
        return example if path.name == "server.yaml" else read_text(path, *args, **kwargs)
    previous = sys.modules.get("RealtimeSTT")
    sys.modules["RealtimeSTT"] = stt
    try:
        with patch.object(Path, "read_text", config_text), patch.object(
            Path, "exists", lambda path: False if path.name == ".env" else exists(path)
        ):
            return importlib.import_module("server")
    finally:
        if previous is None:
            sys.modules.pop("RealtimeSTT", None)
        else:
            sys.modules["RealtimeSTT"] = previous


server = load_server()


class DispatchTest(unittest.TestCase):
    def pipeline(self, voice):
        pipeline = server.VoicePipelineServer.__new__(server.VoicePipelineServer)
        pipeline.cfg = {"voice": voice}
        return pipeline

    def test_sixtydb_dispatch_and_timing(self):
        pipeline = self.pipeline({**VOICE, "provider": "sixtydb"})
        timing = server.TurnTiming(1)
        with patch("server.sixtydb_tts_chunks", return_value=(chunk for chunk in [PCM])) as synth, patch("server.requests.post") as legacy, patch("server.record_usage"):
            self.assertEqual(list(pipeline.tts_chunks_sync("Hi", timing)), [PCM])
        legacy.assert_not_called()
        synth.assert_called_once_with("Hi", pipeline.cfg["voice"], None)
        self.assertEqual(timing.voice_id, "workspace-voice")
        self.assertIsNotNone(timing.first_tts_audio_byte_monotonic)

    def test_elevenlabs_default_stays_selected(self):
        response = Response(PCM)
        response.close = lambda: setattr(response, "closed", True)
        pipeline = self.pipeline({"voice_id": "legacy-voice", "model": "eleven_flash_v2_5"})
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "legacy-key"}), patch("server.requests.post", return_value=response) as post, patch("server.record_usage"):
            self.assertEqual(b"".join(pipeline.tts_chunks_sync("Hi", server.TurnTiming(1))), PCM)
        self.assertIn("api.elevenlabs.io", post.call_args.args[0])
        self.assertTrue(response.closed)

    def test_unknown_provider_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Unsupported TTS"):
            list(self.pipeline({"provider": "invalid"}).tts_chunks_sync("Hi", server.TurnTiming(1)))


class InterruptionTest(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_consumer_signals_worker_and_drops_late_audio(self):
        pipeline = server.VoicePipelineServer.__new__(server.VoicePipelineServer)
        release, closed = threading.Event(), threading.Event()
        received = asyncio.Event()
        captured = []
        def stream(text, timing, stopped):
            captured.append(stopped)
            try:
                yield PCM
                release.wait(2)
                yield PCM
            finally:
                closed.set()
        pipeline.tts_chunks_sync = stream
        ws = types.SimpleNamespace(send_bytes=AsyncMock(side_effect=lambda chunk: received.set()))
        task = asyncio.create_task(pipeline._send_tts_sentence(ws, "Hi", server.TurnTiming(1)))
        try:
            await asyncio.wait_for(received.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(captured[0].is_set())
            release.set()
            self.assertTrue(await asyncio.to_thread(closed.wait, 2))
            await asyncio.sleep(0)
            self.assertEqual(ws.send_bytes.await_count, 1)
        finally:
            release.set()


if __name__ == "__main__":
    unittest.main()
