from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("jarvis_client", ROOT / "client" / "client.py")
assert SPEC and SPEC.loader
CLIENT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CLIENT
SPEC.loader.exec_module(CLIENT)


def pcm(level: int, seconds: float = 0.08) -> bytes:
    return np.full(int(CLIENT.SAMPLE_RATE * seconds), level, dtype=np.int16).tobytes()


def test_endpoint_requires_speech_before_silence() -> None:
    detector = CLIENT.SpeechEndpointDetector(silence_seconds=0.16, min_record_seconds=0.1)
    assert not detector.observe(pcm(0), 0.08)
    assert not detector.observe(pcm(0), 0.16)


def test_endpoint_fires_after_speech_and_configured_silence() -> None:
    detector = CLIENT.SpeechEndpointDetector(
        rms_threshold=400,
        silence_seconds=0.16,
        min_record_seconds=0.1,
    )
    assert not detector.observe(pcm(1200), 0.08)
    assert not detector.observe(pcm(0), 0.16)
    assert detector.observe(pcm(0), 0.24)


def test_noise_below_threshold_does_not_arm_endpoint() -> None:
    detector = CLIENT.SpeechEndpointDetector(rms_threshold=500, silence_seconds=0.08)
    assert not detector.observe(pcm(300), 1.0)
    assert not detector.speech_started
