from __future__ import annotations

import math

import numpy as np
import pytest

from server.voice.resampler import StreamingPCMResampler, resample_pcm16, scipy_signal


def sine_pcm(rate: int, seconds: float, frequency: float = 440.0) -> bytes:
    positions = np.arange(round(rate * seconds), dtype=np.float64)
    signal = np.sin(2 * math.pi * frequency * positions / rate) * 20_000
    return signal.astype("<i2").tobytes()


def dominant_frequency(pcm: bytes, rate: int) -> float:
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    return float(np.fft.rfftfreq(len(samples), 1 / rate)[np.argmax(spectrum)])


def test_resamples_24k_to_16k_with_expected_duration_and_frequency() -> None:
    output = resample_pcm16(sine_pcm(24_000, 1.0), 24_000, 16_000)

    assert len(output) == 16_000 * 2
    assert dominant_frequency(output, 16_000) == 440.0


def test_streamed_chunks_are_identical_to_one_shot_across_odd_boundaries() -> None:
    source = sine_pcm(24_000, 0.25, frequency=997.0)
    expected = resample_pcm16(source, 24_000, 16_000)
    resampler = StreamingPCMResampler(24_000, 16_000)
    sizes = (1, 7, 1023, 2, 4097, 13, 509)
    chunks: list[bytes] = []
    offset = 0
    index = 0
    while offset < len(source):
        size = sizes[index % len(sizes)]
        chunks.append(resampler.process(source[offset : offset + size]))
        offset += size
        index += 1
    chunks.append(resampler.flush())

    actual = b"".join(chunks)
    assert actual == expected

    # A smooth input remains smooth at every emitted provider-chunk boundary.
    samples = np.frombuffer(actual, dtype="<i2").astype(np.int32)
    assert np.max(np.abs(np.diff(samples))) < 9_000


def test_supported_scipy_path_filters_audio_above_target_nyquist() -> None:
    if scipy_signal is None:
        pytest.skip("SciPy is installed by server requirements, not the lean test runtime")
    source = sine_pcm(24_000, 0.25, frequency=10_000)
    output = np.frombuffer(resample_pcm16(source, 24_000, 16_000), dtype="<i2")
    assert float(np.sqrt(np.mean(output.astype(np.float64) ** 2))) < 1_000


def test_upsampling_has_exact_rounded_duration() -> None:
    source = sine_pcm(16_000, 0.1)
    output = resample_pcm16(source, 16_000, 24_000)

    assert len(output) == 2_400 * 2
    assert dominant_frequency(output, 24_000) == 440.0


def test_flush_is_idempotent_and_process_after_flush_is_rejected() -> None:
    resampler = StreamingPCMResampler(24_000, 16_000)
    resampler.process(b"\x00\x00\x01")
    resampler.flush()

    assert resampler.flush() == b""
    try:
        resampler.process(b"\x00\x00")
    except RuntimeError as exc:
        assert "after flush" in str(exc)
    else:
        raise AssertionError("process after flush should fail")
