from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pytest

from server.voice import (
    CancellationToken,
    TTSAuthenticationError,
    TTSConfigurationError,
    TTSHTTPError,
    create_tts_provider,
)
from server.voice.elevenlabs import ElevenLabsTTSProvider
from server.voice.openai import OpenAITTSProvider


def sine_pcm(rate: int, seconds: float, frequency: float = 440.0) -> bytes:
    positions = np.arange(round(rate * seconds), dtype=np.float64)
    signal = np.sin(2 * math.pi * frequency * positions / rate) * 18_000
    return signal.astype("<i2").tobytes()


class FakeResponse:
    def __init__(
        self, chunks: list[bytes] | None = None, status_code: int = 200, text: str = "",
    ) -> None:
        self.chunks = chunks or []
        self.status_code = status_code
        self.text = text
        self.closed = False
        self.close_calls = 0

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        assert chunk_size == 4096
        for chunk in self.chunks:
            if self.closed:
                raise OSError("response closed")
            yield chunk

    def close(self) -> None:
        self.closed = True
        self.close_calls += 1


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, dict]] = []

    def post(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append((url, kwargs))
        return self.response


def test_openai_streams_resampled_pcm_and_reports_neutral_metrics() -> None:
    source = sine_pcm(24_000, 0.1)
    response = FakeResponse([source[:1111], source[1111:]])
    session = FakeSession(response)
    metrics = []
    provider = OpenAITTSProvider(
        api_key="test-key",
        model="gpt-4o-mini-tts",
        voice_id="alloy",
        session=session,
        metrics_hook=metrics.append,
    )

    output = b"".join(provider.stream("Hello founder"))

    assert len(output) == 1_600 * 2
    assert response.closed
    url, call = session.calls[0]
    assert url.endswith("/v1/audio/speech")
    assert call["json"]["response_format"] == "pcm"
    assert call["headers"]["Authorization"] == "Bearer test-key"
    assert metrics[0].provider == "openai"
    assert metrics[0].characters == len("Hello founder")
    assert metrics[0].audio_bytes == len(output)
    assert metrics[0].first_audio_seconds is not None
    assert not metrics[0].cancelled


def test_elevenlabs_uses_existing_request_shape() -> None:
    source = sine_pcm(16_000, 0.05)
    response = FakeResponse([source])
    session = FakeSession(response)
    provider = ElevenLabsTTSProvider(
        api_key="eleven-test-key",
        model="eleven_turbo_v2_5",
        voice_id="voice-1",
        session=session,
    )

    assert b"".join(provider.stream("Ready")) == source
    url, call = session.calls[0]
    assert url.endswith("/text-to-speech/voice-1/stream")
    assert call["params"] == {"output_format": "pcm_16000"}
    assert call["json"]["model_id"] == "eleven_turbo_v2_5"
    assert call["headers"]["xi-api-key"] == "eleven-test-key"


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize("provider_name", ["openai", "elevenlabs"])
def test_authentication_errors_have_a_typed_exception(
    status: int, provider_name: str,
) -> None:
    session = FakeSession(FakeResponse(status_code=status, text="credential rejected"))
    provider = create_tts_provider(
        {"provider": provider_name, "model": "model", "voice_id": "voice"},
        environ={
            "OPENAI_API_KEY": "openai-key",
            "ELEVENLABS_API_KEY": "eleven-key",
        },
        session=session,
    )

    with pytest.raises(TTSAuthenticationError):
        list(provider.stream("hello"))
    assert session.response.closed


@pytest.mark.parametrize("provider_name", ["openai", "elevenlabs"])
def test_http_errors_include_status_and_bounded_detail(provider_name: str) -> None:
    session = FakeSession(FakeResponse(status_code=429, text="rate limited"))
    provider = create_tts_provider(
        {"provider": provider_name, "model": "model", "voice_id": "voice"},
        environ={
            "OPENAI_API_KEY": "openai-key",
            "ELEVENLABS_API_KEY": "eleven-key",
        },
        session=session,
    )

    with pytest.raises(TTSHTTPError) as raised:
        list(provider.stream("hello"))
    assert raised.value.status_code == 429
    assert "rate limited" in str(raised.value)


@pytest.mark.parametrize("provider_name,rate", [("openai", 24_000), ("elevenlabs", 16_000)])
def test_cancellation_closes_the_live_response_and_stops_streaming(
    provider_name: str, rate: int,
) -> None:
    source = sine_pcm(rate, 0.1)
    response = FakeResponse([source[: len(source) // 2], source[len(source) // 2 :]])
    session = FakeSession(response)
    metrics = []
    provider = create_tts_provider(
        {"provider": provider_name, "model": "model", "voice_id": "voice"},
        environ={
            "OPENAI_API_KEY": "openai-key",
            "ELEVENLABS_API_KEY": "eleven-key",
        },
        session=session,
        metrics_hook=metrics.append,
    )
    token = CancellationToken()
    stream = provider.stream("interrupt me", token)

    first = next(stream)
    assert first
    assert token.cancel()
    assert response.closed
    assert list(stream) == []
    assert metrics[0].cancelled
    assert metrics[0].audio_bytes == len(first)


def test_pre_cancelled_request_never_opens_an_http_connection() -> None:
    session = FakeSession(FakeResponse([sine_pcm(24_000, 0.01)]))
    metrics = []
    provider = OpenAITTSProvider(
        api_key="test-key",
        model="model",
        voice_id="voice",
        session=session,
        metrics_hook=metrics.append,
    )
    token = CancellationToken()
    token.cancel()

    assert list(provider.stream("too late", token)) == []
    assert session.calls == []
    assert metrics[0].cancelled
    assert metrics[0].audio_bytes == 0


def test_factory_validates_provider_credentials_and_pcm_contract() -> None:
    with pytest.raises(TTSConfigurationError, match="Unsupported"):
        create_tts_provider(
            {"provider": "unknown", "model": "m", "voice_id": "v"}, environ={}
        )
    with pytest.raises(TTSConfigurationError, match="OPENAI_API_KEY"):
        create_tts_provider(
            {"provider": "openai", "model": "m", "voice_id": "v"}, environ={}
        )
    with pytest.raises(TTSConfigurationError, match="raw PCM"):
        create_tts_provider(
            {
                "provider": "elevenlabs",
                "model": "m",
                "voice_id": "v",
                "output_format": "mp3_44100_128",
            },
            environ={"ELEVENLABS_API_KEY": "key"},
        )
