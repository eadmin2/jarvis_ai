"""ElevenLabs streaming PCM text-to-speech provider."""

from __future__ import annotations

import re
from typing import Any, Iterator

import requests

from .base import (
    CancellationToken,
    MetricsHook,
    TTSAuthenticationError,
    TTSConfigurationError,
    TTSHTTPError,
    TTSProvider,
)
from .resampler import StreamingPCMResampler

PCM_FORMAT_RE = re.compile(r"^pcm_(\d+)$")


class ElevenLabsTTSProvider(TTSProvider):
    name = "elevenlabs"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        voice_id: str,
        output_sample_rate: int = 16_000,
        provider_output_format: str = "pcm_16000",
        timeout: float = 120,
        endpoint_base: str = "https://api.elevenlabs.io/v1",
        voice_settings: dict[str, object] | None = None,
        session: Any = requests,
        metrics_hook: MetricsHook | None = None,
    ) -> None:
        super().__init__(model, voice_id, output_sample_rate, metrics_hook)
        match = PCM_FORMAT_RE.fullmatch(provider_output_format)
        if not match:
            raise TTSConfigurationError(
                "ElevenLabs output_format must be raw PCM such as pcm_16000"
            )
        self.source_sample_rate = int(match.group(1))
        self.provider_output_format = provider_output_format
        self.api_key = api_key
        self.timeout = timeout
        self.endpoint_base = endpoint_base.rstrip("/")
        self.voice_settings = voice_settings or {
            "stability": 0.55,
            "similarity_boost": 0.70,
            "style": 0.10,
            "use_speaker_boost": True,
        }
        self.session = session

    def stream(
        self, text: str, cancellation: CancellationToken | None = None,
    ) -> Iterator[bytes]:
        token = cancellation or CancellationToken()
        started_at = self._clock()
        first_audio_at = None
        audio_bytes = 0
        error: BaseException | None = None
        response = None
        def unsubscribe() -> None:
            pass
        try:
            if token.cancelled:
                return
            response = self.session.post(
                f"{self.endpoint_base}/text-to-speech/{self.voice_id}/stream",
                params={"output_format": self.provider_output_format},
                headers={
                    "xi-api-key": self.api_key,
                    "Accept": "application/octet-stream",
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": self.model,
                    "voice_settings": self.voice_settings,
                },
                stream=True,
                timeout=self.timeout,
            )
            if response.status_code in (401, 403):
                raise TTSAuthenticationError(
                    "ElevenLabs rejected the configured API key"
                )
            if response.status_code >= 400:
                raise TTSHTTPError("ElevenLabs", response.status_code, response.text)

            unsubscribe = token.register(response.close)
            resampler = StreamingPCMResampler(
                self.source_sample_rate, self.output_sample_rate
            )
            try:
                for source_chunk in response.iter_content(chunk_size=4096):
                    if token.cancelled:
                        break
                    if not source_chunk:
                        continue
                    chunk = resampler.process(source_chunk)
                    if not chunk:
                        continue
                    if first_audio_at is None:
                        first_audio_at = self._clock()
                    audio_bytes += len(chunk)
                    yield chunk
                if not token.cancelled:
                    tail = resampler.flush()
                    if tail:
                        if first_audio_at is None:
                            first_audio_at = self._clock()
                        audio_bytes += len(tail)
                        yield tail
            except Exception:
                if not token.cancelled:
                    raise
        except BaseException as exc:
            error = exc
            raise
        finally:
            unsubscribe()
            if response is not None:
                response.close()
            self._emit_metrics(
                text=text,
                started_at=started_at,
                first_audio_at=first_audio_at,
                audio_bytes=audio_bytes,
                cancelled=token.cancelled,
                error=error,
            )
