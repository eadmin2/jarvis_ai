"""OpenAI streaming PCM text-to-speech provider."""

from __future__ import annotations

from typing import Any, Iterator

import requests

from .base import (
    CancellationToken,
    MetricsHook,
    TTSAuthenticationError,
    TTSHTTPError,
    TTSProvider,
)
from .resampler import StreamingPCMResampler


class OpenAITTSProvider(TTSProvider):
    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        voice_id: str,
        output_sample_rate: int = 16_000,
        timeout: float = 120,
        endpoint: str = "https://api.openai.com/v1/audio/speech",
        instructions: str | None = None,
        speed: float | None = None,
        session: Any = requests,
        metrics_hook: MetricsHook | None = None,
    ) -> None:
        super().__init__(model, voice_id, output_sample_rate, metrics_hook)
        self.api_key = api_key
        self.timeout = timeout
        self.endpoint = endpoint
        self.instructions = instructions
        self.speed = speed
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
            payload: dict[str, object] = {
                "model": self.model,
                "voice": self.voice_id,
                "input": text,
                "response_format": "pcm",
            }
            if self.instructions:
                payload["instructions"] = self.instructions
            if self.speed is not None:
                payload["speed"] = self.speed
            response = self.session.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/octet-stream",
                    "Content-Type": "application/json",
                },
                json=payload,
                stream=True,
                timeout=self.timeout,
            )
            if response.status_code in (401, 403):
                raise TTSAuthenticationError("OpenAI rejected the configured API key")
            if response.status_code >= 400:
                raise TTSHTTPError("OpenAI", response.status_code, response.text)

            unsubscribe = token.register(response.close)
            resampler = StreamingPCMResampler(24_000, self.output_sample_rate)
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
