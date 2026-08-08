"""Conservative pre-audio failover for streaming TTS."""

from __future__ import annotations

from typing import Iterator

from .base import (
    CancellationToken,
    MetricsHook,
    TTSEmptyAudioError,
    TTSProvider,
)


class FailoverTTSProvider(TTSProvider):
    """Try a fallback only when the primary emitted no audio.

    Once any bytes have left this iterator, changing providers would splice two
    voices into one utterance. Errors after that point are therefore surfaced
    to the caller. The same cancellation token is passed to both attempts and
    cancellation is never interpreted as a reason to fail over.
    """

    name = "failover"

    def __init__(
        self,
        primary: TTSProvider,
        fallback: TTSProvider,
        metrics_hook: MetricsHook | None = None,
    ) -> None:
        if primary.output_sample_rate != fallback.output_sample_rate:
            raise ValueError("primary and fallback TTS sample rates must match")
        super().__init__(
            primary.model,
            primary.voice_id,
            primary.output_sample_rate,
            metrics_hook,
        )
        self.primary = primary
        self.fallback = fallback
        self.served_provider: str | None = None
        self.attempted_providers: tuple[str, ...] = ()

    def stream(
        self, text: str, cancellation: CancellationToken | None = None,
    ) -> Iterator[bytes]:
        token = cancellation or CancellationToken()
        started_at = self._clock()
        first_audio_at = None
        audio_bytes = 0
        error: BaseException | None = None
        attempted: list[str] = []
        selected = self.primary
        self.served_provider = None

        try:
            for index, provider in enumerate((self.primary, self.fallback)):
                selected = provider
                attempted.append(provider.name)
                self.attempted_providers = tuple(attempted)
                emitted = False
                try:
                    for chunk in provider.stream(text, token):
                        if token.cancelled:
                            return
                        if not chunk:
                            continue
                        emitted = True
                        if first_audio_at is None:
                            first_audio_at = self._clock()
                            self.served_provider = provider.name
                        audio_bytes += len(chunk)
                        yield chunk
                    if token.cancelled:
                        return
                    if not emitted:
                        raise TTSEmptyAudioError(
                            f"{provider.name} returned no audio"
                        )
                    return
                except Exception:
                    if token.cancelled or emitted or index == 1:
                        raise
                    # No bytes escaped, so switching voices is still safe.
                    continue
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._emit_metrics(
                text=text,
                started_at=started_at,
                first_audio_at=first_audio_at,
                audio_bytes=audio_bytes,
                cancelled=token.cancelled,
                error=error,
                provider=self.served_provider or selected.name,
                model=selected.model,
                voice_id=selected.voice_id,
                attempted_providers=tuple(attempted),
            )
