"""Common contracts and lifecycle primitives for streaming TTS providers."""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterator


class TTSError(RuntimeError):
    """Base error raised by a TTS provider."""


class TTSConfigurationError(TTSError):
    """The configured provider cannot be constructed safely."""


class TTSAuthenticationError(TTSError):
    """A provider rejected the configured credential."""


class TTSHTTPError(TTSError):
    """A provider returned a non-authentication HTTP error."""

    def __init__(self, provider: str, status_code: int, detail: str = "") -> None:
        self.provider = provider
        self.status_code = status_code
        suffix = f": {detail[:500]}" if detail else ""
        super().__init__(f"{provider} HTTP {status_code}{suffix}")


class TTSEmptyAudioError(TTSError):
    """A provider completed successfully without returning playable audio."""


class CancellationToken:
    """Thread-safe cooperative cancellation with resource-close callbacks."""

    def __init__(self) -> None:
        self._cancelled = threading.Event()
        self._lock = threading.Lock()
        self._callbacks: dict[int, Callable[[], None]] = {}
        self._next_callback_id = 0

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def cancel(self) -> bool:
        """Cancel once and close registered resources outside the token lock."""
        with self._lock:
            if self._cancelled.is_set():
                return False
            self._cancelled.set()
            callbacks = list(self._callbacks.values())
            self._callbacks.clear()
        for callback in callbacks:
            try:
                callback()
            except Exception:
                # Cancellation is best-effort and must remain safe to call from
                # a WebSocket task while an HTTP worker owns the response.
                pass
        return True

    def register(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Register a close callback and return an idempotent unsubscribe."""
        with self._lock:
            if self._cancelled.is_set():
                run_now = True
                callback_id = -1
            else:
                run_now = False
                callback_id = self._next_callback_id
                self._next_callback_id += 1
                self._callbacks[callback_id] = callback
        if run_now:
            callback()

        def unsubscribe() -> None:
            with self._lock:
                self._callbacks.pop(callback_id, None)

        return unsubscribe


@dataclass(frozen=True)
class TTSMetrics:
    provider: str
    model: str
    voice_id: str
    characters: int
    started_at: float
    first_audio_at: float | None
    completed_at: float
    audio_bytes: int
    output_sample_rate: int
    cancelled: bool
    error: str | None
    attempted_providers: tuple[str, ...] = ()

    @property
    def first_audio_seconds(self) -> float | None:
        if self.first_audio_at is None:
            return None
        return self.first_audio_at - self.started_at

    @property
    def elapsed_seconds(self) -> float:
        return self.completed_at - self.started_at


MetricsHook = Callable[[TTSMetrics], None]


class TTSProvider(ABC):
    """Synchronous iterator contract used by the existing JARVIS worker."""

    name: str

    def __init__(
        self,
        model: str,
        voice_id: str,
        output_sample_rate: int = 16_000,
        metrics_hook: MetricsHook | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.model = model
        self.voice_id = voice_id
        self.output_sample_rate = output_sample_rate
        self.metrics_hook = metrics_hook
        self._clock = clock

    @abstractmethod
    def stream(
        self, text: str, cancellation: CancellationToken | None = None,
    ) -> Iterator[bytes]:
        """Yield signed 16-bit mono PCM chunks at ``output_sample_rate``."""

    def _emit_metrics(
        self,
        *,
        text: str,
        started_at: float,
        first_audio_at: float | None,
        audio_bytes: int,
        cancelled: bool,
        error: BaseException | None,
        provider: str | None = None,
        model: str | None = None,
        voice_id: str | None = None,
        attempted_providers: tuple[str, ...] | None = None,
    ) -> None:
        if self.metrics_hook is None:
            return
        metrics = TTSMetrics(
            provider=provider or self.name,
            model=model or self.model,
            voice_id=voice_id or self.voice_id,
            characters=len(text),
            started_at=started_at,
            first_audio_at=first_audio_at,
            completed_at=self._clock(),
            audio_bytes=audio_bytes,
            output_sample_rate=self.output_sample_rate,
            cancelled=cancelled,
            error=None if error is None else type(error).__name__,
            attempted_providers=attempted_providers or (provider or self.name,),
        )
        try:
            self.metrics_hook(metrics)
        except Exception:
            # Observability must not break speech playback.
            pass
