"""Provider-neutral streaming text-to-speech support for JARVIS."""

from .base import (
    CancellationToken,
    TTSAuthenticationError,
    TTSConfigurationError,
    TTSEmptyAudioError,
    TTSError,
    TTSHTTPError,
    TTSMetrics,
    TTSProvider,
)
from .factory import create_tts_provider
from .failover import FailoverTTSProvider
from .text import PronunciationDictionary, normalize_pronunciations, resolve_voice_preset

__all__ = [
    "CancellationToken",
    "TTSAuthenticationError",
    "TTSConfigurationError",
    "TTSEmptyAudioError",
    "TTSError",
    "TTSHTTPError",
    "TTSMetrics",
    "TTSProvider",
    "FailoverTTSProvider",
    "PronunciationDictionary",
    "create_tts_provider",
    "normalize_pronunciations",
    "resolve_voice_preset",
]
