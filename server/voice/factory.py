"""Validated construction of configured TTS providers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from .base import MetricsHook, TTSConfigurationError, TTSProvider
from .elevenlabs import ElevenLabsTTSProvider
from .failover import FailoverTTSProvider
from .openai import OpenAITTSProvider
from .text import resolve_voice_preset


def _required_text(config: Mapping[str, object], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TTSConfigurationError(f"voice.{key} must be a non-empty string")
    return value.strip()


def _positive_number(config: Mapping[str, object], key: str, default: float) -> float:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise TTSConfigurationError(f"voice.{key} must be a positive number")
    return float(value)


def create_tts_provider(
    config: Mapping[str, object],
    *,
    environ: Mapping[str, str] | None = None,
    session: Any = None,
    metrics_hook: MetricsHook | None = None,
) -> TTSProvider:
    """Build a provider, optionally wrapped in conservative pre-audio failover.

    ``fallback`` is another provider mapping. Shared transport-neutral values
    (sample rate and timeout) inherit from the primary when omitted.
    """
    config = resolve_voice_preset(config)
    raw_fallback = config.get("fallback")
    if raw_fallback is not None:
        if not isinstance(raw_fallback, Mapping):
            raise TTSConfigurationError("voice.fallback must be a mapping")
        if "fallback" in raw_fallback:
            raise TTSConfigurationError("voice.fallback cannot contain another fallback")
        primary_config = dict(config)
        primary_config.pop("fallback", None)
        fallback_config = dict(raw_fallback)
        for shared_key in ("sample_rate", "timeout"):
            if shared_key in primary_config:
                fallback_config.setdefault(shared_key, primary_config[shared_key])
        primary = create_tts_provider(
            primary_config, environ=environ, session=session, metrics_hook=None
        )
        fallback = create_tts_provider(
            fallback_config, environ=environ, session=session, metrics_hook=None
        )
        try:
            return FailoverTTSProvider(primary, fallback, metrics_hook=metrics_hook)
        except ValueError as exc:
            raise TTSConfigurationError(str(exc)) from exc

    env = os.environ if environ is None else environ
    provider = str(config.get("provider", "elevenlabs")).strip().lower()
    model = _required_text(config, "model")
    voice_id = _required_text(config, "voice_id")
    output_rate = int(_positive_number(config, "sample_rate", 16_000))
    timeout = _positive_number(config, "timeout", 120)

    if provider == "openai":
        key_name = str(config.get("api_key_env", "OPENAI_API_KEY"))
        api_key = env.get(key_name, "").strip()
        if not api_key:
            raise TTSConfigurationError(f"voice API key environment variable {key_name} is not set")
        kwargs: dict[str, object] = {
            "api_key": api_key,
            "model": model,
            "voice_id": voice_id,
            "output_sample_rate": output_rate,
            "timeout": timeout,
            "metrics_hook": metrics_hook,
        }
        if session is not None:
            kwargs["session"] = session
        instructions = config.get("instructions")
        if instructions is not None:
            if not isinstance(instructions, str):
                raise TTSConfigurationError("voice.instructions must be a string")
            kwargs["instructions"] = instructions
        speed = config.get("speed")
        if speed is not None:
            if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not 0.25 <= speed <= 4:
                raise TTSConfigurationError("voice.speed must be between 0.25 and 4")
            kwargs["speed"] = float(speed)
        return OpenAITTSProvider(**kwargs)

    if provider == "elevenlabs":
        default_key_names = ("ELEVENLABS_API_KEY", "ELEVEN_API_KEY", "XI_API_KEY")
        configured_key = config.get("api_key_env")
        key_names = (str(configured_key),) if configured_key else default_key_names
        api_key = next((env.get(name, "").strip() for name in key_names if env.get(name)), "")
        if not api_key:
            names = ", ".join(key_names)
            raise TTSConfigurationError(f"voice API key environment variable is not set ({names})")
        kwargs = {
            "api_key": api_key,
            "model": model,
            "voice_id": voice_id,
            "output_sample_rate": output_rate,
            "provider_output_format": str(config.get("output_format", "pcm_16000")),
            "timeout": timeout,
            "metrics_hook": metrics_hook,
        }
        if session is not None:
            kwargs["session"] = session
        voice_settings = config.get("voice_settings")
        if voice_settings is not None:
            if not isinstance(voice_settings, dict):
                raise TTSConfigurationError("voice.voice_settings must be a mapping")
            kwargs["voice_settings"] = voice_settings
        return ElevenLabsTTSProvider(**kwargs)

    raise TTSConfigurationError(
        f"Unsupported voice.provider {provider!r}; expected 'openai' or 'elevenlabs'"
    )
