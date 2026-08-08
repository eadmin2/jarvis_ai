from __future__ import annotations

from collections.abc import Iterator

import pytest

from server.voice import (
    CancellationToken,
    FailoverTTSProvider,
    PronunciationDictionary,
    TTSConfigurationError,
    TTSProvider,
    create_tts_provider,
    normalize_pronunciations,
    resolve_voice_preset,
)


class ScriptedProvider(TTSProvider):
    def __init__(self, name: str, script: list[bytes | BaseException | str]) -> None:
        super().__init__(model=f"{name}-model", voice_id=f"{name}-voice")
        self.name = name
        self.script = script
        self.calls = 0

    def stream(
        self, text: str, cancellation: CancellationToken | None = None,
    ) -> Iterator[bytes]:
        self.calls += 1
        token = cancellation or CancellationToken()
        for item in self.script:
            if item == "cancel":
                token.cancel()
                return
            if isinstance(item, BaseException):
                raise item
            yield item


def test_failover_before_audio_reports_the_provider_that_served_request() -> None:
    primary = ScriptedProvider("openai", [RuntimeError("unavailable")])
    fallback = ScriptedProvider("elevenlabs", [b"fallback-pcm"])
    metrics = []
    provider = FailoverTTSProvider(primary, fallback, metrics_hook=metrics.append)

    assert b"".join(provider.stream("hello")) == b"fallback-pcm"
    assert provider.served_provider == "elevenlabs"
    assert primary.calls == fallback.calls == 1
    assert metrics[0].provider == "elevenlabs"
    assert metrics[0].model == "elevenlabs-model"
    assert metrics[0].attempted_providers == ("openai", "elevenlabs")
    assert metrics[0].error is None


def test_failover_is_forbidden_after_any_audio_was_emitted() -> None:
    primary = ScriptedProvider("openai", [b"partial", RuntimeError("stream broke")])
    fallback = ScriptedProvider("elevenlabs", [b"must-not-play"])
    metrics = []
    provider = FailoverTTSProvider(primary, fallback, metrics_hook=metrics.append)
    stream = provider.stream("hello")

    assert next(stream) == b"partial"
    with pytest.raises(RuntimeError, match="stream broke"):
        list(stream)
    assert fallback.calls == 0
    assert metrics[0].provider == "openai"
    assert metrics[0].attempted_providers == ("openai",)


def test_empty_primary_is_treated_as_pre_audio_failure() -> None:
    primary = ScriptedProvider("openai", [])
    fallback = ScriptedProvider("elevenlabs", [b"audio"])
    provider = FailoverTTSProvider(primary, fallback)

    assert b"".join(provider.stream("hello")) == b"audio"
    assert provider.served_provider == "elevenlabs"


def test_cancellation_never_starts_fallback() -> None:
    primary = ScriptedProvider("openai", ["cancel"])
    fallback = ScriptedProvider("elevenlabs", [b"audio"])
    metrics = []
    provider = FailoverTTSProvider(primary, fallback, metrics_hook=metrics.append)
    token = CancellationToken()

    assert list(provider.stream("hello", token)) == []
    assert token.cancelled
    assert fallback.calls == 0
    assert metrics[0].cancelled
    assert metrics[0].attempted_providers == ("openai",)


def test_factory_builds_primary_and_optional_fallback_mapping() -> None:
    provider = create_tts_provider(
        {
            "provider": "openai",
            "model": "gpt-4o-mini-tts",
            "voice_id": "alloy",
            "sample_rate": 16_000,
            "fallback": {
                "provider": "elevenlabs",
                "model": "eleven_turbo_v2_5",
                "voice_id": "founder-voice",
            },
        },
        environ={
            "OPENAI_API_KEY": "openai-key",
            "ELEVENLABS_API_KEY": "eleven-key",
        },
    )

    assert isinstance(provider, FailoverTTSProvider)
    assert provider.primary.name == "openai"
    assert provider.fallback.name == "elevenlabs"
    assert provider.output_sample_rate == 16_000


def test_factory_rejects_incompatible_fallback_sample_rate() -> None:
    with pytest.raises(TTSConfigurationError, match="sample rates must match"):
        create_tts_provider(
            {
                "provider": "openai",
                "model": "m1",
                "voice_id": "v1",
                "sample_rate": 16_000,
                "fallback": {
                    "provider": "elevenlabs",
                    "model": "m2",
                    "voice_id": "v2",
                    "sample_rate": 24_000,
                },
            },
            environ={
                "OPENAI_API_KEY": "openai-key",
                "ELEVENLABS_API_KEY": "eleven-key",
            },
        )


def test_factory_rejects_nested_failover_chains() -> None:
    with pytest.raises(TTSConfigurationError, match="cannot contain"):
        create_tts_provider(
            {
                "provider": "openai",
                "model": "m1",
                "voice_id": "v1",
                "fallback": {
                    "provider": "elevenlabs",
                    "model": "m2",
                    "voice_id": "v2",
                    "fallback": {
                        "provider": "openai",
                        "model": "m3",
                        "voice_id": "v3",
                    },
                },
            },
            environ={
                "OPENAI_API_KEY": "openai-key",
                "ELEVENLABS_API_KEY": "eleven-key",
            },
        )


def test_pronunciation_dictionary_prefers_phrases_and_respects_boundaries() -> None:
    normalizer = PronunciationDictionary(
        {
            "FAOS": "Fay-oss",
            "FAOS Agent": "Fay-oss agent",
            "Nguyễn": "Ngwin",
        }
    )

    assert normalizer.normalize(
        "FAOS Agent greets Nguyễn; faos works, but XFAOSY stays."
    ) == "Fay-oss agent greets Ngwin; Fay-oss works, but XFAOSY stays."


def test_pronunciation_helper_validates_entries() -> None:
    assert normalize_pronunciations("API ready", {"API": "A P I"}) == "A P I ready"
    with pytest.raises(TTSConfigurationError, match="control characters"):
        normalize_pronunciations("API", {"API": "A\nP\nI"})


def test_named_preset_merges_only_safe_overrides_without_mutating_config() -> None:
    config = {
        "provider": "openai",
        "model": "base-model",
        "voice_id": "base-voice",
        "api_key_env": "OPENAI_API_KEY",
        "preset": "concise",
        "presets": {
            "concise": {
                "model": "voice-model",
                "voice_id": "onyx",
                "instructions": "Warm, concise, and calm.",
                "speed": 1.1,
            }
        },
    }

    resolved = resolve_voice_preset(config)

    assert resolved["model"] == "voice-model"
    assert resolved["voice_id"] == "onyx"
    assert resolved["instructions"] == "Warm, concise, and calm."
    assert resolved["speed"] == 1.1
    assert resolved["provider"] == "openai"
    assert resolved["api_key_env"] == "OPENAI_API_KEY"
    assert config["model"] == "base-model"
    assert "presets" not in resolved


def test_preset_accepts_voice_alias_used_by_provider_apis() -> None:
    resolved = resolve_voice_preset(
        {
            "voice_id": "base",
            "presets": {"deep": {"voice": "onyx"}},
        },
        "deep",
    )

    assert resolved["voice_id"] == "onyx"
    assert "voice" not in resolved


def test_preset_cannot_override_provider_or_transport_security() -> None:
    with pytest.raises(TTSConfigurationError, match="Unsafe.*provider"):
        resolve_voice_preset(
            {
                "preset": "unsafe",
                "presets": {
                    "unsafe": {
                        "provider": "elevenlabs",
                        "api_key_env": "ATTACKER_KEY",
                    }
                },
            }
        )
