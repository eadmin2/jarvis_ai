"""Safe, deterministic voice preset and pronunciation helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping

from .base import TTSConfigurationError

PRESET_OVERRIDE_KEYS = frozenset(
    {"model", "voice", "voice_id", "instructions", "speed"}
)


class PronunciationDictionary:
    """Replace configured words or phrases without matching inside words."""

    def __init__(
        self, pronunciations: Mapping[str, str] | None, *, case_sensitive: bool = False,
    ) -> None:
        entries = dict(pronunciations or {})
        for source, spoken in entries.items():
            if not isinstance(source, str) or not source.strip():
                raise TTSConfigurationError("pronunciation keys must be non-empty strings")
            if not isinstance(spoken, str) or not spoken.strip():
                raise TTSConfigurationError(
                    "pronunciation replacements must be non-empty strings"
                )
            if any(
                character in source or character in spoken
                for character in ("\x00", "\r", "\n")
            ):
                raise TTSConfigurationError(
                    "pronunciation replacements cannot contain control characters"
                )
        self._entries = entries
        if not entries:
            self._pattern = None
            return
        alternatives = sorted(entries, key=len, reverse=True)
        flags = 0 if case_sensitive else re.IGNORECASE
        self._pattern = re.compile(
            r"(?<!\w)(?:" + "|".join(re.escape(key) for key in alternatives) + r")(?!\w)",
            flags,
        )
        self._case_sensitive = case_sensitive
        self._folded = {key.casefold(): value for key, value in entries.items()}

    def normalize(self, text: str) -> str:
        if self._pattern is None or not text:
            return text

        def replace(match: re.Match[str]) -> str:
            key = match.group(0)
            if self._case_sensitive:
                return self._entries[key]
            return self._folded[key.casefold()]

        return self._pattern.sub(replace, text)


def normalize_pronunciations(
    text: str,
    pronunciations: Mapping[str, str] | None,
    *,
    case_sensitive: bool = False,
) -> str:
    """One-shot pronunciation normalization helper."""
    return PronunciationDictionary(
        pronunciations, case_sensitive=case_sensitive
    ).normalize(text)


def resolve_voice_preset(
    config: Mapping[str, object], preset_name: str | None = None,
) -> dict[str, object]:
    """Return a copy with a named preset's allow-listed overrides applied.

    Presets cannot change providers, credentials, endpoints, timeouts, output
    formats, or arbitrary transport settings. Unknown keys fail closed rather
    than being silently passed to a cloud API.
    """
    resolved = dict(config)
    raw_presets = resolved.pop("presets", None)
    selected = preset_name if preset_name is not None else resolved.pop("preset", None)
    if selected is None:
        return resolved
    if not isinstance(selected, str) or not selected.strip():
        raise TTSConfigurationError("voice.preset must be a non-empty string")
    if not isinstance(raw_presets, Mapping):
        raise TTSConfigurationError("voice.presets must be a mapping")
    raw_preset = raw_presets.get(selected)
    if not isinstance(raw_preset, Mapping):
        raise TTSConfigurationError(f"Unknown voice preset {selected!r}")
    unknown = set(raw_preset) - PRESET_OVERRIDE_KEYS
    if unknown:
        names = ", ".join(sorted(str(key) for key in unknown))
        raise TTSConfigurationError(f"Unsafe voice preset override(s): {names}")
    overrides = dict(raw_preset)
    if "voice" in overrides:
        if "voice_id" in overrides:
            raise TTSConfigurationError(
                "voice preset cannot define both voice and voice_id"
            )
        overrides["voice_id"] = overrides.pop("voice")
    resolved.update(overrides)
    return resolved
