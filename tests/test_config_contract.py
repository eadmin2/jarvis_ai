from pathlib import Path

import yaml

from server.brain import GbrainAdapter, HermesAdapter
from server.voice import create_tts_provider
from server.voice.openai import OpenAITTSProvider


ROOT = Path(__file__).resolve().parents[1]


def load_example() -> dict:
    return yaml.safe_load(
        (ROOT / "server" / "config" / "server.example.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_example_selects_supported_local_brain_and_voice() -> None:
    cfg = load_example()
    assert cfg["brain"]["provider"] in {"hermes", "gbrain"}
    provider = create_tts_provider(
        cfg["voice"], environ={"OPENAI_API_KEY": "local-test-key"}
    )
    assert isinstance(provider, OpenAITTSProvider)
    assert provider.output_sample_rate == 16_000


def test_example_constructs_both_brain_adapters_without_network() -> None:
    cfg = load_example()
    assert HermesAdapter(cfg).provider == "hermes"
    assert GbrainAdapter(cfg).provider == "gbrain"
    assert "{session_id}" in cfg["gbrain"]["endpoints"]["stream"]


def test_example_contains_no_credentials() -> None:
    text = (ROOT / "server" / "config" / "server.example.yaml").read_text(
        encoding="utf-8"
    )
    assert "sk-" not in text
    assert "Bearer " not in text
