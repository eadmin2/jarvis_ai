from html.parser import HTMLParser
from pathlib import Path


HUD = Path(__file__).parents[1] / "server" / "hud" / "index.html"


class _HudParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.labels: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.append(values["id"] or "")
        if tag == "label" and values.get("for"):
            self.labels.add(values["for"] or "")


def test_hud_voice_contract_and_accessible_controls() -> None:
    source = HUD.read_text(encoding="utf-8")
    parser = _HudParser()
    parser.feed(source)

    assert len(parser.ids) == len(set(parser.ids)), "HUD element IDs must be unique"
    for control in {
        "chatInput",
        "micSelect",
        "speakerSelect",
        "reviewText",
        "autoEndpoint",
        "minSpeechMs",
        "silenceMs",
    }:
        assert control in parser.ids
        assert control in parser.labels

    for state in {
        "idle",
        "listening",
        "transcribing",
        "thinking",
        "tool_use",
        "speaking",
        "interrupted",
        "stopped",
        "error",
    }:
        assert f"{state}:" in source

    for required in {
        'role="status"',
        'aria-live="polite"',
        'role="meter"',
        "prefers-reduced-motion",
        "enumerateDevices",
        "setSinkId",
        "review_required",
        "voice_usage",
        "jarvis.autoEndpoint",
        "RMS_SPEECH_THRESHOLD",
        "observeEndpoint(rms",
        'e.type==="speech_segment"',
        'e.type==="speech_segment_end"',
        'type:"speech_played"',
        "src.onended",
        "cancelSpeechAcknowledgements();",
        'if(value===null||value==="")return fallback',
    }:
        assert required in source

    # A segment is acknowledged only after its end marker, attached audio
    # sources have naturally ended, and no stop/barge-in cancellation occurred.
    ack_guard = (
        "segment.acked||segment.cancelled||!segment.ended||"
        "!segment.hadAudio||segment.sources.size"
    )
    assert ack_guard in source
    stop_body = source[source.index("function stopPlayback()") :]
    assert stop_body.index("cancelSpeechAcknowledgements()") < stop_body.index(
        "activeSources.forEach"
    )
