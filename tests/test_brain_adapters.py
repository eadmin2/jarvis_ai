from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from server.brain import (
    BrainAdapterError,
    BrainEventType,
    GbrainAdapter,
    HermesAdapter,
)
from server.voice import CancellationToken


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        payload: Any = None,
        text: str = "",
        lines: list[str] | None = None,
    ):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.lines = lines or []
        self.closed = False
        self.encoding = None

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_lines(self, decode_unicode=True):
        yield from self.lines

    def close(self):
        self.closed = True


class FakeTransport:
    def __init__(self, responses: list[FakeResponse]):
        self.responses = responses
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def _take(self, method: str, url: str, kwargs: dict[str, Any]):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)

    def get(self, url: str, **kwargs):
        return self._take("GET", url, kwargs)

    def post(self, url: str, **kwargs):
        return self._take("POST", url, kwargs)


def sse(event: str, payload: str) -> list[str]:
    return [f"event: {event}", f"data: {payload}", ""]


def test_hermes_recorded_stream_normalizes_all_contract_events(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HERMES_TEST_KEY", "secret")
    lines = [
        *sse("run.started", '{"run_id":"run-1"}'),
        *sse("assistant.delta", '{"delta":"Hello"}'),
        *sse("tool.started", '{"tool_name":"calendar","preview":"today"}'),
        *sse("approval.requested", '{"id":"approval-1","action":"send"}'),
        *sse("memory.updated", '{"key":"preference"}'),
        *sse("assistant.completed", '{"content":"Hello there","interrupted":false}'),
        *sse("run.completed", '{"usage":{"input_tokens":3,"output_tokens":2}}'),
    ]
    response = FakeResponse(lines=lines)
    adapter = HermesAdapter(
        {"api_key_env": "HERMES_TEST_KEY"},
        FakeTransport([response]),
        tmp_path / "sessions.json",
    )

    events = list(adapter.stream("session-1", "hello", 10))

    assert [event.type for event in events] == [
        BrainEventType.RUN_STARTED,
        BrainEventType.TEXT_DELTA,
        BrainEventType.TOOL_STARTED,
        BrainEventType.APPROVAL_REQUIRED,
        BrainEventType.MEMORY_UPDATED,
        BrainEventType.RUN_COMPLETED,
    ]
    assert all(event.run_id == "run-1" for event in events)
    assert events[1].data == {"delta": "Hello"}
    assert events[3].data["approval_id"] == "approval-1"
    assert events[-1].data["content"] == "Hello there"
    assert response.closed is True


def test_hermes_interruption_and_legacy_tuple_compatibility(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HERMES_TEST_KEY", "secret")
    response = FakeResponse(
        lines=[
            *sse("run.started", '{"run_id":"run-2"}'),
            *sse("assistant.completed", '{"content":"Partial","interrupted":true}'),
        ]
    )
    adapter = HermesAdapter(
        {"api_key_env": "HERMES_TEST_KEY"},
        FakeTransport([response]),
        tmp_path / "sessions.json",
    )

    tuples = list(adapter.chat_stream_events("session-2", "stop", 10))

    assert tuples[0] == ("run", "run-2")
    assert tuples[1][0] == "final"
    assert '"interrupted": true' in tuples[1][1]


def test_hermes_http_failure_becomes_error_event(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HERMES_TEST_KEY", "secret")
    response = FakeResponse(status_code=503, text="offline")
    adapter = HermesAdapter(
        {"api_key_env": "HERMES_TEST_KEY"},
        FakeTransport([response]),
        tmp_path / "sessions.json",
    )

    events = list(adapter.stream("session-1", "hello"))

    assert len(events) == 1
    assert events[0].type == BrainEventType.ERROR
    assert "503" in events[0].data["message"]
    assert response.closed is True


def test_hermes_named_session_persists_and_command_methods_match_existing_api(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HERMES_TEST_KEY", "secret")
    transport = FakeTransport(
        [
            FakeResponse(payload={"session": {"id": "session-3"}}),
            FakeResponse(status_code=202, text="stopping"),
            FakeResponse(status_code=200, text="allowed"),
        ]
    )
    adapter = HermesAdapter(
        {"api_key_env": "HERMES_TEST_KEY", "session_key": "jarvis:user:main"},
        transport,
        tmp_path / "sessions.json",
    )

    assert adapter.get_session_id("jarvis-main") == "session-3"
    assert adapter.get_session_id("jarvis-main") == "session-3"
    assert adapter.stop_run("run-3")["status_code"] == 202
    assert adapter.post_approval(
        "run-3", {"approval_id": "approval-3", "decision": "allow"}
    )["status_code"] == 200
    assert len(transport.calls) == 3
    assert transport.calls[0][2]["headers"]["X-Hermes-Session-Key"] == "jarvis:user:main"


def test_gbrain_uses_configurable_routes_event_map_and_capability_discovery(tmp_path: Path):
    capabilities = FakeResponse(
        payload={"capabilities": ["streaming", "sessions", "stop", "approvals", "memory", "vision"]}
    )
    stream = FakeResponse(
        lines=[
            *sse("job.opened", '{"runId":"g-1"}'),
            *sse("token", '{"text":"Xin chao"}'),
            *sse("action", '{"toolName":"search"}'),
            *sse("consent", '{"approvalId":"a-1"}'),
            *sse("memory.saved", '{"key":"language"}'),
            *sse("job.done", '{"content":"Xin chao"}'),
        ]
    )
    transport = FakeTransport([capabilities, stream])
    adapter = GbrainAdapter(
        {
            "base_url": "http://gbrain.local:9999",
            "endpoints": {
                "capabilities": "/features",
                "stream": "/threads/{session_id}/events",
            },
            "events": {
                "job.opened": "run_started",
                "token": "text_delta",
                "action": "tool_started",
                "consent": "approval_required",
                "memory.saved": "memory_updated",
                "job.done": "run_completed",
            },
        },
        transport,
        tmp_path / "gbrain.json",
    )

    caps = adapter.discover_capabilities()
    events = list(adapter.stream("thread-1", "hello"))

    assert caps.stop and caps.approvals and caps.memory
    assert caps.extra == frozenset({"vision"})
    assert [event.type for event in events] == [
        BrainEventType.RUN_STARTED,
        BrainEventType.TEXT_DELTA,
        BrainEventType.TOOL_STARTED,
        BrainEventType.APPROVAL_REQUIRED,
        BrainEventType.MEMORY_UPDATED,
        BrainEventType.RUN_COMPLETED,
    ]
    assert events[1].data["delta"] == "Xin chao"
    assert events[2].data["name"] == "search"
    assert events[3].data["approval_id"] == "a-1"
    assert transport.calls[0][1] == "http://gbrain.local:9999/features"
    assert transport.calls[1][1] == "http://gbrain.local:9999/threads/thread-1/events"


def test_gbrain_configurable_session_stop_and_approval_methods(tmp_path: Path):
    transport = FakeTransport(
        [
            FakeResponse(payload={"session": {"id": "thread-2"}}),
            FakeResponse(payload={"id": "thread-2", "title": "jarvis-main"}),
            FakeResponse(status_code=202, text="stopping"),
            FakeResponse(status_code=200, text="allowed"),
        ]
    )
    adapter = GbrainAdapter(
        {
            "base_url": "http://gbrain.local",
            "endpoints": {
                "create_session": "/threads",
                "get_session": "/threads/{session_id}",
                "stop": "/jobs/{run_id}/cancel",
                "approval": "/jobs/{run_id}/decisions",
            },
        },
        transport,
        tmp_path / "gbrain.json",
    )

    assert adapter.get_session_id("jarvis-main") == "thread-2"
    assert adapter.get_session("thread-2")["title"] == "jarvis-main"
    assert adapter.stop_run("job-2")["status_code"] == 202
    assert adapter.submit_approval("job-2", "approval-2", "allow")["status_code"] == 200
    assert [call[1] for call in transport.calls] == [
        "http://gbrain.local/threads",
        "http://gbrain.local/threads/thread-2",
        "http://gbrain.local/jobs/job-2/cancel",
        "http://gbrain.local/jobs/job-2/decisions",
    ]
    assert transport.calls[-1][2]["json"] == {
        "approval_id": "approval-2",
        "decision": "allow",
    }


def test_gbrain_legacy_tuple_stream_matches_existing_consumers(tmp_path: Path):
    response = FakeResponse(
        lines=[
            *sse("run.started", '{"run_id":"g-3"}'),
            *sse("assistant.delta", '{"delta":"Ready"}'),
            *sse("run.completed", '{"content":"Ready"}'),
        ]
    )
    adapter = GbrainAdapter(
        {}, FakeTransport([response]), tmp_path / "gbrain.json"
    )

    events = list(adapter.chat_stream_events("thread-3", "status", 10))

    assert events[0] == ("run", "g-3")
    assert events[1] == ("text", "Ready")
    assert events[2][0] == "final"
    assert '"content": "Ready"' in events[2][1]


def test_gbrain_invalid_mapping_and_invalid_sse_are_normalized_as_errors(tmp_path: Path):
    response = FakeResponse(
        lines=[
            *sse("custom", '{}'),
            *sse("token", "not-json"),
        ]
    )
    adapter = GbrainAdapter(
        {"events": {"custom": "not_an_event", "token": "text_delta"}},
        FakeTransport([response]),
        tmp_path / "gbrain.json",
    )

    events = list(adapter.stream("thread-1", "hello"))

    assert [event.type for event in events] == [BrainEventType.ERROR, BrainEventType.ERROR]
    assert "unknown normalized event type" in events[0].data["message"]
    assert events[1].data["message"] == "invalid SSE JSON"


def test_capability_discovery_rejects_invalid_payload(tmp_path: Path):
    adapter = GbrainAdapter(
        {}, FakeTransport([FakeResponse(payload={"capabilities": "yes"})]), tmp_path / "gbrain.json"
    )

    with pytest.raises(BrainAdapterError, match="capability discovery failed"):
        adapter.discover_capabilities()


@pytest.mark.parametrize("adapter_kind", ["hermes", "gbrain"])
def test_fifty_sequential_turns_preserve_completion_order_and_close_streams(
    adapter_kind: str, monkeypatch, tmp_path: Path
):
    if adapter_kind == "hermes":
        monkeypatch.setenv("HERMES_TEST_KEY", "secret")
        responses = [
            FakeResponse(
                lines=[
                    *sse("run.started", f'{{"run_id":"h-{turn}"}}'),
                    *sse("assistant.delta", f'{{"delta":"turn {turn}"}}'),
                    *sse(
                        "assistant.completed",
                        f'{{"content":"turn {turn}","interrupted":false}}',
                    ),
                    *sse("run.completed", '{"usage":{"output_tokens":2}}'),
                ]
            )
            for turn in range(50)
        ]
        adapter = HermesAdapter(
            {"api_key_env": "HERMES_TEST_KEY"},
            FakeTransport(responses.copy()),
            tmp_path / "hermes.json",
        )
        run_prefix = "h"
    else:
        responses = [
            FakeResponse(
                lines=[
                    *sse("run.started", f'{{"run_id":"g-{turn}"}}'),
                    *sse("assistant.delta", f'{{"delta":"turn {turn}"}}'),
                    *sse("run.completed", f'{{"content":"turn {turn}"}}'),
                ]
            )
            for turn in range(50)
        ]
        adapter = GbrainAdapter(
            {}, FakeTransport(responses.copy()), tmp_path / "gbrain.json"
        )
        run_prefix = "g"

    for turn in range(50):
        events = list(adapter.stream("stable-session", f"request {turn}", 10))
        assert [event.type for event in events] == [
            BrainEventType.RUN_STARTED,
            BrainEventType.TEXT_DELTA,
            BrainEventType.RUN_COMPLETED,
        ]
        assert events[0].run_id == f"{run_prefix}-{turn}"
        assert events[-1].run_id == f"{run_prefix}-{turn}"
        assert events[-1].data["content"] == f"turn {turn}"

    assert all(response.closed for response in responses)


@pytest.mark.parametrize("adapter_kind", ["hermes", "gbrain"])
def test_named_session_is_created_once_and_reused_fifty_times(
    adapter_kind: str, monkeypatch, tmp_path: Path
):
    if adapter_kind == "hermes":
        monkeypatch.setenv("HERMES_TEST_KEY", "secret")
        transport = FakeTransport(
            [FakeResponse(payload={"session": {"id": "stable-session"}})]
        )
        adapter = HermesAdapter(
            {"api_key_env": "HERMES_TEST_KEY"},
            transport,
            tmp_path / "hermes.json",
        )
    else:
        transport = FakeTransport(
            [FakeResponse(payload={"session": {"id": "stable-session"}})]
        )
        adapter = GbrainAdapter({}, transport, tmp_path / "gbrain.json")

    assert [adapter.get_session_id("jarvis-main") for _ in range(50)] == [
        "stable-session"
    ] * 50
    assert len(transport.calls) == 1


@pytest.mark.parametrize("adapter_kind", ["hermes", "gbrain"])
def test_fifty_repeated_cancellations_close_stream_and_command_responses(
    adapter_kind: str, monkeypatch, tmp_path: Path
):
    responses: list[FakeResponse] = []
    for turn in range(50):
        prefix = "h" if adapter_kind == "hermes" else "g"
        responses.extend(
            [
                FakeResponse(
                    lines=[
                        *sse("run.started", f'{{"run_id":"{prefix}-{turn}"}}'),
                        *sse("assistant.delta", '{"delta":"never consumed"}'),
                        *sse("run.completed", '{}'),
                    ]
                ),
                FakeResponse(status_code=202, text="stopping"),
            ]
        )
    transport = FakeTransport(responses.copy())
    if adapter_kind == "hermes":
        monkeypatch.setenv("HERMES_TEST_KEY", "secret")
        adapter = HermesAdapter(
            {"api_key_env": "HERMES_TEST_KEY"},
            transport,
            tmp_path / "hermes.json",
        )
    else:
        adapter = GbrainAdapter({}, transport, tmp_path / "gbrain.json")

    for turn in range(50):
        stream = adapter.stream("stable-session", f"request {turn}", 10)
        started = next(stream)
        assert started.type == BrainEventType.RUN_STARTED
        result = adapter.stop_run(started.run_id or "")
        assert result["status_code"] == 202
        stream.close()

    assert len(transport.calls) == 100
    assert all(response.closed for response in responses)


@pytest.mark.parametrize("adapter_type", [HermesAdapter, GbrainAdapter])
def test_stream_cancellation_closes_provider_response(
    monkeypatch, tmp_path: Path, adapter_type
):
    monkeypatch.setenv("API_SERVER_KEY", "secret")
    response = FakeResponse(
        lines=[
            *sse("run.started", '{"run_id":"cancel-1"}'),
            *sse("assistant.delta", '{"delta":"must not escape"}'),
        ]
    )
    adapter = adapter_type(
        {}, FakeTransport([response]), tmp_path / "cancel-sessions.json"
    )
    cancellation = CancellationToken()
    stream = adapter.stream("session-1", "hello", 10, cancellation)

    assert next(stream).type == BrainEventType.RUN_STARTED
    assert cancellation.cancel() is True
    assert list(stream) == []
    assert response.closed is True
