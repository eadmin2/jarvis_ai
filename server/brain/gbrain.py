"""Configurable local HTTP/SSE adapter for Gbrain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Mapping

from .base import (
    BaseBrainAdapter,
    BrainAdapterError,
    BrainCapabilities,
    BrainEvent,
    BrainEventType,
    CancellationSignal,
)
from .http_adapter import HTTPBrainAdapter
from .sse import SSERecord, iter_sse_records


DEFAULT_ENDPOINTS = {
    "capabilities": "/api/capabilities",
    "create_session": "/api/sessions",
    "get_session": "/api/sessions/{session_id}",
    "stream": "/api/sessions/{session_id}/chat/stream",
    "stop": "/v1/runs/{run_id}/stop",
    "approval": "/v1/runs/{run_id}/approval",
}

DEFAULT_EVENTS = {
    "run.started": BrainEventType.RUN_STARTED.value,
    "assistant.delta": BrainEventType.TEXT_DELTA.value,
    "tool.started": BrainEventType.TOOL_STARTED.value,
    "approval.required": BrainEventType.APPROVAL_REQUIRED.value,
    "approval_request": BrainEventType.APPROVAL_REQUIRED.value,
    "memory.updated": BrainEventType.MEMORY_UPDATED.value,
    "run.completed": BrainEventType.RUN_COMPLETED.value,
    "run.interrupted": BrainEventType.RUN_INTERRUPTED.value,
    "run.failed": BrainEventType.ERROR.value,
    "error": BrainEventType.ERROR.value,
}


class GbrainAdapter(HTTPBrainAdapter, BaseBrainAdapter):
    """Gbrain client whose routes and upstream SSE names come entirely from config."""

    provider = "gbrain"

    def __init__(
        self,
        config: Mapping[str, Any],
        transport: Any = None,
        state_path: Path | None = None,
    ):
        section = config.get("gbrain", config)
        merged = {
            "base_url": "http://127.0.0.1:8767",
            "api_key_env": "GBRAIN_API_KEY",
            "api_key_required": False,
            **dict(section),
        }
        merged["endpoints"] = {**DEFAULT_ENDPOINTS, **dict(section.get("endpoints") or {})}
        merged["events"] = {**DEFAULT_EVENTS, **dict(section.get("events") or {})}
        super().__init__(merged, transport)
        self.state_path = state_path or Path(__file__).resolve().parents[1] / "logs" / "gbrain_sessions.json"

    def _endpoint(self, name: str, **values: str) -> str:
        path = self.config["endpoints"].get(name)
        if not path:
            raise BrainAdapterError(f"Gbrain endpoint '{name}' is disabled")
        return self._url(str(path), **values)

    def discover_capabilities(self) -> BrainCapabilities:
        response = self.transport.get(
            self._endpoint("capabilities"), headers=self.headers(), timeout=15
        )
        try:
            response.raise_for_status()
            return BrainCapabilities.from_payload(self._response_payload(response))
        except Exception as exc:
            raise BrainAdapterError(f"Gbrain capability discovery failed: {exc}") from exc

    def get_session_id(self, name: str, force_new: bool = False) -> str:
        state = self._load_state(self.state_path)
        if state.get(name) and not force_new:
            return state[name]
        response = self.transport.post(
            self._endpoint("create_session"),
            headers=self.headers(),
            json={"title": name},
            timeout=15,
        )
        try:
            response.raise_for_status()
            payload = self._response_payload(response)
            session = payload.get("session", payload)
            session_id = str(session.get("id") or "") if isinstance(session, Mapping) else ""
        except Exception as exc:
            raise BrainAdapterError(f"Gbrain session creation failed: {exc}") from exc
        if not session_id:
            raise BrainAdapterError("Gbrain session creation response had no id")
        state[name] = session_id
        self._save_state(self.state_path, state)
        return session_id

    def get_session(self, session_id: str) -> Mapping[str, Any]:
        response = self.transport.get(
            self._endpoint("get_session", session_id=session_id),
            headers=self.headers(),
            timeout=15,
        )
        try:
            response.raise_for_status()
            return self._response_payload(response)
        except Exception as exc:
            raise BrainAdapterError(f"Gbrain session lookup failed: {exc}") from exc

    def stop_run(self, run_id: str) -> Mapping[str, Any]:
        response = self.transport.post(
            self._endpoint("stop", run_id=run_id), headers=self.headers(), timeout=15
        )
        return self._status_result(response)

    def submit_approval(
        self,
        run_id: str,
        approval_id: str,
        decision: str,
        **metadata: Any,
    ) -> Mapping[str, Any]:
        response = self.transport.post(
            self._endpoint("approval", run_id=run_id),
            headers=self.headers(),
            json={"approval_id": approval_id, "decision": decision, **metadata},
            timeout=15,
        )
        return self._status_result(response)

    def stream(
        self,
        session_id: str,
        input_text: str,
        timeout: float | None = None,
        cancellation: CancellationSignal | None = None,
    ) -> Iterator[BrainEvent]:
        response = None
        run_id: str | None = None
        def unsubscribe() -> None:
            pass
        try:
            if cancellation is not None and cancellation.cancelled:
                return
            response = self.transport.post(
                self._endpoint("stream", session_id=session_id),
                headers={**self.headers(), "Accept": "text/event-stream"},
                json={"input": input_text},
                stream=True,
                timeout=(10, timeout or float(self.config.get("timeout", 240))),
            )
            if response.status_code >= 400:
                yield self._error(
                    f"Gbrain stream HTTP {response.status_code}: {str(response.text)[:300]}",
                    session_id,
                )
                return
            if cancellation is not None:
                unsubscribe = cancellation.register(response.close)
            response.encoding = "utf-8"
            for record in iter_sse_records(response.iter_lines(decode_unicode=True)):
                if cancellation is not None and cancellation.cancelled:
                    break
                event, run_id = self._normalize(record, session_id, run_id)
                if event:
                    yield event
        except Exception as exc:
            if cancellation is None or not cancellation.cancelled:
                yield self._error(f"Gbrain stream error: {exc}", session_id, run_id)
        finally:
            unsubscribe()
            if response is not None:
                response.close()

    def _normalize(
        self, record: SSERecord, session_id: str, run_id: str | None
    ) -> tuple[BrainEvent | None, str | None]:
        mapped = self.config["events"].get(record.event)
        if not mapped:
            return None, run_id
        try:
            event_type = BrainEventType(mapped)
        except ValueError:
            return self._error(f"unknown normalized event type: {mapped}", session_id, run_id), run_id
        data = dict(record.data)
        current_run = str(data.get("run_id") or data.get("runId") or run_id or "") or None
        if event_type == BrainEventType.TEXT_DELTA:
            data = {"delta": data.get("delta") or data.get("text") or data.get("content") or ""}
        elif event_type == BrainEventType.TOOL_STARTED:
            data.setdefault("name", data.get("tool_name") or data.get("toolName") or "tool")
        elif event_type == BrainEventType.APPROVAL_REQUIRED:
            data.setdefault("approval_id", data.get("id") or data.get("approvalId"))
        elif event_type == BrainEventType.ERROR:
            data.setdefault("message", data.get("error") or "Gbrain stream error")
            data.setdefault("provider", "gbrain")
        return BrainEvent(event_type, data, current_run, session_id), current_run

    def chat_stream_events(
        self,
        session_id: str,
        input_text: str,
        timeout: float,
        cancellation: CancellationSignal | None = None,
    ) -> Iterator[tuple[str, str]]:
        """Yield legacy tuples so current voice and typed-chat consumers can use Gbrain."""

        legacy = {
            BrainEventType.RUN_STARTED: "run",
            BrainEventType.TEXT_DELTA: "text",
            BrainEventType.TOOL_STARTED: "tool",
            BrainEventType.APPROVAL_REQUIRED: "approval",
            BrainEventType.RUN_COMPLETED: "final",
            BrainEventType.RUN_INTERRUPTED: "final",
        }
        for event in self.stream(session_id, input_text, timeout, cancellation):
            if event.type == BrainEventType.ERROR:
                raise RuntimeError(str(event.data.get("message") or "Gbrain stream error"))
            kind = legacy.get(event.type)
            if not kind:
                continue
            if event.type == BrainEventType.RUN_STARTED:
                value = event.run_id or ""
            elif event.type == BrainEventType.TEXT_DELTA:
                value = str(event.data.get("delta") or "")
            else:
                value = json.dumps(dict(event.data))[:2000]
            yield kind, value

    @staticmethod
    def _error(message: str, session_id: str, run_id: str | None = None) -> BrainEvent:
        return BrainEvent(
            BrainEventType.ERROR,
            {"message": message, "provider": "gbrain"},
            run_id,
            session_id,
        )
