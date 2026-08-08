"""Hermes Agent API adapter."""

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


class HermesAdapter(HTTPBrainAdapter, BaseBrainAdapter):
    """Normalize the current Hermes sessions API without changing its wire contract."""

    provider = "hermes"

    def __init__(
        self,
        config: Mapping[str, Any],
        transport: Any = None,
        state_path: Path | None = None,
    ):
        section = config.get("hermes", config)
        merged = {
            "base_url": "http://127.0.0.1:8642",
            "api_key_env": "API_SERVER_KEY",
            "api_key_required": True,
            **dict(section),
        }
        super().__init__(merged, transport)
        self.state_path = state_path or Path(__file__).resolve().parents[1] / "logs" / "hermes_sessions.json"
        self._model_locks: set[tuple[str, str, str]] = set()

    def _lock_session_model(self, session_id: str) -> None:
        """Pin an existing named session to the configured Codex runtime.

        Hermes persists a session's prior model selection. Without an explicit
        lock, changing the gateway default does not repair an older session.
        """

        model = str(self.config.get("model") or "").strip()
        provider = str(self.config.get("model_provider") or "").strip()
        if not model:
            return
        lock = (session_id, provider, model)
        if lock in self._model_locks:
            return
        body = {"model": model}
        if provider:
            body["provider"] = provider
        response = self.transport.post(
            self._url("/api/sessions/{session_id}/model", session_id=session_id),
            headers=self.headers(),
            json=body,
            timeout=15,
        )
        try:
            response.raise_for_status()
            payload = self._response_payload(response)
            runtime = payload.get("runtime", payload)
            locked_model = str(runtime.get("model") or "") if isinstance(runtime, Mapping) else ""
            if locked_model and locked_model != model:
                raise BrainAdapterError(
                    f"Hermes locked unexpected model {locked_model!r}; expected {model!r}"
                )
        except BrainAdapterError:
            raise
        except Exception as exc:
            raise BrainAdapterError(f"Hermes session model lock failed: {exc}") from exc
        finally:
            response.close()
        self._model_locks.add(lock)

    def discover_capabilities(self) -> BrainCapabilities:
        path = self.config.get("capabilities_path")
        if not path:
            return BrainCapabilities(streaming=True, sessions=True, stop=True, approvals=True, memory=True)
        response = self.transport.get(self._url(str(path)), headers=self.headers(), timeout=15)
        try:
            response.raise_for_status()
            return BrainCapabilities.from_payload(self._response_payload(response))
        except Exception as exc:
            raise BrainAdapterError(f"Hermes capability discovery failed: {exc}") from exc

    def get_session_id(self, name: str, force_new: bool = False) -> str:
        state = self._load_state(self.state_path)
        if state.get(name) and not force_new:
            session_id = state[name]
            self._lock_session_model(session_id)
            return session_id
        body: dict[str, Any] = {"title": name}
        model = str(self.config.get("model") or "").strip()
        provider = str(self.config.get("model_provider") or "").strip()
        if model:
            body["model"] = model
        if provider:
            body["provider"] = provider
        response = self.transport.post(
            self._url("/api/sessions"), headers=self.headers(), json=body, timeout=15
        )
        try:
            response.raise_for_status()
            payload = self._response_payload(response)
            session = payload.get("session", payload)
            session_id = str(session.get("id") or "") if isinstance(session, Mapping) else ""
        except Exception as exc:
            raise BrainAdapterError(f"Hermes session creation failed: {exc}") from exc
        finally:
            response.close()
        if not session_id:
            raise BrainAdapterError("Hermes session creation response had no id")
        state[name] = session_id
        self._save_state(self.state_path, state)
        self._lock_session_model(session_id)
        return session_id

    def get_session(self, session_id: str) -> Mapping[str, Any]:
        response = self.transport.get(
            self._url("/api/sessions/{session_id}", session_id=session_id),
            headers=self.headers(),
            timeout=15,
        )
        try:
            response.raise_for_status()
            return self._response_payload(response)
        except Exception as exc:
            raise BrainAdapterError(f"Hermes session lookup failed: {exc}") from exc
        finally:
            response.close()

    def stop_run(self, run_id: str) -> Mapping[str, Any]:
        response = self.transport.post(
            self._url("/v1/runs/{run_id}/stop", run_id=run_id), headers=self.headers(), timeout=15
        )
        return self._status_result(response)

    def submit_approval(
        self,
        run_id: str,
        approval_id: str,
        decision: str,
        **metadata: Any,
    ) -> Mapping[str, Any]:
        body = {"approval_id": approval_id, "decision": decision, **metadata}
        response = self.transport.post(
            self._url("/v1/runs/{run_id}/approval", run_id=run_id),
            headers=self.headers(),
            json=body,
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
        pending_completion: dict[str, Any] | None = None
        completion_emitted = False
        def unsubscribe() -> None:
            pass
        try:
            if cancellation is not None and cancellation.cancelled:
                return
            response = self.transport.post(
                self._url("/api/sessions/{session_id}/chat/stream", session_id=session_id),
                headers={**self.headers(), "Accept": "text/event-stream"},
                json={"input": input_text},
                stream=True,
                timeout=(10, timeout or float(self.config.get("timeout", 240))),
            )
            if response.status_code >= 400:
                yield self._error(
                    f"Hermes session chat HTTP {response.status_code}: {str(response.text)[:300]}",
                    session_id,
                )
                return
            if cancellation is not None:
                unsubscribe = cancellation.register(response.close)
            response.encoding = "utf-8"
            for record in iter_sse_records(response.iter_lines(decode_unicode=True)):
                if cancellation is not None and cancellation.cancelled:
                    break
                event, run_id, pending_completion = self._normalize(
                    record, session_id, run_id, pending_completion
                )
                if event:
                    if event.type in {BrainEventType.RUN_COMPLETED, BrainEventType.RUN_INTERRUPTED}:
                        completion_emitted = True
                    yield event
            if pending_completion and not completion_emitted:
                interrupted = bool(pending_completion.get("interrupted"))
                yield BrainEvent(
                    BrainEventType.RUN_INTERRUPTED if interrupted else BrainEventType.RUN_COMPLETED,
                    pending_completion,
                    run_id,
                    session_id,
                )
        except Exception as exc:
            if cancellation is None or not cancellation.cancelled:
                yield self._error(f"Hermes stream error: {exc}", session_id, run_id)
        finally:
            unsubscribe()
            if response is not None:
                response.close()

    def _normalize(
        self,
        record: SSERecord,
        session_id: str,
        run_id: str | None,
        pending_completion: dict[str, Any] | None,
    ) -> tuple[BrainEvent | None, str | None, dict[str, Any] | None]:
        event_name, data = record.event, record.data
        current_run = str(data.get("run_id") or run_id or "") or None
        if event_name == "run.started":
            return BrainEvent(BrainEventType.RUN_STARTED, data, current_run, session_id), current_run, pending_completion
        if event_name == "assistant.delta":
            delta = str(data.get("delta") or "")
            event = BrainEvent(BrainEventType.TEXT_DELTA, {"delta": delta}, current_run, session_id) if delta else None
            return event, current_run, pending_completion
        if event_name == "tool.started":
            name = str(data.get("tool_name") or data.get("name") or "tool")
            if name.startswith("_"):
                return None, current_run, pending_completion
            payload = {"name": name, "preview": str(data.get("preview") or "")[:200]}
            return BrainEvent(BrainEventType.TOOL_STARTED, payload, current_run, session_id), current_run, pending_completion
        if "approval" in event_name:
            payload = dict(data)
            payload.setdefault("approval_id", data.get("id"))
            return BrainEvent(BrainEventType.APPROVAL_REQUIRED, payload, current_run, session_id), current_run, pending_completion
        if event_name in {"memory.updated", "memory_update"}:
            return BrainEvent(BrainEventType.MEMORY_UPDATED, data, current_run, session_id), current_run, pending_completion
        if event_name == "assistant.completed":
            completion = {
                "content": data.get("content") or "",
                "interrupted": bool(data.get("interrupted")),
            }
            return None, current_run, completion
        if event_name == "run.completed":
            payload = {**(pending_completion or {}), **dict(data)}
            interrupted = bool(payload.get("interrupted"))
            event_type = BrainEventType.RUN_INTERRUPTED if interrupted else BrainEventType.RUN_COMPLETED
            return BrainEvent(event_type, payload, current_run, session_id), current_run, None
        if event_name in {"run.interrupted", "run.stopped"}:
            return BrainEvent(BrainEventType.RUN_INTERRUPTED, data, current_run, session_id), current_run, None
        if event_name in {"run.failed", "error"}:
            message = data.get("message") or data.get("error") or json.dumps(data)[:300]
            return self._error(str(message), session_id, current_run, data), current_run, pending_completion
        return None, current_run, pending_completion

    @staticmethod
    def _error(
        message: str,
        session_id: str,
        run_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> BrainEvent:
        payload: dict[str, Any] = {"message": message, "provider": "hermes"}
        if details:
            payload["details"] = dict(details)
        return BrainEvent(BrainEventType.ERROR, payload, run_id, session_id)

    def chat_stream_events(
        self,
        session_id: str,
        input_text: str,
        timeout: float,
        cancellation: CancellationSignal | None = None,
    ) -> Iterator[tuple[str, str]]:
        """Yield the legacy tuples expected by the current VoicePipelineServer."""

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
                raise RuntimeError(str(event.data.get("message") or "Hermes stream error"))
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
