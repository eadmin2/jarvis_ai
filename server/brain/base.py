"""Shared contracts for agent backends used by the JARVIS voice server."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator, Mapping, Protocol


class CancellationSignal(Protocol):
    """Duck-typed cancellation contract shared with the voice pipeline."""

    @property
    def cancelled(self) -> bool: ...

    def register(self, callback: Callable[[], None]) -> Callable[[], None]: ...


class BrainAdapterError(RuntimeError):
    """An adapter operation failed before it could become a stream event."""


class BrainEventType(str, Enum):
    RUN_STARTED = "run_started"
    TEXT_DELTA = "text_delta"
    TOOL_STARTED = "tool_started"
    APPROVAL_REQUIRED = "approval_required"
    MEMORY_UPDATED = "memory_updated"
    RUN_COMPLETED = "run_completed"
    RUN_INTERRUPTED = "run_interrupted"
    ERROR = "error"


@dataclass(frozen=True)
class BrainEvent:
    """One normalized event emitted by any brain provider."""

    type: BrainEventType
    data: Mapping[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"type": self.type.value, "data": dict(self.data)}
        if self.run_id:
            value["run_id"] = self.run_id
        if self.session_id:
            value["session_id"] = self.session_id
        return value


@dataclass(frozen=True)
class BrainCapabilities:
    """Features advertised by a brain provider."""

    streaming: bool = True
    sessions: bool = True
    stop: bool = False
    approvals: bool = False
    memory: bool = False
    extra: frozenset[str] = frozenset()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | list[str]) -> "BrainCapabilities":
        if isinstance(payload, list):
            names = {str(item) for item in payload}
            values: Mapping[str, Any] = {}
        else:
            nested = payload.get("capabilities", payload)
            if isinstance(nested, list):
                names = {str(item) for item in nested}
                values = {}
            elif isinstance(nested, Mapping):
                names = {str(key) for key, enabled in nested.items() if enabled is True}
                values = nested
            else:
                raise BrainAdapterError("capabilities response must be an object or list")

        known = {"streaming", "sessions", "stop", "approvals", "memory"}
        return cls(
            streaming=bool(values.get("streaming", "streaming" in names)),
            sessions=bool(values.get("sessions", "sessions" in names)),
            stop=bool(values.get("stop", "stop" in names)),
            approvals=bool(values.get("approvals", "approvals" in names)),
            memory=bool(values.get("memory", "memory" in names)),
            extra=frozenset(names - known),
        )


class BaseBrainAdapter(ABC):
    """Small synchronous interface shared by Hermes and Gbrain."""

    provider: str

    @abstractmethod
    def discover_capabilities(self) -> BrainCapabilities:
        """Return features supported by this configured provider."""

    @abstractmethod
    def get_session_id(self, name: str, force_new: bool = False) -> str:
        """Resolve a stable named conversation to a provider session."""

    @abstractmethod
    def get_session(self, session_id: str) -> Mapping[str, Any]:
        """Fetch provider metadata for a session."""

    @abstractmethod
    def stream(
        self,
        session_id: str,
        input_text: str,
        timeout: float | None = None,
        cancellation: CancellationSignal | None = None,
    ) -> Iterator[BrainEvent]:
        """Stream a turn as normalized events."""

    @abstractmethod
    def stop_run(self, run_id: str) -> Mapping[str, Any]:
        """Request cancellation of an active run."""

    @abstractmethod
    def submit_approval(
        self,
        run_id: str,
        approval_id: str,
        decision: str,
        **metadata: Any,
    ) -> Mapping[str, Any]:
        """Allow or deny a provider approval request."""

    def post_approval(self, run_id: str, body: Mapping[str, Any]) -> Mapping[str, Any]:
        """Compatibility shim for the existing HermesAPI call site."""

        return self.submit_approval(
            run_id,
            str(body.get("approval_id") or body.get("id") or ""),
            str(body.get("decision") or "deny"),
            **{key: value for key, value in body.items() if key not in {"approval_id", "id", "decision"}},
        )
