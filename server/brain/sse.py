"""Minimal Server-Sent Events parsing with no dependency beyond requests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Iterator


@dataclass(frozen=True)
class SSERecord:
    event: str
    data: dict[str, Any]


def iter_sse_records(lines: Iterable[str | bytes | None]) -> Iterator[SSERecord]:
    """Parse SSE records, including multi-line data and a final unterminated record."""

    event_name = ""
    data_lines: list[str] = []

    def emit() -> SSERecord | None:
        nonlocal event_name, data_lines
        if not data_lines:
            event_name = ""
            return None
        raw_data = "\n".join(data_lines)
        current_event = event_name
        event_name = ""
        data_lines = []
        try:
            payload = json.loads(raw_data)
        except json.JSONDecodeError:
            return SSERecord("error", {"message": "invalid SSE JSON", "raw": raw_data[:300]})
        if not isinstance(payload, dict):
            return SSERecord("error", {"message": "SSE data must be a JSON object"})
        return SSERecord(current_event or str(payload.get("event") or payload.get("type") or ""), payload)

    for raw in lines:
        if raw is None:
            continue
        line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        if line == "":
            record = emit()
            if record:
                yield record
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            event_name = line[6:].lstrip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    record = emit()
    if record:
        yield record
