"""Reusable HTTP behavior for local brain providers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import requests

from .base import BrainAdapterError


class HTTPBrainAdapter:
    """Internal helper; concrete adapters still define their public contract."""

    def __init__(self, config: Mapping[str, Any], transport: Any = None):
        self.config = dict(config)
        self.transport = transport or requests.Session()

    @property
    def base_url(self) -> str:
        return str(self.config.get("base_url") or "").rstrip("/")

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        env_name = str(self.config.get("api_key_env") or "")
        key = os.environ.get(env_name, "") if env_name else ""
        if key:
            headers["Authorization"] = f"Bearer {key}"
        elif self.config.get("api_key_required", bool(env_name)):
            raise BrainAdapterError(f"{self.provider} API key not found in environment variable {env_name}")
        session_key = self.config.get("session_key")
        if session_key:
            headers["X-Hermes-Session-Key"] = str(session_key)
        return headers

    def _url(self, path: str, **values: str) -> str:
        try:
            rendered = path.format(**values)
        except KeyError as exc:
            raise BrainAdapterError(f"missing endpoint template value: {exc.args[0]}") from exc
        if rendered.startswith("http://") or rendered.startswith("https://"):
            return rendered
        return f"{self.base_url}/{rendered.lstrip('/')}"

    @staticmethod
    def _response_payload(response: Any) -> Mapping[str, Any]:
        try:
            payload = response.json()
        except Exception as exc:
            raise BrainAdapterError("provider returned a non-JSON response") from exc
        if not isinstance(payload, Mapping):
            raise BrainAdapterError("provider returned JSON that was not an object")
        return payload

    @staticmethod
    def _status_result(response: Any) -> dict[str, Any]:
        try:
            return {
                "status_code": int(response.status_code),
                "body": str(getattr(response, "text", ""))[:300],
            }
        finally:
            response.close()

    @staticmethod
    def _load_state(path: Path) -> dict[str, str]:
        try:
            import json

            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _save_state(path: Path, state: Mapping[str, str]) -> None:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(state)), encoding="utf-8")
