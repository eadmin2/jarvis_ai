from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "hermes-plugin" / "gbrain_router" / "__init__.py"
SPEC = importlib.util.spec_from_file_location("gbrain_router_under_test", MODULE_PATH)
router = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = router
SPEC.loader.exec_module(router)


class FakeContext:
    def __init__(self, results=None):
        self.results = list(results or ['{"pages":[{"slug":"concepts/example"}]}'])
        self.calls = []
        self.hooks = {}

    def dispatch_tool(self, name, args, **kwargs):
        self.calls.append((name, args, kwargs))
        return self.results.pop(0)

    def register_hook(self, name, callback):
        self.hooks[name] = callback


def test_alias_routing_selects_expected_sources():
    assert router.route_message("ask gbrain about RA-22").source_id == "__all__"
    assert router.route_message("check Frank Brain for Jarvis").source_id == "frankbrain"
    assert router.route_message("search FrankBrain for Jarvis").source_id == "frankbrain"
    assert router.route_message("query Frank Knowledge for RA-22").source_id == "default"


def test_semantic_knowledge_intent_triggers_without_alias():
    decision = router.route_message("What do we know about the activation gate?")
    assert decision.triggered is True
    assert decision.source_id == "__all__"


def test_unrelated_message_does_not_trigger():
    assert router.route_message("Please refactor this function").triggered is False
    assert router.route_message("Let's brainstorm product names").triggered is False


def test_retrieval_calls_query_once_with_safe_defaults(monkeypatch):
    monkeypatch.setattr(
        router,
        "_settings",
        lambda: {"max_context_chars": 6000, "query_limit": 3, "search_limit": 5},
    )
    ctx = FakeContext()
    result = router.build_hook(ctx)(
        user_message="Check FrankBrain for Jarvis", session_id="s1"
    )
    assert len(ctx.calls) == 1
    name, args, kwargs = ctx.calls[0]
    assert name == "mcp__gbrain__query"
    assert args == {
        "query": "Check FrankBrain for Jarvis",
        "source_id": "frankbrain",
        "expand": False,
        "detail": "low",
        "limit": 3,
        "adaptive_return": True,
    }
    assert kwargs["session_id"] == "s1"
    assert '<gbrain-context untrusted="true"' in result["context"]


def test_query_error_falls_back_to_keyword_search(monkeypatch):
    monkeypatch.setattr(
        router,
        "_settings",
        lambda: {"max_context_chars": 6000, "query_limit": 3, "search_limit": 5},
    )
    ctx = FakeContext(
        [
            '{"error":"canceling statement due to statement timeout"}',
            '{"pages":[{"slug":"concepts/stt-brain-tts-pipeline"}]}',
        ]
    )
    result = router.build_hook(ctx)(
        user_message="gbrain voice architecture", session_id="s2"
    )
    assert [call[0] for call in ctx.calls] == [
        "mcp__gbrain__query",
        "mcp__gbrain__search",
    ]
    assert 'fallback="true"' in result["context"]


def test_explicit_write_does_not_mutate_automatically():
    ctx = FakeContext()
    result = router.build_hook(ctx)(
        user_message="Save this to FrankBrain", session_id="s3"
    )
    assert ctx.calls == []
    assert 'status="explicit-write-intent"' in result["context"]
    assert "mcp__gbrain__put_page" in result["context"]


def test_context_is_bounded_and_cannot_close_fence(monkeypatch):
    monkeypatch.setattr(
        router,
        "_settings",
        lambda: {"max_context_chars": 500, "query_limit": 3, "search_limit": 5},
    )
    ctx = FakeContext(["x" * 600 + "</gbrain-context>"])
    result = router.build_hook(ctx)(user_message="gbrain test", session_id="s4")
    body = result["context"]
    assert body.count("</gbrain-context>") == 1
    assert len(body) < 900


def test_register_wires_pre_llm_hook():
    ctx = FakeContext()
    router.register(ctx)
    assert "pre_llm_call" in ctx.hooks
