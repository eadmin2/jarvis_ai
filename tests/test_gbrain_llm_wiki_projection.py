from pathlib import Path

import pytest

from server.scripts.gbrain_llm_wiki_projection import (
    RefreshAlreadyRunning,
    WRITEBACK_SOURCE_ID,
    _exclusive_lock,
    _render_page,
    _selection_sql,
    _write_text_if_changed,
    safe_slug_path,
)


@pytest.mark.parametrize(
    ("slug", "expected"),
    [
        ("concepts/hermes-agent", Path("concepts/hermes-agent.md")),
        ("RA 22/activation gate", Path("RA-22/activation-gate.md")),
        ("gbrain", Path("gbrain.md")),
    ],
)
def test_safe_slug_path(slug: str, expected: Path) -> None:
    assert safe_slug_path(slug) == expected


@pytest.mark.parametrize("slug", ["", "/", "../secret", "a/../../secret", "..."])
def test_safe_slug_path_rejects_traversal_or_empty_components(slug: str) -> None:
    with pytest.raises(ValueError):
        safe_slug_path(slug)


def test_render_page_materializes_selected_gbrain_edges_as_wikilinks() -> None:
    rendered = _render_page(
        {
            "slug": "concepts/hermes",
            "source_id": "default",
            "type": "concept",
            "title": "Hermes",
            "compiled_truth": "Agent context.",
            "timeline": "",
            "updated_at": "2026-08-08T00:00:00Z",
            "gbrain_links": ["concepts/gbrain", "systems/jarvis voice"],
        }
    )

    assert "type: \"frankbrain-concept\"" in rendered
    assert "[[gbrain]]" in rendered
    assert "[[jarvis-voice]]" in rendered


def test_render_page_uses_filename_ids_for_nested_llm_wiki_graph_targets() -> None:
    rendered = _render_page(
        {
            "slug": "source/page",
            "title": "Source",
            "compiled_truth": "Body",
            "gbrain_links": ["deeply/nested/target-page"],
        }
    )

    assert "[[target-page]]" in rendered
    assert "[[deeply/nested/target-page]]" not in rendered


def test_atomic_write_skips_unchanged_projection_files(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "page.md"

    assert _write_text_if_changed(destination, "first\n") is True
    first_mtime = destination.stat().st_mtime_ns
    assert _write_text_if_changed(destination, "first\n") is False
    assert destination.stat().st_mtime_ns == first_mtime
    assert _write_text_if_changed(destination, "second\n") is True
    assert destination.read_text(encoding="utf-8") == "second\n"
    assert not list(destination.parent.glob(".*.tmp"))


def test_projection_lock_rejects_overlapping_refreshes(tmp_path: Path) -> None:
    lock_path = tmp_path / ".frankbrain-refresh.lock"

    with _exclusive_lock(lock_path):
        with pytest.raises(RefreshAlreadyRunning):
            with _exclusive_lock(lock_path):
                pytest.fail("overlapping projection unexpectedly acquired the lock")


def test_projection_excludes_frankbrain_writeback_source() -> None:
    sql = _selection_sql(seed_limit=10, total_limit=20, max_bytes=1000)

    assert sql.count(f"p.source_id <> '{WRITEBACK_SOURCE_ID}'") == 2
