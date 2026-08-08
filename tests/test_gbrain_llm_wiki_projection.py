from pathlib import Path

import pytest

from server.scripts.gbrain_llm_wiki_projection import _render_page, safe_slug_path


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
