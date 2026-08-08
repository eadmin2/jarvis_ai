from pathlib import Path

from server.scripts.frankbrain_sync import (
    _build_import_command,
    _is_authored_page,
    _mirror_authored_pages,
)


def test_writeback_boundary_excludes_projection_and_app_aggregates() -> None:
    assert _is_authored_page(Path("queries/founder-answer.md")) is True
    assert _is_authored_page(Path("concepts/voice-loop.md")) is True
    assert _is_authored_page(Path("frankbrain/concepts/projected.md")) is False
    assert _is_authored_page(Path("index.md")) is False
    assert _is_authored_page(Path("log.md")) is False
    assert _is_authored_page(Path("image.png")) is False


def test_mirror_copies_authored_pages_and_prunes_only_staging(tmp_path: Path) -> None:
    project = tmp_path / "FrankBrain"
    wiki = project / "wiki"
    (wiki / "queries").mkdir(parents=True)
    (wiki / "frankbrain").mkdir()
    (wiki / "queries" / "answer.md").write_text("# Answer\n", encoding="utf-8")
    (wiki / "frankbrain" / "projected.md").write_text("# Generated\n", encoding="utf-8")
    (wiki / "index.md").write_text("# Index\n", encoding="utf-8")
    staging = tmp_path / "writeback"
    (staging / "queries").mkdir(parents=True)
    (staging / "queries" / "stale.md").write_text("# Stale\n", encoding="utf-8")

    result = _mirror_authored_pages(project, staging)

    assert result == {"pages": 1, "updated": 1, "removed": 1}
    assert (staging / "queries" / "answer.md").read_text(encoding="utf-8") == "# Answer\n"
    assert not (staging / "queries" / "stale.md").exists()
    assert not (staging / "frankbrain" / "projected.md").exists()


def test_gbrain_import_is_explicitly_source_scoped() -> None:
    command = _build_import_command(
        Path("/Users/founder/.bun/bin/gbrain"),
        Path("/Users/founder/Library/Application Support/FrankBrain/writeback"),
        "frankbrain",
    )

    assert command == [
        "/Users/founder/.bun/bin/gbrain",
        "import",
        "/Users/founder/Library/Application Support/FrankBrain/writeback",
        "--source-id",
        "frankbrain",
        "--workers",
        "1",
        "--json",
    ]
