#!/usr/bin/env python3
"""Synchronize canonical knowledge with the founder-local FrankBrain wiki.

The generated ``wiki/frankbrain`` projection flows from the canonical backend
to LLM Wiki. Pages authored by LLM Wiki outside that directory flow into the
dedicated, federated ``frankbrain`` source. Generated and app-aggregate files
never flow back, preventing projection feedback loops.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

try:
    from .gbrain_llm_wiki_projection import (
        RefreshAlreadyRunning,
        _exclusive_lock,
        _write_text_if_changed,
    )
except ImportError:  # Direct execution adds this script directory to sys.path.
    from gbrain_llm_wiki_projection import (
        RefreshAlreadyRunning,
        _exclusive_lock,
        _write_text_if_changed,
    )


SOURCE_ID = "frankbrain"
APP_MANAGED_ROOT_FILES = {
    "index.md",
    "log.md",
    "overview.md",
    "purpose.md",
    "schema.md",
}


def _is_authored_page(relative: Path) -> bool:
    """Return whether a wiki-relative Markdown file should flow to Gbrain."""
    if relative.suffix.lower() != ".md" or not relative.parts:
        return False
    if relative.parts[0] == "frankbrain":
        return False
    return not (len(relative.parts) == 1 and relative.name.lower() in APP_MANAGED_ROOT_FILES)


def _authored_pages(project: Path) -> list[tuple[Path, Path]]:
    wiki_root = project.resolve() / "wiki"
    if not wiki_root.is_dir():
        return []
    pages: list[tuple[Path, Path]] = []
    for source in wiki_root.rglob("*.md"):
        relative = source.relative_to(wiki_root)
        if _is_authored_page(relative):
            pages.append((source, relative))
    return sorted(pages, key=lambda item: item[1].as_posix())


def _mirror_authored_pages(project: Path, staging: Path) -> dict[str, int]:
    """Mirror only LLM Wiki-authored pages into the Gbrain import staging tree."""
    staging = staging.resolve()
    expected: set[Path] = set()
    updated = 0
    for source, relative in _authored_pages(project):
        destination = staging / relative
        expected.add(destination)
        updated += _write_text_if_changed(
            destination, source.read_text(encoding="utf-8")
        )

    removed = 0
    if staging.is_dir():
        for stale in staging.rglob("*.md"):
            if stale not in expected:
                stale.unlink()
                removed += 1
        for directory in sorted(
            (path for path in staging.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            try:
                directory.rmdir()
            except OSError:
                pass
    staging.mkdir(parents=True, exist_ok=True)
    return {"pages": len(expected), "updated": updated, "removed": removed}


def _build_import_command(gbrain: Path, staging: Path, source_id: str) -> list[str]:
    return [
        str(gbrain),
        "import",
        str(staging),
        "--source-id",
        source_id,
        "--workers",
        "1",
        "--json",
    ]


def _run_command(
    command: Sequence[str],
    runner: Callable[..., Any] = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    completed = runner(command, check=True, text=True, capture_output=True)
    if completed.stderr:
        print(completed.stderr.rstrip(), file=sys.stderr)
    return completed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--gbrain", type=Path, required=True)
    parser.add_argument("--source-id", default=SOURCE_ID)
    parser.add_argument("--write", action="store_true", help="synchronize; default is dry-run")
    return parser


def _sync(args: argparse.Namespace) -> int:
    pages = _authored_pages(args.project)
    if not args.write:
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "project": str(args.project),
                    "authored_pages": len(pages),
                    "source_id": args.source_id,
                },
                indent=2,
            )
        )
        return 0

    mirror = _mirror_authored_pages(args.project, args.staging)
    import_result: dict[str, Any] | None = None
    if mirror["pages"]:
        completed = _run_command(
            _build_import_command(args.gbrain, args.staging.resolve(), args.source_id)
        )
        try:
            import_result = json.loads(completed.stdout)
        except json.JSONDecodeError:
            import_result = {"stdout": completed.stdout.strip()}

    projection_script = Path(__file__).with_name("gbrain_llm_wiki_projection.py")
    projection = _run_command(
        [
            "/opt/homebrew/bin/python3",
            str(projection_script),
            "--project",
            str(args.project.resolve()),
            "--write",
        ]
    )
    print(
        json.dumps(
            {
                "mode": "write",
                "project": str(args.project),
                "writeback": mirror,
                "gbrain_import": import_result,
                "projection": projection.stdout.strip(),
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if not args.write:
        return _sync(args)
    lock_path = args.staging.resolve().parent / ".frankbrain-sync.lock"
    try:
        with _exclusive_lock(lock_path):
            return _sync(args)
    except RefreshAlreadyRunning:
        print(
            json.dumps(
                {
                    "mode": "skipped",
                    "project": str(args.project),
                    "reason": "sync-already-running",
                },
                indent=2,
            )
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
