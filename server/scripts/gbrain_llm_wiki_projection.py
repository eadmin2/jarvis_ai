#!/usr/bin/env python3
"""Create a bounded, one-way FrankBrain projection for LLM Wiki.

The local knowledge backend remains canonical. This script reads selected local
PostgreSQL pages and writes derived Markdown under an LLM Wiki project's
``wiki/frankbrain`` directory. It never changes the backend and never removes
files from the projection directory.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import json
import os
import re
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterator


DEFAULT_PSQL = "/opt/homebrew/opt/postgresql@17/bin/psql"
WRITEBACK_SOURCE_ID = "frankbrain"
DEFAULT_TOPICS = (
    "hermes | jarvis | gbrain | faos | founder | voice | dashboard | "
    "agentic | governance | research | ra22 | ra6"
)
SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


class RefreshAlreadyRunning(RuntimeError):
    """Raised when another FrankBrain projection owns the refresh lock."""


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    """Hold a non-blocking process lock for one projection refresh."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RefreshAlreadyRunning(str(path)) from exc
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _write_text_if_changed(destination: Path, content: str) -> bool:
    """Atomically replace a projection file only when its content changed."""
    try:
        if destination.read_text(encoding="utf-8") == content:
            return False
    except FileNotFoundError:
        pass

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def safe_slug_path(slug: str) -> Path:
    """Map an untrusted database slug to a relative Markdown path."""
    pure = PurePosixPath(slug.strip().strip("/"))
    if not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"unsafe or empty slug: {slug!r}")
    parts = [SAFE_COMPONENT.sub("-", part).strip(".-") for part in pure.parts]
    if any(not part for part in parts):
        raise ValueError(f"slug has an empty sanitized component: {slug!r}")
    return Path(*parts).with_suffix(".md")


def _selection_sql(seed_limit: int, total_limit: int, max_bytes: int) -> str:
    # Limits are parsed as positive integers before interpolation. Topics are a
    # code-owned constant, not caller-controlled SQL.
    return f"""
COPY (
  WITH ranked_seeds AS (
    SELECT p.id,
           ts_rank_cd(p.search_vector, to_tsquery('english', '{DEFAULT_TOPICS}')) AS rank
      FROM pages p
     WHERE p.deleted_at IS NULL
       AND p.page_kind = 'markdown'
       AND p.source_id <> '{WRITEBACK_SOURCE_ID}'
       AND octet_length(p.compiled_truth) <= {max_bytes}
       AND (
         p.search_vector @@ to_tsquery('english', '{DEFAULT_TOPICS}')
         OR lower(p.slug || ' ' || p.title) ~
            '(hermes|jarvis|gbrain|faos|founder|voice|dashboard|agentic|governance|ra-?22|ra-?6)'
       )
     ORDER BY rank DESC, p.updated_at DESC
     LIMIT {seed_limit}
  ), neighbor_ids AS (
    SELECT CASE WHEN l.from_page_id = s.id THEN l.to_page_id ELSE l.from_page_id END AS id,
           max(s.rank) AS rank
      FROM ranked_seeds s
      JOIN links l ON l.from_page_id = s.id OR l.to_page_id = s.id
     GROUP BY 1
  ), selected AS (
    SELECT id, rank, 0 AS tier FROM ranked_seeds
    UNION ALL
    SELECT n.id, n.rank, 1 AS tier
      FROM neighbor_ids n
     WHERE NOT EXISTS (SELECT 1 FROM ranked_seeds s WHERE s.id = n.id)
  )
  SELECT replace(
           encode(convert_to(row_to_json(out_row)::text, 'UTF8'), 'base64'),
           E'\\n',
           ''
         )
    FROM (
      SELECT p.slug, p.source_id, p.type, p.title, p.compiled_truth, p.timeline,
             p.updated_at, s.tier, s.rank,
             ARRAY(
               SELECT DISTINCT linked.slug
                 FROM links l
                 JOIN selected linked_selection
                   ON linked_selection.id = CASE
                     WHEN l.from_page_id = p.id THEN l.to_page_id
                     ELSE l.from_page_id
                   END
                 JOIN pages linked ON linked.id = linked_selection.id
                WHERE l.from_page_id = p.id OR l.to_page_id = p.id
                ORDER BY linked.slug
             ) AS gbrain_links
        FROM selected s
        JOIN pages p ON p.id = s.id
       WHERE p.deleted_at IS NULL
         AND p.page_kind = 'markdown'
         AND p.source_id <> '{WRITEBACK_SOURCE_ID}'
         AND octet_length(p.compiled_truth) <= {max_bytes}
       ORDER BY s.tier, s.rank DESC, p.updated_at DESC
       LIMIT {total_limit}
    ) out_row
) TO STDOUT;
"""


def _read_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    command = [
        args.psql,
        "-X",
        "-h",
        args.host,
        "-p",
        str(args.port),
        "-U",
        args.user,
        "-d",
        args.database,
        "-At",
        "-c",
        _selection_sql(args.seed_limit, args.limit, args.max_bytes),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return [
        json.loads(base64.b64decode(line).decode("utf-8"))
        for line in completed.stdout.splitlines()
        if line.strip()
    ]


def _render_page(row: dict[str, Any]) -> str:
    title = str(row.get("title") or row["slug"])
    page_type = str(row.get("type") or "note")
    updated_at = str(row.get("updated_at") or "")
    body = str(row.get("compiled_truth") or "").strip()
    timeline = str(row.get("timeline") or "").strip()
    metadata = [
        "---",
        f"title: {json.dumps(title, ensure_ascii=False)}",
        f"type: {json.dumps('frankbrain-' + page_type, ensure_ascii=False)}",
        f"backend_slug: {json.dumps(str(row['slug']), ensure_ascii=False)}",
        f"backend_source_id: {json.dumps(str(row.get('source_id') or 'default'))}",
        f"backend_updated_at: {json.dumps(updated_at)}",
        "projection: frankbrain-one-way",
        "---",
        "",
        f"# {title}",
        "",
        body,
    ]
    if timeline:
        metadata.extend(["", "## FrankBrain timeline", "", timeline])
    links = row.get("gbrain_links") or []
    if links:
        wikilinks = []
        for slug in links:
            # LLM Wiki currently indexes graph nodes by filename stem rather
            # than project-relative path. Use that exact identifier so links
            # to nested projection pages resolve in the graph visualization.
            target = safe_slug_path(str(slug)).stem
            wikilinks.append(f"- [[{target}]]")
        metadata.extend(["", "## FrankBrain graph links", "", *wikilinks])
    return "\n".join(metadata).rstrip() + "\n"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True, help="LLM Wiki project root")
    parser.add_argument("--psql", default=DEFAULT_PSQL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=_positive_int, default=5432)
    parser.add_argument("--user", default="thanhlt")
    parser.add_argument("--database", default="gbrain")
    parser.add_argument("--seed-limit", type=_positive_int, default=100)
    parser.add_argument("--limit", type=_positive_int, default=250)
    parser.add_argument("--max-bytes", type=_positive_int, default=500_000)
    parser.add_argument("--write", action="store_true", help="write files; default is dry-run")
    return parser


def _run(args: argparse.Namespace) -> int:
    rows = _read_rows(args)
    projected: list[dict[str, Any]] = []
    for row in rows:
        relative = safe_slug_path(str(row["slug"]))
        projected.append({"slug": row["slug"], "path": relative.as_posix(), "tier": row["tier"]})

    summary = {
        "mode": "write" if args.write else "dry-run",
        "project": str(args.project),
        "pages": len(projected),
        "seed_pages": sum(item["tier"] == 0 for item in projected),
        "neighbor_pages": sum(item["tier"] == 1 for item in projected),
    }
    if not args.write:
        print(json.dumps(summary, indent=2))
        return 0

    output_dir = args.project.resolve() / "wiki" / "frankbrain"
    output_dir.mkdir(parents=True, exist_ok=True)
    updated_files = 0
    for row, item in zip(rows, projected, strict=True):
        destination = output_dir / item["path"]
        updated_files += _write_text_if_changed(destination, _render_page(row))

    manifest = {
        **summary,
        "generated_at": datetime.now(UTC).isoformat(),
        "display_name": "FrankBrain",
        "source_backend": "gbrain",
        "direction": "gbrain-to-frankbrain",
        "updated_files": updated_files,
        "unchanged_files": len(projected) - updated_files,
        "files": projected,
    }
    _write_text_if_changed(
        output_dir / "projection-manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps({**summary, "updated_files": updated_files}, indent=2))
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.seed_limit > args.limit:
        raise SystemExit("--seed-limit cannot exceed --limit")
    if not args.write:
        return _run(args)

    lock_path = args.project.resolve() / ".frankbrain-refresh.lock"
    try:
        with _exclusive_lock(lock_path):
            return _run(args)
    except RefreshAlreadyRunning:
        print(
            json.dumps(
                {
                    "mode": "skipped",
                    "project": str(args.project),
                    "reason": "refresh-already-running",
                },
                indent=2,
            )
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
