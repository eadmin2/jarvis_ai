#!/usr/bin/env python3
"""Install the founder-local FrankBrain projection and LaunchAgent files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from .gbrain_llm_wiki_projection import _write_text_if_changed
except ImportError:  # Direct execution adds this script directory to sys.path.
    from gbrain_llm_wiki_projection import _write_text_if_changed


LABEL = "com.frankbrain.refresh"


def _render_plist(template: str, sync_script: Path, home: Path) -> str:
    rendered = template.replace("SYNC_SCRIPT", str(sync_script))
    rendered = rendered.replace("/Users/YOUR_USER", str(home))
    rendered = rendered.replace("YOUR_USER", home.name)
    if "SYNC_SCRIPT" in rendered or "YOUR_USER" in rendered:
        raise ValueError("unresolved FrankBrain LaunchAgent placeholder")
    return rendered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--home",
        type=Path,
        default=Path.home(),
        help="founder home directory; defaults to the current user home",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="write files; default reports the planned destinations",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    source_scripts = [
        repo_root / "server/scripts/gbrain_llm_wiki_projection.py",
        repo_root / "server/scripts/frankbrain_sync.py",
    ]
    template_path = repo_root / "launchd/com.frankbrain.refresh.plist"
    home = args.home.resolve()
    support_dir = home / "Library/Application Support/FrankBrain/bin"
    writeback_dir = support_dir.parent / "writeback"
    installed_scripts = [support_dir / source.name for source in source_scripts]
    installed_sync_script = support_dir / "frankbrain_sync.py"
    installed_plist = home / "Library/LaunchAgents" / f"{LABEL}.plist"
    plan = {
        "mode": "install" if args.install else "dry-run",
        "installed_scripts": [str(path) for path in installed_scripts],
        "writeback_directory": str(writeback_dir),
        "launch_agent": str(installed_plist),
    }
    if not args.install:
        print(json.dumps(plan, indent=2))
        return 0

    scripts_changed = 0
    writeback_dir.mkdir(parents=True, exist_ok=True)
    for source, destination in zip(source_scripts, installed_scripts, strict=True):
        scripts_changed += _write_text_if_changed(
            destination, source.read_text(encoding="utf-8")
        )
        os.chmod(destination, 0o700)
    rendered_plist = _render_plist(
        template_path.read_text(encoding="utf-8"), installed_sync_script, home
    )
    plist_changed = _write_text_if_changed(installed_plist, rendered_plist)
    os.chmod(installed_plist, 0o600)
    print(
        json.dumps(
            {
                **plan,
                "scripts_updated": scripts_changed,
                "launch_agent_updated": plist_changed,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
