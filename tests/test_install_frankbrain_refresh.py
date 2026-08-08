from pathlib import Path

from server.scripts.install_frankbrain_refresh import _render_plist


def test_render_plist_resolves_founder_local_paths() -> None:
    rendered = _render_plist(
        "<string>SYNC_SCRIPT</string><string>/Users/YOUR_USER/Library/Logs/x</string>",
        Path("/Users/founder/Library/Application Support/FrankBrain/bin/sync.py"),
        Path("/Users/founder"),
    )

    assert "SYNC_SCRIPT" not in rendered
    assert "YOUR_USER" not in rendered
    assert "/Users/founder/Library/Application Support/FrankBrain/bin/sync.py" in rendered
    assert "/Users/founder/Library/Logs/x" in rendered


def test_launch_agent_does_not_inherit_agent_credentials() -> None:
    template = (
        "<string>SYNC_SCRIPT</string><string>/Users/YOUR_USER/x</string>"
        "<key>GBRAIN_MCP_TOKEN_CODEX</key><string></string>"
        "<key>GBRAIN_MCP_TOKEN_CLAUDE_CODE</key><string></string>"
    )

    rendered = _render_plist(template, Path("/sync.py"), Path("/Users/founder"))

    assert "<key>GBRAIN_MCP_TOKEN_CODEX</key><string></string>" in rendered
    assert "<key>GBRAIN_MCP_TOKEN_CLAUDE_CODE</key><string></string>" in rendered
