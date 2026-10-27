"""Native leaderboard window (read-only): a dark, live trading screen in its own app window.

It renders ``window.html`` inside an Edge WebView2 window via pywebview. There is no web
server and no localhost page: the page calls ``Api.live()`` through pywebview's
JavaScript bridge, and that method only reads the session's ``live.json``. The window
has no controls that affect trading.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from atlas.dashboard import data

HTML = Path(__file__).with_name("window.html")


class Api:
    """The only object exposed to the page. Read-only by construction."""

    def __init__(self, session_dir: Path) -> None:
        self._dir = session_dir

    def live(self) -> dict[str, Any] | None:
        live = data.load_live(self._dir)
        if not live:
            return None
        age = (datetime.now(UTC) - datetime.fromisoformat(live["updated_wall"])).total_seconds()
        return {"live": live, "rows": data.leaderboard(live), "age": age}


def run_window(session_dir: Path) -> int:
    import webview  # optional dependency (the `window` extra)

    webview.create_window(f"Atlas · {session_dir.name}", html=HTML.read_text(encoding="utf-8"),
                          js_api=Api(session_dir), width=1480, height=920, min_size=(1100, 680),
                          background_color="#06080C", text_select=True)
    webview.start()
    return 0


def window_available() -> bool:
    try:
        import webview  # noqa: F401
    except ImportError:
        return False
    return True


def open_app_window(session_dir: Path) -> None:
    """Launch the window as its own process (no console), so it outlives nothing and blocks nothing."""
    cmd = [sys.executable, "-m", "atlas.cli", "paper-window", "--session-dir", str(session_dir)]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    subprocess.Popen(cmd, creationflags=flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
