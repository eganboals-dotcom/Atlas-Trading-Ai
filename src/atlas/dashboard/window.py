"""Native leaderboard window (read-only): a dark, live trading screen in its own app window.

It renders ``window.html`` inside an Edge WebView2 window via pywebview. There is no web
server and no localhost page: the page calls ``Api.live()`` through pywebview's
JavaScript bridge, and that method only reads the session's ``live.json``. A second tab
shows the investing bots through ``Api.investing()``, which reads the nightly summary's
``bots.json`` and marks it to the latest trades from Alpaca's read-only market-data endpoint.
The window has no controls that affect trading.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from atlas.dashboard import data
from atlas.feeds import alpaca

HTML = Path(__file__).with_name("window.html")
LIVE_EVERY_S = 30  # one latest-trades request for every symbol the investing bots hold, beside the paper session's polling


class Api:
    """The only object exposed to the page. Read-only by construction."""

    def __init__(self, session_dir: Path, fetch: alpaca.Fetch | None = None) -> None:
        self._dir = session_dir
        self._fetch = fetch  # tests pass a fake; the window reads .env on first use
        self._asked = -LIVE_EVERY_S
        self._live: dict[str, Any] = {}

    def live(self) -> dict[str, Any] | None:
        live = data.load_live(self._dir)
        if not live:
            return None
        age = (datetime.now(UTC) - datetime.fromisoformat(live["updated_wall"])).total_seconds()
        return {"live": live, "rows": data.leaderboard(live), "age": age}

    def investing(self) -> dict[str, Any] | None:
        """The investing bots, which the nightly summary writes next to the paper sessions' artifacts, marked to the
        latest trades of what they hold."""
        try:
            d = data.load_json(self._dir.parent.parent / "reports" / "summary" / "bots.json")
        except (json.JSONDecodeError, OSError):  # mid-write; the next refresh succeeds
            return None
        return mark(d, self._prices(sorted(d.get("closes", {})))) if d else None

    def _prices(self, symbols: list[str]) -> dict[str, Any]:
        """Latest IEX trades through the read-only market-data module, at most one request every LIVE_EVERY_S."""
        now = time.monotonic()
        if symbols and now - self._asked >= LIVE_EVERY_S:
            self._asked = now
            try:
                fetch = self._fetch or alpaca.http_fetch(alpaca.load_credentials(self._dir.parent.parent.parent / ".env"))
                trades = fetch(alpaca.TRADES_PATH, {"symbols": ",".join(symbols), "feed": "iex"}).get("trades") or {}
                self._live = {"trades": {s: {"p": float(t["p"]), "t": t["t"]} for s, t in trades.items() if t.get("p")}}
            except (OSError, KeyError, TypeError, ValueError) as e:  # no keys, offline or a bad reply: shown on the tab
                self._live = {"error": str(e)}
        return self._live


def mark(d: dict[str, Any], live: dict[str, Any]) -> dict[str, Any]:
    """Each account at the latest trades from a day after its last close, or at that close when there are none.
    shortcut: uses the book's weights, not their drift since the last rebalance; good enough for a day's estimate."""
    closes = d.get("closes", {})
    fresh = {s: q["p"] / closes[s] - 1 for s, q in live.get("trades", {}).items() if s in closes and q["t"][:10] > d["prices_through"]}
    bots = [{**b, "moves": {s: fresh[s] for s in b.get("weights", {}) if s in fresh},
             "now": b["value"] * (1 + sum(w * fresh.get(s, 0.0) for s, w in b.get("weights", {}).items()))} for b in d["bots"]]
    times = [q["t"] for s, q in live.get("trades", {}).items() if s in fresh]
    return {**d, "bots": bots, "live": bool(fresh), "live_at": max(times, default=None), "live_error": live.get("error")}


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
