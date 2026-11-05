from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from paper_helpers import REPO, session

from atlas.dashboard import data

FORBIDDEN = ("atlas.paper.broker", "atlas.paper.session", "atlas.paper.risk", "atlas.brokers", "atlas.environment")


def test_dashboard_imports_nothing_that_can_trade() -> None:
    for path in (REPO / "src/atlas/dashboard").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            for m in mods:
                assert not m.startswith(FORBIDDEN), f"{path.name} imports {m}"


def test_investing_tab_marks_each_account_to_the_latest_trades(tmp_path: Path) -> None:
    from atlas.dashboard.window import Api, mark

    d = {"prices_through": "2026-11-04", "closes": {"SPY": 100.0, "TLT": 50.0}, "bots": [
        {"name": "ETF rotation", "value": 100_000.0, "weights": {"TLT": 0.5}},
        {"name": "SPY", "value": 101_000.0, "weights": {"SPY": 1.0}}]}
    live = {"trades": {"SPY": {"p": 98.0, "t": "2026-11-05T16:00:00Z"}, "TLT": {"p": 51.0, "t": "2026-11-04T20:59:00Z"}}}
    m = mark(d, live)
    assert m["live"] and m["live_at"] == "2026-11-05T16:00:00Z"
    assert m["bots"][1]["now"] == pytest.approx(98_980) and m["bots"][1]["moves"] == {"SPY": pytest.approx(-0.02)}
    assert m["bots"][0]["now"] == 100_000  # TLT's trade is from the close's own day: nothing new
    assert mark(d, {"error": "offline"})["live_error"] == "offline" and not mark(d, {})["live"]

    asked = []
    fake = lambda path, params: asked.append((path, params)) or {"trades": {"SPY": {"p": 99.0, "t": "2026-11-05T15:00:00Z"}}}
    summary = tmp_path / "artifacts" / "reports" / "summary"
    summary.mkdir(parents=True)
    (summary / "bots.json").write_text(json.dumps(d), encoding="utf-8")
    api = Api(tmp_path / "artifacts" / "paper" / "live_2026-11-05", fetch=fake)
    assert api.investing()["bots"][1]["now"] == pytest.approx(99_990)
    api.investing()
    assert asked == [("/v2/stocks/trades/latest", {"symbols": "SPY,TLT", "feed": "iex"})]  # one request per LIVE_EVERY_S


def test_loaders_read_a_session(tmp_path: Path) -> None:
    s = session(tmp_path)
    s.run()
    paper_dir = s.dir.parent
    assert data.list_sessions(paper_dir) == [s.dir]
    assert data.load_report(s.dir)["mode"] == "paper"
    eq = data.equity_frame(s.dir)
    assert eq.height > 0 and set(eq["account"].unique()) == {"champion", "shadow-1", "Flip", "Hold"}
    assert data.fills(s.dir) and all(f["simulated"] for f in data.fills(s.dir))
    assert data.list_sessions(tmp_path / "missing") == []


def test_streamlit_app_renders(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    import sys

    from streamlit.testing.v1 import AppTest

    s = session(tmp_path)
    s.run()
    monkeypatch.setattr(sys, "argv", ["app.py", "--paper-dir", str(s.dir.parent),
                                      "--reports-dir", str(tmp_path / "none")])
    at = AppTest.from_file(str(REPO / "src/atlas/dashboard/app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    heads = [h.value for h in at.subheader]
    assert "Leaderboard" in heads and "Accounts" in heads  # live tab (from live.json) and history tab
