from __future__ import annotations

import ast
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


def test_loaders_read_a_session(tmp_path: Path) -> None:
    s = session(tmp_path)
    s.run()
    paper_dir = s.dir.parent
    assert data.list_sessions(paper_dir) == [s.dir]
    assert data.load_report(s.dir)["mode"] == "paper"
    eq = data.equity_frame(s.dir)
    assert eq.height > 0 and set(eq["account"].unique()) == {"champion", "shadow-1", "coin-flip", "benchmark"}
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
