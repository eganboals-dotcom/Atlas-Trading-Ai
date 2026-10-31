"""The pre-FOMC tracker: its rule stays frozen, only finished sessions count, and a scored event can't change or go."""

from __future__ import annotations

import importlib.util
import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v21_fomc_track", ROOT / "scripts" / "v21_fomc_track.py")
track = importlib.util.module_from_spec(spec)
spec.loader.exec_module(track)


def config(tmp_path: Path) -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "fomc_track_v21.yaml").read_text(encoding="utf-8"))
    cfg["bars"] = str(tmp_path / "bars.parquet")
    cfg["out_dir"] = str(tmp_path / "out")
    return cfg


def prices(rows: list[tuple[date, float]]) -> pl.DataFrame:
    return pl.DataFrame(dict(day=[d for d, _ in rows], o=[p for _, p in rows]), schema={"day": pl.Date, "o": pl.Float64})


def stub_fetch(monkeypatch: pytest.MonkeyPatch, rows: list[tuple[date, float]], seen: list | None = None) -> None:
    def fake(cfg: dict, first: date, last: date) -> pl.DataFrame:
        if seen is not None:
            seen.append((first, last))
        return prices([r for r in rows if first <= r[0] <= last])

    monkeypatch.setattr(track, "fetch_prices", fake)


DAY = date(2026, 12, 9)
ROWS = [(date(2026, 12, 7), 100.0), (date(2026, 12, 8), 100.0), (DAY, 101.0), (date(2026, 12, 10), 90.0), (date(2026, 12, 11), 91.0)]


def test_event_buys_the_session_before_and_skips_the_reaction() -> None:
    rule = {"cost_bps": 0.0}
    events, other = track.score(rule, {DAY}, prices(ROWS))
    assert events == {DAY: pytest.approx(0.01)}
    assert other == [pytest.approx(0.0), pytest.approx(91 / 90 - 1)]  # 12-10's fall after the decision is in neither


def test_costs_come_off_both_sides() -> None:
    events, _ = track.score({"cost_bps": 0.2}, {DAY}, prices(ROWS))
    assert events[DAY] == pytest.approx(101 * (1 - 2e-5) / (100 * (1 + 2e-5)) - 1)


def test_today_is_never_fetched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list = []
    stub_fetch(monkeypatch, ROWS, seen)
    track.run(config(tmp_path), DAY)
    assert seen[0][1] == DAY - timedelta(days=1)
    assert json.loads((tmp_path / "out" / "state.json").read_text())["scored"] == {}


def test_event_is_scored_once_and_keeps_its_score(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = config(tmp_path)
    stub_fetch(monkeypatch, ROWS)
    track.run(cfg, DAY + timedelta(days=1))
    state = tmp_path / "out" / "state.json"
    first = json.loads(state.read_text())["scored"][str(DAY)]
    pl.DataFrame(dict(day=[d for d, _ in ROWS[:3]], o=[1.0, 1.0, 5.0])).write_parquet(cfg["bars"])  # later bars differ
    track.run(cfg, DAY + timedelta(days=1))
    assert json.loads(state.read_text())["scored"][str(DAY)] == first


def test_changing_the_rule_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = config(tmp_path)
    stub_fetch(monkeypatch, [])
    track.run(cfg, date(2026, 11, 1))
    cfg["rule"]["cost_bps"] = 0.0
    with pytest.raises(SystemExit, match="changed"):
        track.run(cfg, date(2026, 11, 1))


def test_removing_a_scored_event_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = config(tmp_path)
    stub_fetch(monkeypatch, ROWS)
    track.run(cfg, DAY + timedelta(days=1))
    cfg["decisions"] = cfg["decisions"][1:]
    with pytest.raises(SystemExit, match="removed"):
        track.run(cfg, DAY + timedelta(days=1))


def test_no_verdict_before_the_review() -> None:
    import numpy as np

    rule = {"review_events": 40, "bootstrap_samples": 100, "seed": 1}
    assert "Not reviewed" in track.verdict(rule, np.full(39, 0.01), [0.0])
    good, bad = np.r_[np.full(30, 0.01), np.full(10, 0.002)], np.r_[np.full(20, 0.01), np.full(20, -0.01)]
    assert "PASS" in track.verdict(rule, good, [0.0])
    assert "FAIL" in track.verdict(rule, bad, [0.0])
    assert "FAIL" in track.verdict(rule, good, [0.05])  # must also beat the other windows
