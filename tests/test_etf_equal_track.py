"""The equal-ETF tracker: equal at the start and each quarter, drifting in between, costs on trades only, and its rule stays frozen."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "trackers"))
import v36_etf_equal_track as track  # noqa: E402
from atlas.swing.data import Panel  # noqa: E402

RULE = dict(etfs=["A", "B"], first_day="2026-12-30", cost_bps=2.0, review_sessions=1260)
DAYS = (date(2026, 12, 29), date(2026, 12, 30), date(2026, 12, 31), date(2027, 1, 4))


def panel(a: list[float], b: list[float], spy: list[float]) -> Panel:
    close = np.array([a, b, spy], dtype=float).T
    return Panel(dates=DAYS, symbols=("A", "B", "SPY"), open=close, high=close, low=close, close=close, volume=np.ones_like(close))


def test_equal_at_the_start_drifts_and_resets_each_quarter() -> None:
    r, w = track.returns(RULE, panel([100, 110, 121, 121], [100, 100, 100, 100], [100, 101, 102, 103]), np.zeros(4))
    a_share = 0.55 / 1.05  # A's share after its first 10% day
    drifted = a_share * 1.1 / (1 + a_share * 0.1)
    assert r["equal"] == pytest.approx([(1 - 2e-4) * 1.05 - 1, a_share * 0.1, -2 * (drifted - 0.5) * 2e-4])  # buying all costs 2 bps; the reset trades the drift
    assert w == pytest.approx([0.5, 0.5])
    assert r["SPY"] == pytest.approx([0.01, 1 / 101, 1 / 102])


def test_nothing_is_scored_before_the_first_day() -> None:
    r, w = track.returns({**RULE, "first_day": "2027-02-01"}, panel([100] * 4, [100] * 4, [100] * 4), np.zeros(4))
    assert r == {} and w == pytest.approx([0.5, 0.5])


def test_a_changed_rule_is_refused(tmp_path: Path) -> None:
    rule = yaml.safe_load((ROOT / "configs" / "etf_equal_track_v36.yaml").read_text(encoding="utf-8"))["rule"]
    assert len(rule["etfs"]) == 13
    (tmp_path / "state.json").write_text('{"fingerprint": "%s", "frozen": "2026-11-05"}' % track.fingerprint(rule), encoding="utf-8")
    assert track.check_state(rule, tmp_path / "state.json")["frozen"] == "2026-11-05"
    with pytest.raises(SystemExit):
        track.check_state({**rule, "cost_bps": 1.0}, tmp_path / "state.json")
