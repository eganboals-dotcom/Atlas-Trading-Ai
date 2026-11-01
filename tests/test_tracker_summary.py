"""The one-page summary: SPY over a rule's own days, reading each tracker's table, and flagging what needs a person."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tracker_summary as s  # noqa: E402

SPY = pl.DataFrame(dict(date=[date(2026, 11, 2), date(2026, 11, 3), date(2026, 11, 4)], close=[100.0, 110.0, 99.0]))
TABLE = """# Earnings quality: forward tracker

Days 2026-11-03 to 2026-11-04. Annual figures and Sharpe are shown after 60 days.

| Book | Return | vs Hold | Return/yr | Sharpe | Max DD |
|---|---|---|---|---|---|
| Earnings quality | +1.50% | +0.50% | - | - | 1.0% |
| Hold | +1.00% |  | - | - | 0.5% |
"""


def test_spy_runs_from_the_close_before_the_first_day() -> None:
    assert s.spy_return(SPY, date(2026, 11, 3), date(2026, 11, 3)) == pytest.approx(0.10)
    assert s.spy_return(SPY, date(2026, 11, 3), date(2026, 11, 4)) == pytest.approx(-0.01)
    assert s.spy_return(SPY, date(2026, 11, 2), date(2026, 11, 4)) is None


def test_a_table_row_is_read_with_its_days() -> None:
    assert s.table_row(TABLE, "Earnings quality") == dict(first=date(2026, 11, 3), last=date(2026, 11, 4), ret=pytest.approx(0.015), hold=pytest.approx(0.01))
    assert s.table_row("# nothing scored yet", "Earnings quality") is None


def test_a_negative_return_is_read() -> None:
    assert s.table_row(TABLE.replace("+1.50%", "-2.25%"), "Earnings quality")["ret"] == pytest.approx(-0.0225)


def test_only_the_last_run_counts_as_a_problem() -> None:
    log = "=== Sat 01:30\nTraceback (most recent call last):\n=== Sun 01:30\nall fine\n"
    assert s.run_problems(log) == []
    assert "Traceback" in s.run_problems(log + "=== Mon 01:30\nTraceback (most recent call last):\n")[0]


def test_a_rule_behind_the_market_data_is_flagged() -> None:
    rows = [dict(tracker="u300/u300", last=date(2026, 11, 3))]
    assert s.stale(rows, date(2026, 11, 4)) == ["u300/u300 stops at 2026-11-03, SPY's bars at 2026-11-04"]
    assert s.stale(rows, date(2026, 11, 3)) == []


def test_a_rule_can_beat_hold_and_lose_to_spy() -> None:
    rows = [dict(tracker="x", rule="r", days=2, first=date(2026, 11, 3), last=date(2026, 11, 3), ret=0.05, hold=0.01)]
    text = s.build(rows, SPY, [], [], [], date(2026, 11, 5))
    assert "| x | r | +5.00% | +4.00% | -5.00% | 2026-11-03 |" in text
    assert "Nothing." not in s.build(rows, SPY, ["Last run: Traceback"], [], [], date(2026, 11, 5))
    assert "Only 2 days so far" in text
