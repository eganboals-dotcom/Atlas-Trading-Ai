"""The one-page summary: SPY over a rule's own days, reading each tracker's table, and flagging what needs a person."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "trackers"))
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


def test_a_bot_grows_with_its_tracker_and_is_set_against_spy() -> None:
    b = s.bot("Earnings quality", TABLE, SPY)
    assert b["value"] == pytest.approx(101_500) and b["spy"] == pytest.approx(99_000) and b["invested"] == "100%" and b["since"] == date(2026, 11, 3)
    assert "| Earnings quality | $101,500 | $99,000 | +2,500 | 100% | 2026-11-03 |" in s.bot_lines([b])


def test_a_bot_with_no_days_still_has_its_money() -> None:
    text = "# Rotation\n\nBook 1, from the close of 2026-10-30: groups real, invested 70%. Replaces sleeve 1 of 4.\n\nCombined weights: F 35.0%, GLD 35.0%\n"
    b = s.bot("Rotation", text, SPY)
    assert (b["value"], b["spy"], b["invested"], b["since"]) == (100_000.0, 100_000.0, "70%", None)
    assert b["holds"] == "F 35.0%, GLD 35.0%"
    assert "not yet" in s.bot_lines([b])[6]


def test_a_quality_bot_lists_its_companies() -> None:
    assert s.holdings("Book 1, from filings through 2026-10-30, 187 eligible, 37 held:\n\nABBV, ADBE\n") == "equal weights: ABBV, ADBE"
    assert s.holdings("# nothing yet") == "nothing yet"


def bars(closes: list[float], volumes: list[float]) -> pl.DataFrame:
    return pl.DataFrame(dict(date=[date(2026, 11, 2 + i) for i in range(len(closes))], close=closes, volume=volumes))


def test_a_spin_off_shown_as_a_crash_is_flagged() -> None:
    assert "RTX moved -71% on 2026-11-03" in s.suspect_moves(bars([148.0, 43.0], [1e6, 3.2e6]), "RTX", date(2026, 10, 27))[0]


def test_a_real_crash_trades_heavily_and_isnt_flagged() -> None:
    assert s.suspect_moves(bars([100.0, 40.0], [1e6, 2e7]), "X", date(2026, 10, 27)) == []


def test_a_small_move_or_one_before_the_trackers_isnt_flagged() -> None:
    assert s.suspect_moves(bars([100.0, 70.0], [1e6, 1e6]), "X", date(2026, 10, 27)) == []
    assert s.suspect_moves(bars([148.0, 43.0], [1e6, 3.2e6]), "X", date(2026, 11, 3)) == []


def test_a_rule_can_beat_hold_and_lose_to_spy() -> None:
    rows = [dict(tracker="x", rule="r", days=2, first=date(2026, 11, 3), last=date(2026, 11, 3), ret=0.05, hold=0.01)]
    text = s.build(rows, SPY, [], [], [], date(2026, 11, 5))
    assert "| x | r | +5.00% | +4.00% | -5.00% | 2026-11-03 |" in text
    assert "Nothing." not in s.build(rows, SPY, ["Last run: Traceback"], [], [], date(2026, 11, 5))
    assert "Only 2 days so far" in text
