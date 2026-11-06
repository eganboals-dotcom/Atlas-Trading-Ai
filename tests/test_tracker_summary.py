"""The one-page summary: SPY over a rule's own days, reading each tracker's table, and flagging what needs a person."""

from __future__ import annotations

import json
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


def test_the_window_gets_each_bot_with_its_backtest_and_spy() -> None:
    d = json.loads(s.bots_json([s.bot("Earnings quality", TABLE, SPY), s.bot("ETF rotation", "# nothing", SPY)], SPY, date(2026, 11, 5)))
    assert (d["prices_through"], d["start"], d["spy_backtest"], d["days"]) == ("2026-11-04", 100_000, 0.166, 2)
    assert [b["name"] for b in d["bots"]] == ["Earnings quality", "ETF rotation", "SPY"]  # best account first, then the market
    assert d["bots"][0]["since"] == "2026-11-03" and d["bots"][0]["backtest"]["per_year"] == 0.231
    assert d["bots"][1]["since"] is None and d["bots"][1]["backtest"]["per_year"] == 0.171  # v34, an estimate run after the freeze
    assert d["bots"][2]["value"] == pytest.approx(99_000) and d["bots"][2]["since"] == "2026-11-03"  # same money, same first day


def test_history_keeps_every_day_and_adds_the_newest() -> None:
    old = {"2026-11-02": {"Earnings quality": 100_000, "SPY": 100_000}, "2026-11-03": {"Earnings quality": 99_000, "SPY": 110_000}}
    d = json.loads(s.bots_json([s.bot("Earnings quality", TABLE, SPY)], SPY, date(2026, 11, 5), old))
    assert list(d["history"]) == ["2026-11-02", "2026-11-03", "2026-11-04"]
    assert d["history"]["2026-11-03"]["SPY"] == 110_000  # an old day is kept as it was
    assert d["history"]["2026-11-04"] == {"Earnings quality": pytest.approx(101_500), "SPY": pytest.approx(99_000)}


def test_weights_come_from_the_mix_or_the_equal_list() -> None:
    assert s.weights("Combined weights: F 35.0%, GLD 35.0%\n") == {"F": 0.35, "GLD": 0.35}
    assert s.weights("Book 1, from filings through 2026-10-30, 187 eligible, 2 held:\n\nABBV, ADBE\n") == {"ABBV": 0.5, "ADBE": 0.5}
    assert s.weights("# nothing yet") == {}


def test_a_bot_with_no_days_still_has_its_money() -> None:
    text = "# Rotation\n\nBook 1, from the close of 2026-10-30: groups real, invested 70%. Replaces sleeve 1 of 4.\n\nCombined weights: F 35.0%, GLD 35.0%\n"
    b = s.bot("Rotation", text, SPY)
    assert (b["value"], b["spy"], b["invested"], b["since"]) == (100_000.0, 100_000.0, "70%", None)
    assert b["holds"] == "F 35.0%, GLD 35.0%"
    assert "not yet" in s.bot_lines([b])[6]


def test_invested_counts_every_sleeve_not_just_the_newest() -> None:
    text = "Book 2, from the close of 2026-11-06: groups growth, invested 100%. Replaces sleeve 2 of 4.\n\nCombined weights: A 40.0%, B 37.5%\n"
    assert s.bot("Rotation", text, SPY)["invested"] == "78%"  # one sleeve at 100%, three still at 70%


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


def test_the_fomc_figures_are_not_flagged(tmp_path, monkeypatch) -> None:
    (tmp_path / "fomc_track").mkdir()
    (tmp_path / "fomc_track" / "tracker.md").write_text("- Events: 1, mean +5.0 bps, up on 100%\n\n- Other windows: 2, mean +67.3 bps\n\n- Next decision: 2026-12-09\n\n- bars missing for SPY\n", encoding="utf-8")
    monkeypatch.setattr(s, "REPORTS", tmp_path)
    assert s.report_notes() == ["fomc_track: bars missing for SPY"]


def test_a_rule_can_beat_hold_and_lose_to_spy() -> None:
    rows = [dict(tracker="x", rule="r", days=2, first=date(2026, 11, 3), last=date(2026, 11, 3), ret=0.05, hold=0.01)]
    text = s.build(rows, SPY, [], [], [], date(2026, 11, 5))
    assert "| x | r | +5.00% | +4.00% | -5.00% | 2026-11-03 |" in text
    assert "Nothing." not in s.build(rows, SPY, ["Last run: Traceback"], [], [], date(2026, 11, 5))
    assert "Only 2 days so far" in text
