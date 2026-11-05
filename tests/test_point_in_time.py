"""The list of who really existed: typos and reused tickers can't get in, joins make no return, and later data can't reach a pool."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "research" / "investing"))
import v31_point_in_time as pit  # noqa: E402


def bars(first: date, closes: list[float], volume: float = 1000.0) -> pl.DataFrame:
    days = [first + timedelta(days=i) for i in range(len(closes))]
    return pl.DataFrame(dict(date=days, open=closes, high=closes, low=closes, close=closes, volume=[volume] * len(closes)), schema=pit.SCHEMA)


def filings(rows: list[tuple[int, str | None, date]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={"cik": pl.Int64, "symbol": pl.String, "day": pl.Date}, orient="row")


def test_clean_keeps_tickers_and_refuses_junk() -> None:
    assert pit.clean(" amnb ") == "AMNB"
    assert pit.clean("BRK-B") == "BRK.B"
    assert pit.clean("GOOG, GOOGL") == "GOOG"
    for junk in ("NONE", "N/A", "", None, "TOOLONGNAME"):
        assert pit.clean(junk) is None


def test_a_typo_cannot_start_a_segment() -> None:
    rows = [(1, "AAPL", date(2020, 1, d)) for d in (2, 3, 4)] + [(1, "APPL", date(2020, 1, 5))] + [(1, "AAPL", date(2020, 4, 2))]
    assert [(s["symbol"], s["first"], s["last"]) for s in pit.segments(filings(rows))] == [("AAPL", date(2020, 1, 2), date(2020, 4, 2))]


def test_a_rename_makes_two_segments_and_a_reused_ticker_stays_apart() -> None:
    rows = [(1, "FB", date(2020, 1, 2)), (1, "META", date(2022, 7, 1)), (2, "LINE", date(2016, 2, 1)), (3, "LINE", date(2024, 8, 1))]
    s = pit.segments(filings(rows))
    assert [(x["cik"], x["symbol"]) for x in s] == [(1, "FB"), (1, "META"), (2, "LINE"), (3, "LINE")]


def test_a_new_listing_gets_no_bars_from_before_it_first_filed() -> None:
    company = [dict(cik=3, symbol="LINE", first=date(2024, 7, 25), last=date(2024, 8, 1))]
    old = bars(date(2024, 7, 20), [0.16] * 5 + [72.0] * 5)
    joined = pit.stitch(pit.company_pieces(company, lambda s: old))
    assert joined["date"][0] == date(2024, 7, 25) and joined["close"].min() == 72.0


def test_quiet_insiders_arent_a_stop() -> None:
    company = [dict(cik=1, symbol="XRX", first=date(2016, 1, 4), last=date(2019, 5, 14))]
    joined = pit.stitch(pit.company_pieces(company, lambda s: bars(date(2019, 5, 1), [5.0] * 400)))
    assert joined["date"][-1] == date(2019, 5, 1) + timedelta(days=399)


def test_a_break_in_the_bars_keeps_a_later_user_of_the_ticker_out() -> None:
    company = [dict(cik=1, symbol="SHLD", first=date(2016, 1, 4), last=date(2019, 5, 14))]
    old = bars(date(2019, 5, 1), [5.0] * 30)
    later = bars(date(2019, 5, 30) + timedelta(days=pit.cfg["max_gap_days"] + 1), [90.0] * 30)
    joined = pit.stitch(pit.company_pieces(company, lambda s: pl.concat([old, later])))
    assert joined["date"][-1] == date(2019, 5, 30) and joined["close"].max() == 5.0


def test_a_new_owner_ends_the_old_company_at_the_handover() -> None:
    old = dict(cik=1, symbol="WRK", first=date(2016, 1, 4), last=date(2018, 11, 6))
    new = dict(cik=2, symbol="WRK", first=date(2019, 1, 2), last=date(2024, 7, 1))
    day = pit.taken_on(old, pit.takeovers([old, new])["WRK"])
    assert day == date(2019, 1, 2)
    joined = pit.stitch(pit.company_pieces([old], lambda s: bars(date(2018, 11, 1), [30.0] * 300), day))
    assert joined["date"][-1] == date(2019, 1, 1)


def test_a_typo_by_another_company_doesnt_take_a_ticker() -> None:
    real = dict(cik=1, symbol="BAC", first=date(2016, 1, 4), last=date(2026, 1, 20))
    typo = dict(cik=2, symbol="BAC", first=date(2020, 3, 2), last=date(2020, 3, 2))
    assert pit.taken_on(real, [real, typo]) is None


def test_a_ticker_with_no_bars_hands_its_days_to_the_next() -> None:
    company = [dict(cik=1, symbol="FB", first=date(2016, 1, 7), last=date(2022, 6, 2)), dict(cik=1, symbol="META", first=date(2022, 6, 9), last=date(2022, 6, 20))]
    meta = bars(date(2020, 1, 1), [100.0] * 900)
    joined = pit.stitch(pit.company_pieces(company, lambda s: meta if s == "META" else pl.DataFrame(schema=pit.SCHEMA)))
    assert joined["date"][0] == date(2020, 1, 1)


def test_a_rename_leaves_no_hole_before_the_first_filing_under_the_new_ticker() -> None:
    company = [dict(cik=1, symbol="FB", first=date(2016, 1, 7), last=date(2022, 6, 2)), dict(cik=1, symbol="META", first=date(2022, 7, 7), last=date(2022, 8, 1))]
    old, new = bars(date(2022, 5, 30), [100.0] * 10), bars(date(2022, 5, 30), [100.0] * 60)
    joined = pit.stitch(pit.company_pieces(company, lambda s: old if s == "FB" else new))
    assert joined.height == 60 and joined["date"].diff().drop_nulls().max() == timedelta(days=1)


def test_a_join_makes_no_return() -> None:
    joined = pit.stitch([(bars(date(2020, 1, 1), [10.0, 20.0]), date(2020, 1, 1), date(2020, 1, 2)), (bars(date(2020, 1, 3), [100.0, 110.0]), date(2020, 1, 3), date(2020, 1, 4))])
    assert joined["close"].to_list() == [50.0, 100.0, 100.0, 110.0]


def test_a_bar_with_no_volume_doesnt_count() -> None:
    stale = bars(date(2020, 1, 1), [10.0, 10.0]).with_columns(volume=pl.Series([100.0, 0.0]))
    assert pit.stitch([(stale, date(2020, 1, 1), date(2020, 1, 2))]).height == 1


def test_a_company_that_halved_counts_as_failed_and_one_that_held_as_bought() -> None:
    n = 300
    close = np.full((n, 3), 100.0)
    traded = np.ones((n, 3), dtype=bool)
    traded[200:, 0] = traded[200:, 1] = False
    close[150:, 0] = 40.0  # fell, then stopped
    assert pit.stop_points(close, traded, n - 1) == [(0, 199, True), (1, 199, False)]


def test_stopping_near_the_end_isnt_a_stop() -> None:
    traded = np.ones((300, 1), dtype=bool)
    traded[-3:] = False
    assert pit.stop_points(np.full((300, 1), 100.0), traded, 299) == []


def test_the_pool_uses_no_later_volume_and_needs_a_bar_on_the_last_close() -> None:
    d = np.tile(np.array([3.0, 2.0, 1.0, 0.0]), (200, 1))
    d[100:, 3] = 99.0  # large only later
    d[99, 0] = 0.0  # no bar on the last close
    assert pit.pool_at(d, 100, 2).tolist() == [1, 2]
    assert pit.pool_at(d, 100 + pit.cfg["volume_days"] + 1, 2).tolist()[0] == 3


def test_a_list_needs_a_full_year_of_bars() -> None:
    d = np.tile(np.array([1.0, 5.0]), (400, 1))
    d[:300, 1] = 0.0  # listed too recently
    assert pit.list_pool(d, 399, 2)[0].tolist() == [0]


def test_the_pre_registered_settings_match_the_rules_they_check() -> None:
    assert pit.cfg["quality_cost_bps"] == 2.0 and pit.cfg["every"] == 63 and pit.cfg["top_share"] == 0.20
    assert pit.cfg["rotation_cost_bps"] == pit.v.cfg["cost_bps"]
    assert (pit.cfg["first_scored"], pit.cfg["last_scored"]) == (str(pit.v.cfg["first_scored"]), str(pit.v.cfg["last_scored"]))
