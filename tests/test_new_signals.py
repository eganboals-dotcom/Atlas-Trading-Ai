"""Earnings surprises, insider buying and short interest: each counts only once it was public."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from atlas.swing.data import Panel
from atlas.swing.fundamentals import SCHEMA, earnings_surprise, extract
from atlas.swing.insiders import insider_features
from atlas.swing.shorts import PUBLISH_LAG_DAYS, parse, short_features


def days(start: date, n: int) -> tuple[date, ...]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return tuple(out)


def panel(dates: tuple[date, ...], syms: tuple[str, ...]) -> Panel:
    c = np.full((len(dates), len(syms)), 100.0)
    return Panel(dates, syms, c, c, c, c, np.full_like(c, 1e6))


def test_quarterly_eps_is_extracted_from_single_quarters_only() -> None:
    e = lambda v, s, en, f: {"val": v, "start": s, "end": en, "filed": f, "form": "10-Q", "fp": "Q1"}  # noqa: E731
    facts = {"facts": {"us-gaap": {"EarningsPerShareDiluted": {"units": {"USD/shares": [
        e(1.0, "2020-01-01", "2020-03-31", "2020-05-01"), e(2.5, "2020-01-01", "2020-06-30", "2020-08-01")]}}}}}
    rows = extract(facts).filter(pl.col("item") == "eps_q")
    assert rows["val"].to_list() == [1.0]  # the six-month figure is not a quarter


def test_earnings_surprise_counts_from_the_day_after_the_filing_for_one_quarter() -> None:
    rows = []
    for k in range(12):  # quarter ends every ~91 days from 2017; EPS grows 0.1 a quarter, with noise
        end = date(2017, 3, 31) + timedelta(days=91 * k)
        filed = end + timedelta(days=35)
        eps = 1.0 + 0.1 * k + (0.05 if k % 2 else -0.05)
        rows.append(("eps_q", end, eps, filed))
        if k >= 4:  # each filing repeats last year's quarter (restated by 10% for a split, say)
            rows.append(("eps_q", end - timedelta(days=364), rows[-5 if k > 4 else 0][2], filed))
    last_end = date(2017, 3, 31) + timedelta(days=91 * 11)
    # a big positive surprise in the last quarter
    rows.append(("eps_q", last_end + timedelta(days=91), 5.0, last_end + timedelta(days=91 + 35)))
    df = pl.DataFrame(rows, schema=SCHEMA, orient="row")
    filed = last_end + timedelta(days=91 + 35)
    ds = (filed, filed + timedelta(days=1), filed + timedelta(days=60), filed + timedelta(days=100))
    sue = earnings_surprise(df, ds)
    assert np.isnan(sue[0]) or sue[0] < 5  # on the filing day itself the new surprise is not yet known
    assert sue[1] > 5 and sue[2] == sue[1] and np.isnan(sue[3])  # known next day, for one quarter only


def test_insider_clusters_need_different_insiders_and_count_after_filing() -> None:
    ds = days(date(2020, 1, 1), 200)
    p = panel(ds, ("AAA", "BBB"))
    buys = pl.DataFrame({"issuer_cik": [1, 1, 1, 2], "owner_cik": [10, 10, 11, 20],
                         "filed": [date(2020, 3, 2), date(2020, 3, 3), date(2020, 3, 4), date(2020, 3, 2)],
                         "trans_date": [date(2020, 2, 28)] * 4, "value": [1e5] * 4})
    n = insider_features(p, buys, {"AAA": [1], "BBB": [2]})["ins_buyers"]
    i = {d: k for k, d in enumerate(ds)}
    assert n[i[date(2020, 3, 4)], 0] == 1  # two filings by the same insider, the second insider not yet public
    assert n[i[date(2020, 3, 5)], 0] == 2 and n[i[date(2020, 3, 5)], 1] == 1
    assert n[i[date(2020, 6, 4)], 0] == 0  # more than 90 days later


def test_short_interest_counts_only_after_publication_and_goes_stale() -> None:
    text = ("accountingYearMonthNumber|symbolCode|issueName|issuerServicesGroupExchangeCode|marketClassCode|"
            "currentShortPositionQuantity|previousShortPositionQuantity|stockSplitFlag|averageDailyVolumeQuantity|"
            "daysToCoverQuantity|revisionFlag|changePercent|changePreviousNumber|settlementDate\n"
            "20220615|AAA|A Corp|A|NYSE|1000|900||500|2.00||0|0|2022-06-15\n"
            "20220615|BRK/B|Berkshire|A|NYSE|10|9||10|1.00||0|0|2022-06-15\n"
            "20220615|BFB|Brown-Forman B|A|NYSE|10|9||10|3.00||0|0|2022-06-15\n")
    t = parse(text.encode())
    assert set(t["symbol"]) == {"AAA", "BRK.B", "BFB"}
    ds = days(date(2022, 6, 1), 80)
    out = short_features(panel(ds, ("AAA", "BRK.B", "ZZZ", "BF.B")), t)["short_dtc"]
    assert np.nanmax(out[:, 1]) == 1.0 and np.nanmax(out[:, 3]) == 3.0  # either spelling of a share class
    i = {d: k for k, d in enumerate(ds)}
    avail = date(2022, 6, 15) + timedelta(days=PUBLISH_LAG_DAYS)
    nxt = min(d for d in ds if d > avail)
    assert np.isnan(out[i[min(d for d in ds if d >= avail)] - 1, 0]) and out[i[nxt], 0] == 2.0
    assert np.isnan(out[:, 2]).all()  # never reported
    assert np.isnan(out[-1, 0])  # stale by mid-September


def test_headline_eps_surprises_are_parsed_with_signs_and_only_with_both_numbers() -> None:
    from atlas.swing.news import parse_surprise

    assert parse_surprise("Wells Fargo Q4 EPS $1.21 Beats $1.19 Estimate, Sales $20.98B Miss $21.75B Estimate") == (1.21, 1.19)
    assert parse_surprise("Ford Q2 Adj. EPS $(0.05) Misses $0.02 Estimate") == (-0.05, 0.02)
    assert parse_surprise("Uber Q1 EPS $-0.20 Beats $-0.25 Estimate") == (-0.2, -0.25)
    assert parse_surprise("AT&T Reports Q4 Adj. EPS $0.63, Inline, Sales $42.1B vs $42.8B Est.") == (0.63, 0.63)
    for head in ("JPMorgan Chase Q4 GAAP EPS $1.98 May Not Compare To $2.25 Estimate",
                 "Bank Of America Lower After Q4 EPS Beat", "Baird: Gilead Set For Q1 EPS Miss"):
        assert parse_surprise(head) is None


def test_headline_surprise_counts_from_the_next_open_and_only_for_one_named_company() -> None:
    from datetime import datetime

    from atlas.swing.news import SURPRISE_FRESH_DAYS, surprise_features
    from atlas.timeutils import MARKET_TZ

    ds = days(date(2023, 1, 2), 120)
    p = panel(ds, ("AAA", "BBB"))
    price = np.full(p.close.shape, 50.0)

    def ns(d: date, h: int, m: int = 0) -> int:
        return int(datetime(d.year, d.month, d.day, h, m, tzinfo=MARKET_TZ).timestamp() * 1e9)

    arts = pl.DataFrame({"created_ns": [ns(ds[10], 7), ns(ds[20], 16, 5), ns(ds[30], 7), ns(ds[40], 11)],
                         "symbols": [["AAA"], ["BBB"], ["AAA", "BBB"], ["BBB", "ZZZ"]],
                         "headline": ["A Q1 EPS $1.50 Beats $1.00 Estimate", "B Q1 EPS $0.80 Misses $1.30 Estimate",
                                      "A and B Q1 EPS $9.00 Beats $1.00 Estimate", "B Q2 EPS $1.10 Beats $1.00 Estimate"]})
    s = surprise_features(p, arts, price)["news_surprise"]
    assert np.isnan(s[9, 0]) and s[10, 0] == 0.5 / 50  # before the open: usable at that open
    assert np.isnan(s[20, 1]) and s[21, 1] == -0.5 / 50  # after the close: from the next trading day's open
    assert np.isnan(s[10 + SURPRISE_FRESH_DAYS, 0])  # stale after the window
    assert s[30, 0] == 0.5 / 50  # a headline naming both companies is ignored
    assert s[40, 1] == -0.5 / 50 and np.isclose(s[41, 1], 0.1 / 50)  # mid-session: from the next open; ZZZ isn't ours
