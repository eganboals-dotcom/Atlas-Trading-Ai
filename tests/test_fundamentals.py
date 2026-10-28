"""SEC fundamentals: annual figures only, nothing used before it was filed, market value from raw prices."""

from __future__ import annotations

import dataclasses
from datetime import date, timedelta

import numpy as np
import pytest

from atlas.agents.genome import sample_genome
from atlas.swing.data import Panel
from atlas.swing.families import FAMILIES, exposures
from atlas.swing.fundamentals import as_of, extract, fundamental_features


def fact(val, end, filed, form="10-K", fp="FY", start=None):
    e = {"val": val, "end": end, "filed": filed, "form": form, "fp": fp}
    if start:
        e["start"] = start
    return e


FACTS = {"facts": {
    "us-gaap": {
        "GrossProfit": {"units": {"USD": [
            fact(100.0, "2019-12-31", "2020-02-20", start="2019-01-01"),
            fact(30.0, "2019-12-31", "2020-02-20", start="2019-10-01"),  # a quarter inside the 10-K: skipped
            fact(25.0, "2020-03-31", "2020-05-01", form="10-Q", fp="Q1", start="2020-01-01"),  # quarterly: skipped
            fact(120.0, "2020-12-31", "2021-02-18", start="2020-01-01"),
            fact(110.0, "2019-12-31", "2021-02-18", start="2019-01-01"),  # prior year shown again in 2021's 10-K
        ]}},
        "Assets": {"units": {"USD": [fact(1000.0, "2019-12-31", "2020-02-20"), fact(1200.0, "2020-12-31", "2021-02-18")]}},
        "StockholdersEquity": {"units": {"USD": [fact(400.0, "2019-12-31", "2020-02-20")]}},
        "NetIncomeLoss": {"units": {"USD": [fact(50.0, "2019-12-31", "2020-02-20", start="2019-01-01")]}},
    },
    "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
        fact(10.0, "2020-01-31", "2020-02-20"), fact(40.0, "2020-07-31", "2020-08-01", form="10-Q", fp="Q2")]}}},
}}


def test_extract_keeps_full_fiscal_years_and_all_share_counts() -> None:
    rows = extract(FACTS)
    gp = rows.filter(rows["item"] == "gross_profit").sort("filed")
    assert gp["val"].to_list() == [100.0, 120.0, 110.0]
    assert rows.filter(rows["item"] == "shares")["val"].to_list() == [10.0, 40.0]


def test_values_count_only_from_the_day_after_filing_and_later_years_win() -> None:
    rows = extract(FACTS)
    days = (date(2020, 2, 20), date(2020, 2, 21), date(2021, 2, 18), date(2021, 2, 19))
    np.testing.assert_array_equal(as_of(rows, "gross_profit", days), [np.nan, 100.0, 100.0, 120.0])
    # the restated prior year (110, filed 2021) never replaces the newer fiscal year (120)
    assert as_of(rows, "gross_profit", (date(2022, 1, 1),))[0] == 120.0


def test_market_value_uses_raw_prices_and_the_latest_share_count(tmp_path) -> None:
    extract(FACTS).write_parquet(tmp_path / "S0.parquet")
    dates = (date(2020, 2, 24), date(2020, 7, 31), date(2020, 8, 3), date(2020, 8, 4))
    adj = np.array([[25.0], [25.0], [25.0], [25.0]])
    raw = np.array([[100.0], [25.0], [25.0], [25.0]])  # a 4:1 split on 7/31; adjusted prices hide it
    p = Panel(dates, ("S0",), adj, adj, adj, adj, adj)
    f = fundamental_features(p, tmp_path, raw)
    np.testing.assert_allclose(f["fund_gpa"][1:, 0], 0.1)
    np.testing.assert_allclose(f["fund_bm"][1:, 0], [400 / 1000, 400 / 1000, 400 / 1000])  # 100x10 before, 25x40 after
    assert np.isnan(f["fund_bm"][0, 0])  # no previous close on the first day


def make_panel(days=600, n=6):
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, (days, n)), axis=0))
    dates, d = [], date(2016, 1, 4)
    while len(dates) < days:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    return Panel(tuple(dates), tuple(f"S{j}" for j in range(n)), close, close, close, close, np.ones_like(close))


def test_fund_rank_holds_the_best_and_sees_nothing_later() -> None:
    p = make_panel()
    rng = np.random.default_rng(1)
    feats = {k: np.repeat(rng.normal(size=(1, 6)), 600, axis=0) for k in ("fund_gpa", "fund_roa", "fund_bm", "fund_ep")}
    q = dataclasses.replace(p, features=feats)
    xi, _ = exposures(q, "fund_rank", {"factor": 2.0, "top_k": 2, "rebalance": 21})
    best = set(np.argsort(-feats["fund_bm"][0])[:2])
    assert set(np.flatnonzero(xi[100])) == best and xi[100].sum() == pytest.approx(6)
    k = 400
    for _ in range(6):
        g = sample_genome(FAMILIES["fund_rank"].space, rng)
        later = {n: v.copy() for n, v in feats.items()}
        for v in later.values():
            v[k + 1:] = rng.normal(size=v[k + 1:].shape)
        a = exposures(q, "fund_rank", g)[0]
        b = exposures(dataclasses.replace(p, features=later), "fund_rank", g)[0]
        np.testing.assert_array_equal(a[:k + 1], b[:k + 1])
