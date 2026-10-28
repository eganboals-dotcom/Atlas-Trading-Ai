"""Swing (multi-day) research: no look-ahead, leg and cost accounting, holdout kept out of reach."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from atlas.agents.genome import sample_genome
from atlas.swing.backtest import run, simulate
from atlas.swing.data import Panel, load_panel, save_daily
from atlas.swing.families import FAMILIES, HOLD, exposures
from atlas.swing.walkforward import SwingConfig, folds, walkforward


def panel(days: int = 600, n: int = 3, seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, (days, n)), axis=0))
    opn = np.empty_like(close)
    opn[0] = close[0]
    opn[1:] = close[:-1] * np.exp(rng.normal(0.0002, 0.006, (days - 1, n)))
    hi, lo = np.maximum(opn, close) * 1.004, np.minimum(opn, close) * 0.996
    dates = tuple(date(2016, 1, 4) + timedelta(days=i) for i in range(days))
    return Panel(dates, tuple(f"S{j}" for j in range(n)), opn, hi, lo, close, np.full_like(close, 1e6))


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_decisions_for_day_t_use_nothing_after_the_open_of_t(family: str) -> None:
    p = panel()
    rng = np.random.default_rng(1)
    k = 400
    for _ in range(5):
        g = sample_genome(FAMILIES[family].space, rng)
        before = exposures(p, family, g)
        bumped = Panel(p.dates, p.symbols, p.open.copy(), p.high.copy(), p.low.copy(), p.close.copy(), p.volume)
        bumped.close[k:] *= 1.5  # the close of day k and everything later
        bumped.open[k + 1:] *= 1.5  # opens after day k
        bumped.high[k + 1:] *= 1.5  # highs and lows after day k (a bracket's exits on day k may use day k's)
        bumped.low[k + 1:] *= 0.7
        after = exposures(bumped, family, g)
        for b, a in zip(before[:2], after[:2]):  # the exposures of day k are set by its open
            np.testing.assert_array_equal(b[:k + 1], a[:k + 1])
        for b, a in zip(before[2:], after[2:]):  # a bracket's exit on day k may depend on day k's own bar
            np.testing.assert_array_equal(b[:k], a[:k])


def one_symbol() -> Panel:
    p = panel(n=1)
    return p


def test_hold_earns_open_of_first_day_to_last_close_less_two_costs() -> None:
    p = one_symbol()
    xi, xo = exposures(p, *HOLD)
    s, e = 300, 420
    r = run(p, xi, xo, s, e, cost_bps=0.0).returns
    assert math.isclose(np.prod(1 + r), p.close[e - 1, 0] / p.open[s, 0], rel_tol=1e-12)
    rc = run(p, xi, xo, s, e, cost_bps=10.0)
    assert rc.trades == 2  # in at the first open, out at the last close
    assert math.isclose(r.sum() - rc.returns.sum(), 2 * 10 / 1e4, rel_tol=1e-9)


def test_overnight_only_earns_the_overnight_legs_and_pays_two_trades_a_night() -> None:
    p = one_symbol()
    xi, xo = exposures(p, "session", {"mode": 1.0, "trend_filter": 0.0, "trend_n": 100})
    s, e = 300, 360
    r = run(p, xi, xo, s, e, cost_bps=0.0).returns
    assert math.isclose(np.prod(1 + r), np.prod(1 + p.overnight[s + 1:e, 0]), rel_tol=1e-12)
    rc = run(p, xi, xo, s, e, cost_bps=3.0)
    assert rc.trades == 2 * (e - s - 1)  # bought at each close but the last, sold at the next open
    assert rc.exposure < 0.6


def test_day_only_never_holds_overnight() -> None:
    p = one_symbol()
    xi, xo = exposures(p, "session", {"mode": 2.0, "trend_filter": 0.0, "trend_n": 100})
    assert xo.max() == 0 and xi[1:].min() == 1
    r = run(p, xi, xo, 300, 330, cost_bps=0.0).returns
    np.testing.assert_allclose(r, p.intraday[300:330, 0])


def test_folds_do_not_overlap_and_respect_the_embargo() -> None:
    cfg = SwingConfig(experiment_name="t", symbols=["S0"], holdout_start=date(2030, 1, 1), warmup_days=50,
                      train_days=100, embargo_days=5, validation_days=40, step_days=40)
    plan = folds(400, cfg)
    assert plan[0] == (50, 150, 155, 195)
    assert all(v0 - t1 == 5 and v1 <= 400 for _, t1, v0, v1 in plan)
    assert all(b[2] == a[2] + 40 for a, b in zip(plan, plan[1:]))


def test_walkforward_never_reads_the_holdout(tmp_path) -> None:
    p = panel(days=700, n=2)
    cut = 600
    rows = {}
    for j, s in enumerate(p.symbols):
        rows[s] = [{"date": d, "open": p.open[i, j], "high": p.high[i, j], "low": p.low[i, j],
                    "close": p.close[i, j] if i < cut else float("nan"), "volume": 1e6} for i, d in enumerate(p.dates)]
    save_daily(rows, tmp_path / "daily", {"source": "test"})
    assert len(load_panel(tmp_path / "daily").dates) == 700
    cfg = SwingConfig(experiment_name="t", dataset_dir=tmp_path / "daily", symbols=list(p.symbols),
                      holdout_start=p.dates[cut], warmup_days=60, train_days=200, embargo_days=5,
                      validation_days=60, step_days=60, population=6, generations=2, bootstrap_samples=50)
    out = walkforward(cfg, tmp_path / "reports")
    import json

    rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert rep["holdout_days"] == 100 and rep["development_days"][1] == str(p.dates[cut - 1])
    vals = [h[sc][k][m] for strat, h in rep["headline"].items() for sc in h for k in (strat, "hold")
            for m in ("ann_return", "sharpe", "max_drawdown")]
    vals += [m["sharpe"] for v in rep["others"].values() for m in v.values()]
    assert all(math.isfinite(v) for v in vals)  # NaN holdout closes never reached a computation
    assert (out / "report.md").read_text(encoding="utf-8").startswith("# Swing walk-forward")


def test_a_bot_that_barely_trades_cannot_beat_an_active_one() -> None:
    from atlas.swing.backtest import Result
    from atlas.swing.walkforward import fitness

    cfg = SwingConfig(experiment_name="t", symbols=["S0"], holdout_start=date(2030, 1, 1))
    rng = np.random.default_rng(3)
    lucky = Result(np.where(rng.random(504) < 0.02, 0.01, 0.0), exposure=0.02, trades=4)  # few trades, all wins
    active = Result(rng.normal(0.0002, 0.01, 504), exposure=0.9, trades=10)
    busy_but_idle = Result(np.zeros(504), exposure=0.1, trades=40)
    assert fitness(lucky, cfg) < fitness(active, cfg)
    assert fitness(lucky, cfg) < fitness(busy_but_idle, cfg) < fitness(active, cfg)  # closer to the gate ranks higher
    assert fitness(active, cfg) == pytest.approx(float(active.returns.mean() / active.returns.std() * np.sqrt(252)))


def test_rotation_puts_the_whole_portfolio_in_its_top_pick() -> None:
    p = panel(n=4)
    g = {"mode": 0.0, "lookback": 60, "skip": 0, "top_k": 1, "rebalance": 1000, "abs_filter": 0.0}
    xi, xo = exposures(p, "rotation", g)
    assert xi.max() == 0 and xo.max() == 0  # only day 0 is a rebalance day, and nothing is ranked yet: cash
    g["rebalance"] = 100
    xi, _ = exposures(p, "rotation", g)
    pc = p.close
    day = 200  # a rebalance day: the pick is the best 60-day return up to the close of day 199
    best = int(np.argmax(pc[day - 1] / pc[day - 61] - 1))
    assert xi[day, best] == 4 and xi[day].sum() == 4  # 4 slots of 1/4 each, all in one symbol
    r = run(p, xi, xi, day, day + 50, cost_bps=0.0).returns  # held through: one symbol's return
    one = run(Panel(p.dates, p.symbols[best:best + 1], *(getattr(p, c)[:, best:best + 1]
                                                         for c in ("open", "high", "low", "close", "volume"))),
              np.ones((600, 1)), np.ones((600, 1)), day, day + 50, cost_bps=0.0).returns
    np.testing.assert_allclose(r, one, atol=1e-12)


def test_leg_select_holds_only_the_leg_that_has_been_paying() -> None:
    p = panel(n=1)
    o, c = np.empty((600, 1)), np.empty((600, 1))
    o[0], c[0] = 100.0, 99.95
    for t in range(1, 600):  # every night +10 bps, every day session -5 bps
        o[t] = c[t - 1] * 1.001
        c[t] = o[t] * 0.9995
    q = Panel(p.dates, p.symbols, o, np.maximum(o, c), np.minimum(o, c), c, p.volume)
    xi, xo = exposures(q, "leg_select", {"lookback": 20, "on_bps": 0.0, "id_bps": 0.0, "rebalance": 5})
    assert xo[100:].min() == 1 and xi[100:].max() == 0


def test_month_position_counts_trading_days() -> None:
    from atlas.swing.families import month_position

    dates = (date(2024, 1, 30), date(2024, 1, 31), date(2024, 2, 1), date(2024, 2, 2), date(2024, 2, 5))
    first, left = month_position(dates)
    assert list(first) == [1, 2, 1, 2, 3] and list(left) == [2, 1, 3, 2, 1]


def flat_panel(o, h, l, c) -> Panel:
    """One symbol with the given daily open/high/low/close."""
    a = [np.asarray(v, dtype=float)[:, None] for v in (o, h, l, c)]
    dates = tuple(date(2020, 1, 1) + timedelta(days=i) for i in range(len(o)))
    return Panel(dates, ("S0",), *a, np.full_like(a[0], 1e6))


BRACKET = {"trend_filter": 0.0, "trend_n": 200, "entry": 0.0, "drop_bps": 100.0, "tp_bps": 100.0, "sl_bps": 100.0,
           "max_days": 1}


def test_day_trade_bracket_exits_at_target_stop_or_close() -> None:
    # day 0 (no prior close), 1: target hit, 2: stop hit, 3: both, closed up, 4: both, closed down, 5: neither
    q = flat_panel([100] * 6, [100, 101.5, 100.2, 102, 102, 100.5], [100, 99.5, 98.5, 98, 98, 99.6],
                   [100, 100.3, 99.0, 101, 99.5, 100.2])
    x = exposures(q, "bracket", BRACKET)
    assert list(x[0][:, 0]) == [0, 1, 1, 1, 1, 1] and x[1].max() == 0  # a day trade never holds overnight
    r = simulate(q, x, 0, 6, cost_bps=0.0).returns
    # both touched: an up day is read as open -> low -> high (stop first), a down day as open -> high -> low
    np.testing.assert_allclose(r, [0, 0.01, -0.01, -0.01, 0.01, 0.002], atol=1e-12)
    rc = simulate(q, x, 0, 6, cost_bps=2.0).returns
    np.testing.assert_allclose(r[1:] - rc[1:], 2 * 2 / 1e4)  # in at the open, out once


def test_multi_day_bracket_holds_and_exits_on_a_gap_through_the_target() -> None:
    g = {**BRACKET, "max_days": 5}
    #   day 0   1: buy at 100   2: quiet, held   3: opens 102 (> 101 target): sold at that open
    q = flat_panel([100, 100, 100.2, 102], [100, 100.5, 100.6, 102.5], [100, 99.6, 99.8, 101.5],
                   [100, 100.1, 100.4, 102])
    xi, xo, ex = exposures(q, "bracket", g)
    # sold at day 3's open, and that slot waits until the next day to buy again
    assert list(xi[:, 0]) == [0, 1, 1, 0] and list(xo[:, 0]) == [0, 1, 1, 0]
    assert np.isnan(ex).all()  # no level touched during a day
    r = simulate(q, (xi, xo, ex), 0, 4, cost_bps=0.0).returns
    assert math.isclose(np.prod(1 + r), 102 / 100, rel_tol=1e-12)  # the gap above the target is kept


def test_idle_cash_earns_the_cash_return_and_hold_earns_none_of_it() -> None:
    p = one_symbol()
    cash = np.full(len(p.dates), 0.0001)
    flat = (np.zeros((600, 1)), np.zeros((600, 1)))
    np.testing.assert_allclose(simulate(p, flat, 300, 320, 0.0, cash).returns, 0.0001)
    hold = exposures(p, *HOLD)
    a, b = simulate(p, hold, 300, 320, 0.0, cash).returns, simulate(p, hold, 300, 320, 0.0).returns
    np.testing.assert_allclose(a[1:-1], b[1:-1])  # fully invested days earn no cash


def test_risk_rotation_caps_volatility_and_goes_to_cash_below_trend() -> None:
    p = panel(n=4)
    g = {"lookback": 63, "skip": 0, "top_k": 2, "rebalance": 21, "abs_filter": 0.0, "trend_filter": 0.0,
         "trend_n": 100, "vol_scale": 1.0, "vol_n": 20, "target_vol": 0.05, "overlay_every": 5.0}
    xi, xo = exposures(p, "risk_rotation", g)
    full = exposures(p, "rotation", {**g, "mode": 0.0})[0]
    assert (xi.sum(axis=1) <= full.sum(axis=1) + 1e-12).all() and xi[300:].sum(axis=1).max() < 4
    down = Panel(p.dates, p.symbols, *(getattr(p, c)[::-1].copy() for c in ("open", "high", "low", "close")),
                 p.volume)
    g2 = {**g, "trend_filter": 1.0, "vol_scale": 0.0, "overlay_every": 1.0}
    xi2, _ = exposures(down, "risk_rotation", g2)
    pc = np.vstack([np.full((1, 4), np.nan), down.close[:-1]])  # closes up to t-1
    below = np.array([pc[t] <= pc[t - 99:t + 1].mean(axis=0) for t in range(300, 600)])
    assert below.any() and (xi2[300:][below] == 0).all()


def test_event_drift_enters_the_day_after_a_high_volume_gap() -> None:
    p = one_symbol()
    o, c, v = p.open.copy(), p.close.copy(), np.full_like(p.close, 1e6)
    o[400] = c[399] * 1.05  # day 400 gaps up 5% on 3x volume
    v[400] = 3e6
    q = Panel(p.dates, p.symbols, o, np.maximum(o, c), np.minimum(o, c), c, v)
    g = {"mode": 0.0, "trend_filter": 0.0, "trend_n": 200, "direction": 1.0, "gap_bps": 300.0, "vol_mult": 2.0,
         "confirm": 0.0, "hold_days": 3}
    xi, _ = exposures(q, "event_drift", g)
    assert xi[400, 0] == 0 and list(xi[401:404, 0]) == [1, 1, 1] and xi[404, 0] == 0


def test_news_is_bucketed_by_what_is_known_at_each_open() -> None:
    from atlas.swing.news import bucket
    from atlas.timeutils import MARKET_TZ, datetime_to_ns
    from datetime import datetime

    dates = (date(2024, 3, 7), date(2024, 3, 8), date(2024, 3, 11))  # Thu, Fri, Mon

    def at(d, hh, mm):
        return bucket(datetime_to_ns(datetime(d.year, d.month, d.day, hh, mm, tzinfo=MARKET_TZ)), dates)

    assert at(date(2024, 3, 7), 8, 0) == (0, "pre")
    assert at(date(2024, 3, 7), 9, 20) == (0, "day")  # inside the 15-minute margin: too late for that open
    assert at(date(2024, 3, 7), 15, 59) == (0, "day")
    assert at(date(2024, 3, 7), 16, 5) == (1, "pre")  # after the close: before Friday's open
    assert at(date(2024, 3, 9), 12, 0) == (2, "pre")  # Saturday: before Monday's open
    assert at(date(2024, 3, 11), 17, 0) == (3, "pre")  # past the panel's last day: dropped


def test_earnings_headlines_are_recognized() -> None:
    from atlas.swing.news import EARNINGS

    for h in ("Apple Q3 EPS $1.40 Beats $1.35 Estimate", "Microsoft Reports Fourth-Quarter Results",
              "Intel misses estimates, shares fall", "JPMorgan earnings preview"):
        assert EARNINGS.search(h), h
    for h in ("Apple unveils new iPhone", "Disney names new CEO"):
        assert not EARNINGS.search(h), h


@pytest.mark.parametrize("family", ["news_gap", "earnings_drift", "news_tone"])
def test_news_bots_use_only_news_published_before_the_open(family: str) -> None:
    import dataclasses

    from atlas.swing.news import FEATURES

    p = panel(n=3)
    rng = np.random.default_rng(5)
    feats = {k: (rng.random(p.close.shape) < 0.3).astype(float) * rng.integers(1, 4, p.close.shape)
             for k in FEATURES}
    p = dataclasses.replace(p, features=feats)
    k = 400
    traded = False
    for _ in range(8):
        g = sample_genome(FAMILIES[family].space, rng)
        if "gap_bps" in g:
            g["gap_bps"] = 50.0  # small, so the bot trades often enough to test
        before = exposures(p, family, g)
        later = {n: v.copy() for n, v in feats.items()}
        for n in FEATURES:
            if n.endswith("_day"):
                later[n][k:] = 9 - later[n][k:]  # news during day k and after
            else:
                later[n][k + 1:] = 9 - later[n][k + 1:]  # news after the open of day k
        after = exposures(dataclasses.replace(p, features=later), family, g)
        for b, a in zip(before, after):
            np.testing.assert_array_equal(b[:k + 1], a[:k + 1])
        traded |= before[0].sum() > 0
    assert traded


def test_news_gap_follows_news_and_fades_silence() -> None:
    import dataclasses

    p = one_symbol()
    o = p.open.copy()
    o[300], o[350] = p.close[299] * 1.05, p.close[349] * 0.95  # +5% gap with news, -5% gap without
    pre = np.zeros_like(o)
    pre[300] = 2
    q = dataclasses.replace(Panel(p.dates, p.symbols, o, np.maximum(o, p.close), np.minimum(o, p.close), p.close,
                                  p.volume), features={"news_pre": pre})
    g = {"trend_filter": 0.0, "trend_n": 200, "rule": 2.0, "gap_bps": 400.0, "min_articles": 1, "hold_days": 2}
    xi, xo = exposures(q, "news_gap", g)
    assert list(xi[300:302, 0]) == [1, 1] and list(xi[350:352, 0]) == [1, 1]
    follow_only, _ = exposures(q, "news_gap", {**g, "rule": 0.0})
    assert follow_only[300, 0] == 1 and follow_only[350, 0] == 0


def test_sitting_in_cash_earns_interest_but_no_sharpe() -> None:
    from atlas.swing.backtest import summary

    rng = np.random.default_rng(2)
    cash = 0.0001 + rng.normal(0, 1e-6, 500)  # ~2.5% a year, almost no volatility
    s = summary(cash, cash)
    assert s["ann_return"] > 0.02 and s["sharpe"] == pytest.approx(0.0, abs=1e-9)
    assert summary(cash)["sharpe"] > 50  # what a raw Sharpe would have claimed


def test_news_tags_find_rating_changes_and_earnings_surprises() -> None:
    from atlas.swing.news import TAGS

    cases = {"Barclays Upgrades IBM to Overweight": {"up"}, "Needham Downgrades Meta Platforms to Underperform": {"down"},
             "Microsoft Q4 EPS $1.46 Beats $1.37 Estimate": {"beat"},
             "Intel Q2 EPS $(0.04) Misses $0.10 Estimate": {"miss"},
             "Verizon Q3 Adj. EPS $1.21 Beats $1.19 Estimate, Sales $33.8B Miss $34.282B Estimate": {"beat"},
             "Apple unveils new iPhone": set()}
    for h, want in cases.items():
        assert {t for t, rx in TAGS.items() if rx.search(h)} == want, h
