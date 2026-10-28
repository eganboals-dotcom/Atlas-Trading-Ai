"""Swing (multi-day) research: no look-ahead, leg and cost accounting, holdout kept out of reach."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from atlas.agents.genome import sample_genome
from atlas.swing.backtest import run
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
        after = exposures(bumped, family, g)
        for b, a in zip(before, after):
            np.testing.assert_array_equal(b[:k + 1], a[:k + 1])


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
    vals = [h[k][m] for h in rep["headline"].values() for k in ("champion", "hold")
            for m in ("ann_return", "sharpe", "max_drawdown")]
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
