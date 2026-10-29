"""Tilt bots: always invested like Hold, only the mix leans with a factor's rank, and nothing from the future."""

from __future__ import annotations

import dataclasses
from datetime import date, timedelta

import numpy as np
import pytest

from atlas.swing.data import Panel
from atlas.swing.families import HOLD, exposures, tilt, tilt_features


def make_panel(days: int = 700, n: int = 8, seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    vol = np.linspace(0.005, 0.03, n)  # symbol 0 the steadiest
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 1, (days, n)) * vol, axis=0))
    close[:300, -1] = np.nan  # the last symbol lists later
    opn = close * np.exp(rng.normal(0, 0.002, close.shape))
    dates, d = [], date(2016, 1, 4)
    while len(dates) < days:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    return Panel(tuple(dates), tuple(f"S{j}" for j in range(n)), opn, np.fmax(opn, close), np.fmin(opn, close), close,
                 np.full_like(close, 1e6))


@pytest.mark.parametrize("factor", [0.0, 1.0, 2.0, 3.0])
def test_tilt_is_invested_exactly_like_hold(factor: float) -> None:
    p = make_panel()
    gpa = np.tile(np.linspace(0.1, 0.8, 8), (len(p.dates), 1))
    p = dataclasses.replace(p, features={"fund_gpa": gpa})
    x = tilt(p, {"factor": factor, "tilt": 0.7, "rebalance": 21})
    hold = exposures(p, *HOLD)[1]
    assert np.allclose(x.sum(axis=1)[21:], hold.sum(axis=1)[21:])  # same capital in the market every day
    assert (x >= 0).all() and np.allclose(x[350:].max(axis=1) / x[350:].min(axis=1).clip(1e-9), 1.7 / 0.3, rtol=0.3)
    assert np.allclose(tilt(p, {"factor": factor, "tilt": 0.0, "rebalance": 21})[21:], hold[21:])  # no tilt = Hold


def test_tilt_leans_toward_the_best_ranked() -> None:
    p = make_panel()
    gpa = np.tile(np.linspace(0.1, 0.8, 8), (len(p.dates), 1))  # symbol 7 the most profitable
    p = dataclasses.replace(p, features={"fund_gpa": gpa})
    q = tilt(p, {"factor": 0.0, "tilt": 0.5, "rebalance": 63})[-1]
    assert np.argmax(q) == 7 and np.argmin(q) == 0
    lv = tilt(p, {"factor": 2.0, "tilt": 0.5, "rebalance": 63})[-1]
    assert np.argmax(lv) == 0  # the steadiest symbol gets the most weight


def test_tilt_never_sees_the_future() -> None:
    p = make_panel()
    t = 500
    q = dataclasses.replace(p, close=p.close.copy(), open=p.open.copy())
    q.close[t:] *= np.linspace(0.5, 2.0, 8)  # rewrite everything from day t on
    q.open[t:] *= np.linspace(0.5, 2.0, 8)
    for name in ("tilt_momentum", "tilt_lowvol"):
        a, b = tilt_features(p)[name], tilt_features(q)[name]
        assert np.array_equal(a[: t + 1], b[: t + 1], equal_nan=True)  # day t uses closes up to t-1 only
    g = {"factor": 3.0, "tilt": 0.5, "rebalance": 10}
    assert np.allclose(tilt(p, g)[: t + 1], tilt(q, g)[: t + 1])


def test_combined_tilt_averages_the_ranks_it_has() -> None:
    p = make_panel()
    n = len(p.dates)
    q = np.tile(np.linspace(0.0, 1.0, 8), (n, 1))  # quality ranks: symbol 7 best
    e = np.tile(np.linspace(1.0, 0.0, 8), (n, 1))  # earnings ranks: symbol 0 best
    s = np.full((n, 8), np.nan)
    s[:, :] = np.linspace(0.0, 1.0, 8)  # short-interest ranks agree with quality
    p = dataclasses.replace(p, features={"tilt_quality": q, "tilt_earnings": e, "tilt_short": s})
    w = tilt(p, {"factor": 8.0, "tilt": 0.5, "rebalance": 21})[-1]
    assert np.argmax(w) == 7  # two of three ranks favour symbol 7
    e2 = np.tile([0.0, 0.2, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0], (n, 1))
    p2 = dataclasses.replace(p, features={"tilt_quality": q, "tilt_earnings": e2, "tilt_short": np.full((n, 8), np.nan)})
    w2 = tilt(p2, {"factor": 8.0, "tilt": 0.5, "rebalance": 21})[-1]
    assert np.argmax(w2) == 2  # no short interest that day: the mean of the other two
