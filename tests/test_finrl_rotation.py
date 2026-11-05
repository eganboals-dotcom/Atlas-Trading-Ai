"""The copy of FinRL's rule: its pieces match the code they copy, later prices can't reach a book, and a book trades the next session."""

from __future__ import annotations

import dataclasses
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "research" / "investing"))
import v27_finrl_rotation as rot  # noqa: E402
import v29_monthly_rotation as slow  # noqa: E402
from atlas.swing.data import Panel  # noqa: E402

SYMBOLS = ("A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "SPY", "QQQ", "IAU", "XLU", "XLV")


def weekdays(n: int) -> tuple[date, ...]:
    days, d = [], date(2019, 1, 7)
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return tuple(days)


def market(drift: dict[str, float] | None = None, days: int = 400, seed: int = 3) -> rot.Rotation:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0005, 0.01, (days, len(SYMBOLS)))
    for s, d in (drift or {}).items():
        r[:, SYMBOLS.index(s)] += d
    close = 100 * np.cumprod(1 + r, axis=0)
    return rot.rotation(Panel(weekdays(days), SYMBOLS, close, close, close, close, np.ones_like(close)))


GROUPS = {"a": [0, 1, 2, 3], "b": [4, 5, 6, 7]}


def test_week_ends_are_the_last_session_of_each_week():
    days = weekdays(11)
    assert [days[i].weekday() for i in rot.week_ends(days)] == [4, 4]


def test_week_ends_handle_a_short_week():
    days = (date(2024, 3, 25), date(2024, 3, 28), date(2024, 4, 1))  # Good Friday
    assert rot.week_ends(days) == [1]


def test_robust_z_matches_robust_stats():
    assert rot.robust_z(np.array([1.0, 2.0, 4.0])) == pytest.approx(2.0)  # median 2, MAD 1
    assert rot.robust_z(np.array([1.0, 1.0, 1.0])) == 0.0
    assert rot.robust_z(np.array([1.0, 1.0, 1.0, 2.0])) == 10.0


def test_a_group_that_lags_the_benchmark_is_not_kept():
    m = market({"QQQ": 0.01})
    k = len(m.ends) - 1
    assert rot.choose(m, k, GROUPS).kept == ("fallback",)


def test_fallback_is_equal_at_the_budget():
    m = market({"QQQ": 0.01})
    k = len(m.ends) - 1
    b = rot.choose(m, k, GROUPS)
    assert sorted(b.weights) == sorted(m.fallback)
    assert sum(b.weights.values()) == pytest.approx(b.budget)
    assert len(set(b.weights.values())) == 1


def test_strong_groups_are_kept_and_top_assets_taken():
    m = market({s: 0.004 for s in SYMBOLS[:8]})
    b = rot.choose(m, len(m.ends) - 1, GROUPS)
    assert set(b.kept) == {"a", "b"}
    assert len(b.weights) == 2 * rot.rule["top_assets"]
    assert sum(b.weights.values()) <= 1 + 1e-9


def test_budget_steps_down_with_the_risk_score():
    m = market({"SPY": 0.0})
    # A steep fall in SPY's last weeks breaks the 26-week mean and the 13-week peak.
    close = m.p.close.copy()
    close[-30:, m.spy] *= np.linspace(1.0, 0.75, 30)
    fallen = rot.rotation(dataclasses.replace(m.p, close=close))
    k = len(fallen.ends) - 1
    assert rot.budget(fallen, k, close[fallen.ends[: k + 1]]) == 0.7


def test_a_shock_session_cuts_the_budget_at_once():
    m = market({"SPY": 0.002, "QQQ": 0.002})
    k = len(m.ends) - 1
    t = m.ends[k]
    close = m.p.close.copy()
    close[t:, m.qqq] *= 0.95
    hit = rot.rotation(dataclasses.replace(m.p, close=close))
    assert rot.budget(hit, k, close[hit.ends[: k + 1]]) == 0.7


def test_later_prices_cannot_change_a_book():
    m = market({s: 0.002 for s in SYMBOLS[:4]})
    k = len(m.ends) - 5
    t = m.ends[k]
    before = rot.choose(m, k, GROUPS)
    close = m.p.close.copy()
    close[t + 1:] *= np.random.default_rng(9).uniform(0.5, 1.5, close[t + 1:].shape)
    changed = rot.rotation(dataclasses.replace(m.p, close=close))
    assert rot.choose(changed, k, GROUPS) == dataclasses.replace(before, t=t)


def test_a_book_is_held_from_the_session_after_its_signal():
    x = rot.exposure([{0: 0.5}, {1: 0.25}], [2, 5], (9, 4))
    assert x[:3, 0].tolist() == [0.0, 0.0, 0.0]
    assert x[3:6, 0].tolist() == [2.0, 2.0, 2.0]
    assert x[6:, 0].tolist() == [0.0, 0.0, 0.0]
    assert x[6:, 1].tolist() == [1.0, 1.0, 1.0]
    assert x[:6, 1].tolist() == [0.0] * 6


def test_random_runs_keep_the_weights():
    m = market({s: 0.004 for s in SYMBOLS[:8]})
    b = rot.choose(m, len(m.ends) - 1, GROUPS)
    assert sorted(v for _, ws in b.slots for v in ws) == sorted(b.weights.values())


def test_the_published_groups_are_kept_verbatim():
    assert rot.cfg["groups_v121"]["growth"] == ["AAPL", "MSFT", "NVDA", "META", "AMZN", "GOOGL", "TSLA"]
    assert rot.cfg["groups_v121"]["defensive"] == rot.cfg["free_fixed"]["defensive"]
    assert rot.cfg["free_fixed"]["growth"] == []


def test_monthly_keeps_the_last_book_of_each_month():
    days = tuple(date(2024, 1, 5) + timedelta(days=7 * i) for i in range(10))  # Fridays, January to early March
    books = [rot.Book(i, {}, (), 1.0, ()) for i in range(10)]
    assert [days[b.t] for b in slow.monthly(books, days)] == [date(2024, 1, 26), date(2024, 2, 23), date(2024, 3, 8)]
