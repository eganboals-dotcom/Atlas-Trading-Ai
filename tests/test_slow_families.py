"""Slow (minutes-to-hours) families: session clock, entries/exits, pending orders, data-rate independence."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, time
from types import MappingProxyType

import pytest

from atlas.agents.base import AgentRecord, AgentRuntime, Observation
from atlas.agents.features import SymbolFeatures
from atlas.agents.registry import FAMILY_CLASSES, create_agent
from atlas.config import AgentLimits
from atlas.timeutils import MARKET_TZ, NS_PER_S, datetime_to_ns

DAY = date(2024, 6, 3)
OPEN = datetime_to_ns(datetime.combine(DAY, time(9, 30), tzinfo=MARKET_TZ))
MIN = 60 * NS_PER_S


def feats(mid: float, spread_bps: float = 1.0) -> SymbolFeatures:
    half = mid * spread_bps / 2e4
    f = SymbolFeatures.invalid("SPY")
    return SymbolFeatures("SPY", True, mid, mid - half, mid + half, spread_bps, 500, 500, 100.0, 0.0,
                          f.returns_bps, 1.0, 1.0, f.flow_imbalance, f.trade_volume)


class Driver:
    """Feeds an agent one price per step and fills its targets instantly (or not, to test pending orders)."""

    def __init__(self, family: str, **genome) -> None:
        g = {**FAMILY_CLASSES[family].default_genome, **genome}
        self.agent = create_agent(AgentRecord("t", family, 0, g))
        self.agent.initialize(AgentRuntime(("SPY",), AgentLimits()), seed=0)
        self.qty = 0
        self.log: list[tuple[float, int]] = []  # (minute, target) whenever the target changes

    def step(self, minute: float, mid: float, fill: bool = True) -> int:
        now = OPEN + int(minute * MIN)
        obs = Observation(now, MappingProxyType({"SPY": feats(mid)}), 100_000.0, 100_000.0,
                          MappingProxyType({"SPY": self.qty} if self.qty else {}))
        target = self.agent.act(obs)["SPY"]
        if not self.log or self.log[-1][1] != target:
            self.log.append((minute, target))
        if fill:
            self.qty = target
        return target

    def run(self, path, every_s: float = 60.0) -> None:
        """``path(minute) -> mid``, sampled every ``every_s`` seconds over the session."""
        n = int(390 * 60 / every_s)
        for i in range(n + 1):
            m = i * every_s / 60
            self.step(m, path(m))


def test_smart_hold_steps_aside_on_a_drawdown_and_comes_back_after_the_calm_period() -> None:
    d = Driver("smart_hold", entry_min=1, dd_bps=50.0, calm_min=30, vol_limit_bps=40.0, stop_bps=400.0,
               max_hold_min=390.0, flatten_min=5)

    def path(m: float) -> float:
        if m < 60:
            return 500 + m * 0.05  # calm rise to 503
        if m < 70:
            return 503 - (m - 60) * 0.4  # sharp drop: 80 bps below the high
        return 499 + (m - 70) * 0.01  # slow recovery

    d.run(path)
    entries = [m for m, t in d.log if t > 0]
    exits = [m for m, t in d.log if t == 0 and m > 0]
    assert entries[0] == 1  # in right after the open, like Hold
    assert 60 < exits[0] < 70  # out during the drop
    assert entries[1] >= exits[0] + 30  # waits out the calm period before coming back
    assert d.log[-1] == (385, 0)  # flat into the close


def test_opening_range_waits_for_the_range_then_buys_one_breakout() -> None:
    d = Driver("opening_range", range_min=30, buffer_bps=5.0, target_bps=40.0, last_entry_min=240,
               stop_bps=400.0, max_hold_min=390.0)
    d.run(lambda m: 500 + (1.0 if m < 30 and int(m) % 2 else 0.0) + (0.0 if m < 100 else (m - 100) * 0.03))
    entries = [m for m, t in d.log if t > 0]
    assert len(entries) == 1 and entries[0] > 100  # range high 501 + 5 bps, broken only after minute ~100
    exit_m = next(m for m, t in d.log if t == 0 and m > entries[0])
    assert exit_m < 385  # took profit at the 40 bps target before the close


@pytest.mark.parametrize("morning, expect_trade", [(0.30, True), (-0.30, False)])
def test_intraday_momentum_only_follows_a_strong_morning(morning: float, expect_trade: bool) -> None:
    d = Driver("intraday_momentum", signal_min=30, entry_min=330, threshold_bps=20.0, max_hold_min=60.0)
    d.run(lambda m: 500 * (1 + morning / 100 * min(m, 30) / 30))
    entries = [m for m, t in d.log if t > 0]
    assert (entries == [330]) is expect_trade or (not entries and not expect_trade)


def test_a_slow_entry_that_has_not_filled_is_not_re_decided() -> None:
    d = Driver("smart_hold", entry_min=1)
    first = d.step(1, 500.0, fill=False)
    assert first > 0
    assert d.step(1 + 5 / 60, 500.0, fill=False) == first  # still working: same target, no second entry
    assert d.agent._books["SPY"].entries == 1


def test_an_exit_that_has_not_filled_keeps_the_target_at_zero() -> None:
    d = Driver("smart_hold", entry_min=1, flatten_min=5)
    d.step(1, 500.0)
    assert d.qty > 0
    assert d.step(385, 500.0, fill=False) == 0  # flatten time
    assert d.step(385 + 2 / 60, 500.0, fill=False) == 0  # still held: must not flip back to holding


@pytest.mark.parametrize("family", ["trend_follow", "smart_hold", "pullback", "opening_range"])
def test_same_decisions_on_one_second_and_one_minute_data(family: str) -> None:
    # Live sees a quote every second, training a bar every minute: trades should land in the same minute.
    def path(m: float) -> float:
        k = int(m)  # price changes once a minute, so both feeds see the same minute closes
        return 500 * math.exp(0.002 * math.sin(k / 25) + 0.0001 * k)

    slow, fast = Driver(family), Driver(family)
    slow.run(path, every_s=60)
    fast.run(path, every_s=2)
    assert [(int(m), t > 0) for m, t in slow.log] == [(int(m), t > 0) for m, t in fast.log]


def test_state_resets_each_session() -> None:
    d = Driver("opening_range", range_min=5, buffer_bps=0.0, last_entry_min=300)
    d.run(lambda m: 500 + max(0.0, m - 10) * 0.02)
    assert d.agent._books["SPY"].entries == 1
    nxt = OPEN + 86_400 * NS_PER_S + 6 * MIN  # next day, minute 6
    obs = Observation(nxt, MappingProxyType({"SPY": feats(400.0)}), 1e5, 1e5, MappingProxyType({}))
    d.agent.act(obs)
    assert d.agent._books["SPY"].entries == 0 and d.agent._books["SPY"].open_px == 400.0


def test_bar_sessions_replay_end_to_end(tmp_path) -> None:
    from helpers import sim_config

    from atlas.data.alpaca_history import bar_events, merge_bar_sessions, regular_hours_days
    from atlas.environment.replay import ReplayEngine
    from atlas.agents.adapter import AgentStrategy

    def iso(ns: int) -> str:
        return datetime.fromtimestamp(ns / NS_PER_S, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    bars = [{"t": iso(OPEN + k * MIN), "c": 500 + 0.02 * k, "v": 1000} for k in range(-30, 400)]  # incl. pre/post
    day = regular_hours_days(bars)
    assert list(day) == [DAY] and len(day[DAY]) == 390
    events = merge_bar_sessions({"SPY": bar_events(day[DAY], "SPY", 1.0)})
    q = [e for e in events if e.event_type == "quote"][0]
    assert q.event_ts == OPEN + MIN and math.isclose(q.spread_bps, 1.0, rel_tol=1e-3)
    sim = sim_config().model_copy(update={"stale_quote_ms": 65_000.0, "initial_cash": 14_285.71})
    rec = AgentRecord("h", "smart_hold", 0, dict(FAMILY_CLASSES["smart_hold"].default_genome))
    strat = AgentStrategy(create_agent(rec), ["SPY"], AgentLimits(), decision_interval_ns=2 * NS_PER_S,
                          stale_ns=65 * NS_PER_S, seed=0)
    result = ReplayEngine(sim, ["SPY"]).run(events, strat)
    assert len(result.fills) >= 2  # bought after the open, sold before the close
    bought = sum(f.quantity for f in result.fills if str(f.side) == "buy")
    sold = sum(f.quantity for f in result.fills if str(f.side) == "sell")
    assert bought == sold > 0


def test_open_dependent_bots_sit_out_a_day_they_joined_late() -> None:
    # e.g. the paper session restarted at 11:00: the first price seen is not the day's open
    d = Driver("opening_range", range_min=5, buffer_bps=0.0, last_entry_min=300)
    for k in range(90, 200):
        d.step(k, 500 + (k - 90) * 0.05)
    assert d.agent._books["SPY"].entries == 0
    h = Driver("smart_hold", entry_min=1)  # doesn't need the open: still trades
    for k in range(90, 100):
        h.step(k, 500.0)
    assert h.agent._books["SPY"].entries == 1
