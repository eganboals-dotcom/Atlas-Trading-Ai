"""A decision at time t may only use events received by t."""

from __future__ import annotations

import dataclasses

from helpers import ScriptedStrategy, data_config, fixed, quote, sim_config, trade, ts

from atlas.config import LatencyConfig
from atlas.data.events import Quote, visible_ts
from atlas.data.synthetic import generate_events
from atlas.data.validation import validate_events
from atlas.environment.orders import OrderIntent, Side
from atlas.environment.replay import ReplayEngine


def test_strategy_never_sees_unreceived_quote() -> None:
    events = [
        quote(0, recv_ms=1),
        quote(10, bid=200.0, ask=200.02, recv_ms=500),  # happened at 10 ms, arrives late
        trade(100, recv_ms=101),  # triggers a decision at 101 ms
    ]
    strategy = ScriptedStrategy()
    ReplayEngine(sim_config(), ["SPY"]).run(events, strategy)
    at_101 = next(o for o in strategy.observations if o["now"] == ts(101))
    assert at_101["bid"] == 100.00
    assert at_101["max_visible_event_ts"] <= at_101["now"]


def test_observed_events_were_received_before_decision() -> None:
    events = validate_events(generate_events(data_config()), 2_000).valid_events
    received_at = {e.event_ts: visible_ts(e, 0) for e in events if isinstance(e, Quote) and e.symbol == "SPY"}
    strategy = ScriptedStrategy()
    ReplayEngine(sim_config(), ["SPY", "QQQ"]).run(events, strategy)
    assert strategy.observations
    for obs in strategy.observations:
        assert obs["max_visible_event_ts"] <= obs["now"]
        if obs["quote_event_ts"] is not None:
            assert received_at[obs["quote_event_ts"]] <= obs["now"]


def test_future_shifted_values_do_not_change_past_decisions() -> None:
    """Corrupt every event after a cutoff; everything decided before it must be unchanged."""
    events = validate_events(generate_events(data_config()), 2_000).valid_events
    cutoff = ts(60_000)

    def shift(e):
        if e.event_ts < cutoff:
            return e
        if isinstance(e, Quote):
            return dataclasses.replace(e, bid=e.bid * 3, ask=e.ask * 3)
        return dataclasses.replace(e, price=e.price * 3)

    future_shifted = [shift(e) for e in events]
    script = [(ms, OrderIntent("SPY", Side.BUY if i % 2 == 0 else Side.SELL, 10))
              for i, ms in enumerate(range(1_000, 59_000, 5_000))]
    latency = LatencyConfig(feed=fixed(1), decision=fixed(1), order=fixed(5))

    def observe(evts):
        strategy = ScriptedStrategy(script)
        result = ReplayEngine(sim_config(latency=latency), ["SPY", "QQQ"]).run(evts, strategy)
        return strategy.observations, result

    obs_a, res_a = observe(events)
    obs_b, res_b = observe(future_shifted)
    before_a = [o for o in obs_a if o["now"] < cutoff]
    before_b = [o for o in obs_b if o["now"] < cutoff]
    assert before_a and before_a == before_b
    assert [f for f in res_a.fills if f.ts < cutoff] == [f for f in res_b.fills if f.ts < cutoff]
    assert [o for o in obs_a if o["now"] >= cutoff] != [o for o in obs_b if o["now"] >= cutoff]
