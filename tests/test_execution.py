from __future__ import annotations

import pytest
from helpers import ScriptedStrategy, fixed, quote, sim_config, trade

from atlas.config import LatencyConfig
from atlas.environment.orders import OrderIntent, OrderStatus, OrderType, Side
from atlas.environment.replay import ReplayEngine

BUY = lambda qty=100: OrderIntent("SPY", Side.BUY, qty)  # noqa: E731
SELL = lambda qty=100: OrderIntent("SPY", Side.SELL, qty)  # noqa: E731


def run(events, script, **cfg):
    strategy = ScriptedStrategy(script)
    result = ReplayEngine(sim_config(**cfg), ["SPY"]).run(events, strategy)
    return result, strategy


def test_buy_fills_at_ask_and_sell_at_bid_not_midpoint() -> None:
    events = [quote(0), quote(100), quote(200)]
    result, _ = run(events, [(0, BUY()), (150, SELL())])
    buy, sell = result.fills
    assert buy.price == pytest.approx(100.02)
    assert sell.price == pytest.approx(100.00)
    assert result.final["realized_pnl"] == pytest.approx(-2.0)  # paid the spread


def test_slippage_is_adverse_on_both_sides() -> None:
    events = [quote(0), quote(100), quote(200)]
    result, _ = run(events, [(0, BUY()), (150, SELL())], slippage={"bps": 10})
    buy, sell = result.fills
    assert buy.price == pytest.approx(100.02 * 1.001)
    assert sell.price == pytest.approx(100.00 * 0.999)


def test_spread_widening_is_applied() -> None:
    result, _ = run([quote(0)], [(0, BUY())], spread={"widen_bps": 5})
    assert result.fills[0].price == pytest.approx(100.02 * 1.0005)


def test_fees_are_charged() -> None:
    events = [quote(0), quote(100)]
    result, _ = run(
        events, [(0, BUY()), (50, SELL())],
        fees={"per_share": 0.005, "min_per_order": 1.0, "sell_regulatory_bps": 1.0},
    )
    buy, sell = result.fills
    assert buy.fee == pytest.approx(1.0)  # 0.50 per-share fee lifted to the minimum
    assert sell.fee == pytest.approx(1.0 + 100 * 100.00 * 1e-4)
    assert result.final["fees_paid"] == pytest.approx(buy.fee + sell.fee)


def test_latency_fills_at_market_prevailing_on_arrival() -> None:
    """Price moves between decision and arrival; the order pays the new price."""
    events = [quote(0), quote(10, bid=101.00, ask=101.02), quote(500, bid=101.00, ask=101.02)]
    latency = LatencyConfig(feed=fixed(0), decision=fixed(1), order=fixed(19))
    result, strategy = run(events, [(0, BUY())], latency=latency)
    order = strategy.updates[0]
    assert order.arrival_ts - order.decision_ts == 20_000_000
    assert result.fills[0].price == pytest.approx(101.02)
    assert result.audit[0]["visible_ask"] == pytest.approx(100.02)


def test_partial_fill_limited_by_displayed_size() -> None:
    result, strategy = run([quote(0, ask_size=1_000)], [(0, BUY(800))], fill={"max_participation": 0.5})
    assert result.fills[0].quantity == 500
    order = strategy.updates[0]
    assert order.status is OrderStatus.CANCELLED
    assert [s for _, s in order.history][-2:] == [OrderStatus.PARTIALLY_FILLED, OrderStatus.CANCELLED]


def test_limit_touching_last_trade_does_not_fill() -> None:
    """A buy limit at the last trade price, below the ask, is not marketable."""
    events = [quote(0), trade(1, price=100.01), quote(100)]
    touching = OrderIntent("SPY", Side.BUY, 10, OrderType.LIMIT, limit_price=100.01)
    result, strategy = run(events, [(1, touching)])
    assert result.fills == []
    assert strategy.updates[0].status is OrderStatus.EXPIRED


def test_marketable_limit_fills() -> None:
    crossing = OrderIntent("SPY", Side.BUY, 10, OrderType.LIMIT, limit_price=100.02)
    result, _ = run([quote(0)], [(0, crossing)])
    assert result.fills[0].price == pytest.approx(100.02)


def test_limit_does_not_fill_if_slippage_breaches_limit() -> None:
    crossing = OrderIntent("SPY", Side.BUY, 10, OrderType.LIMIT, limit_price=100.02)
    result, strategy = run([quote(0)], [(0, crossing)], slippage={"bps": 1})
    assert result.fills == []
    assert strategy.updates[0].status is OrderStatus.EXPIRED


def test_stale_quote_at_decision_is_rejected() -> None:
    events = [quote(0), trade(5_000)]  # trade triggers a decision 5 s after the only quote
    result, strategy = run(events, [(5_000, BUY())], stale_quote_ms=1_000)
    assert result.fills == []
    assert strategy.updates[0].status is OrderStatus.REJECTED
    assert "stale" in strategy.updates[0].reject_reason


def test_no_fresh_venue_quote_at_arrival_is_rejected() -> None:
    latency = LatencyConfig(feed=fixed(0), decision=fixed(0), order=fixed(2_000))
    result, strategy = run([quote(0)], [(0, BUY())], latency=latency, stale_quote_ms=1_000)
    assert result.fills == []
    assert strategy.updates[0].reject_reason == "no fresh venue quote at arrival"


def test_buy_without_cash_is_rejected() -> None:
    result, strategy = run([quote(0)], [(0, BUY())], initial_cash=50.0)
    assert result.fills == []
    assert strategy.updates[0].reject_reason == "insufficient cash"


def test_buy_is_capped_by_cash() -> None:
    result, _ = run([quote(0)], [(0, BUY())], initial_cash=1_000.0)
    assert result.fills[0].quantity == 9  # 9 * 100.02 fits; 10 does not


def test_sell_without_position_is_rejected_long_only() -> None:
    result, strategy = run([quote(0)], [(0, SELL())])
    assert result.fills == []
    assert strategy.updates[0].reject_reason == "no position to sell (long-only)"


def test_unknown_symbol_is_rejected() -> None:
    result, strategy = run([quote(0)], [(0, OrderIntent("TSLA", Side.BUY, 1))])
    assert strategy.updates[0].status is OrderStatus.REJECTED


def test_invalid_order_intents_raise() -> None:
    with pytest.raises(ValueError):
        OrderIntent("SPY", Side.BUY, 0)
    with pytest.raises(ValueError):
        OrderIntent("SPY", Side.BUY, 1, OrderType.LIMIT)
    with pytest.raises(ValueError):
        OrderIntent("SPY", Side.BUY, 1, OrderType.MARKET, limit_price=1.0)


def test_every_fill_is_marked_simulated_and_audited() -> None:
    result, _ = run([quote(0), quote(100)], [(0, BUY()), (50, SELL())])
    assert all(f.simulated for f in result.fills)
    assert {r["type"] for r in result.audit} == {"order", "execution"}
    assert len([r for r in result.audit if r.get("fill_qty")]) == len(result.fills)
