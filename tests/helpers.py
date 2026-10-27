"""Shared builders for hand-crafted market scenarios."""

from __future__ import annotations

from datetime import date
from typing import Any, Callable, Sequence

from atlas.config import DataConfig, LatencyConfig, LatencyDistribution, SimulatorConfig
from atlas.data.events import Quote, Trade
from atlas.environment.market_state import MarketView, PortfolioState
from atlas.environment.orders import OrderIntent, SimOrder
from atlas.timeutils import NS_PER_MS, session_bounds_ns

SESSION_DAY = date(2026, 10, 20)
OPEN_NS, _ = session_bounds_ns(SESSION_DAY)


def ts(ms: float) -> int:
    """Timestamp ``ms`` milliseconds after the regular-session open."""
    return OPEN_NS + int(ms * NS_PER_MS)


def quote(
    t_ms: float,
    bid: float | None = 100.00,
    ask: float | None = 100.02,
    symbol: str = "SPY",
    seq: int | None = None,
    recv_ms: float | None = None,
    bid_size: int | None = 1_000,
    ask_size: int | None = 1_000,
) -> Quote:
    return Quote(
        symbol=symbol,
        event_ts=ts(t_ms),
        receive_ts=None if recv_ms is None else ts(recv_ms),
        source="test",
        seq=seq,
        bid=bid,
        ask=ask,
        bid_size=bid_size,
        ask_size=ask_size,
    )


def trade(t_ms: float, price: float | None = 100.01, size: int | None = 10, symbol: str = "SPY",
          seq: int | None = None, recv_ms: float | None = None) -> Trade:
    return Trade(
        symbol=symbol,
        event_ts=ts(t_ms),
        receive_ts=None if recv_ms is None else ts(recv_ms),
        source="test",
        seq=seq,
        price=price,
        size=size,
    )


def fixed(ms: float) -> LatencyDistribution:
    return LatencyDistribution(kind="fixed", value_ms=ms)


def sim_config(**overrides: Any) -> SimulatorConfig:
    """Frictionless, zero-latency config unless overridden — makes expected values exact."""
    base: dict[str, Any] = {
        "data_config": "configs/data.yaml",
        "seed": 1,
        "initial_cash": 100_000.0,
        "slippage": {"bps": 0.0},
        "latency": LatencyConfig(feed=fixed(0), decision=fixed(0), order=fixed(0)),
        "stale_quote_ms": 1_000.0,
    }
    base.update(overrides)
    return SimulatorConfig.model_validate(base)


def data_config(**overrides: Any) -> DataConfig:
    base: dict[str, Any] = {
        "provider": "synthetic",
        "symbols": ["SPY", "QQQ"],
        "session_date": SESSION_DAY,
        "duration_s": 120,
        "synthetic": {"seed": 3, "initial_prices": {"SPY": 580.0, "QQQ": 500.0}},
    }
    base.update(overrides)
    return DataConfig.model_validate(base)


class ScriptedStrategy:
    """Submits pre-scripted orders on the first decision at or after given times.

    Also records everything it observes, for no-lookahead assertions.
    """

    def __init__(self, script: Sequence[tuple[float, OrderIntent]] = (),
                 hook: Callable[[MarketView, PortfolioState], None] | None = None) -> None:
        self._script = sorted(((ts(ms), intent) for ms, intent in script), key=lambda x: x[0])
        self._hook = hook
        self.observations: list[dict[str, Any]] = []
        self.updates: list[SimOrder] = []

    def on_market(self, market: MarketView, portfolio: PortfolioState) -> list[OrderIntent]:
        q = market.quote("SPY")
        self.observations.append(
            {
                "now": market.now_ns,
                "quote_event_ts": None if q is None else q.event_ts,
                "bid": None if q is None else q.bid,
                "max_visible_event_ts": market.max_visible_event_ts,
            }
        )
        if self._hook:
            self._hook(market, portfolio)
        out = []
        while self._script and self._script[0][0] <= market.now_ns:
            out.append(self._script.pop(0)[1])
        return out

    def on_order_update(self, order: SimOrder) -> None:
        self.updates.append(order)
