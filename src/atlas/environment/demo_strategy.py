"""Plumbing-only demo strategy: buy, hold for a while, sell; repeat.

It exists to exercise the replay engine end to end. It is not a research strategy and
its results mean nothing about profitability. Phase 2 replaces it with the agent
interface and the 50-agent population.
"""

from __future__ import annotations

from typing import Sequence

from atlas.config import DemoStrategyConfig
from atlas.environment.market_state import MarketView, PortfolioState
from atlas.environment.orders import OrderIntent, OrderStatus, Side, SimOrder
from atlas.timeutils import NS_PER_S

_TERMINAL = {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED}


class BuyHoldSellDemo:
    def __init__(self, config: DemoStrategyConfig, symbol: str) -> None:
        self.config = config
        self.symbol = symbol
        self._pending = False
        self._entry_ts: int | None = None
        self._trips = 0

    def on_market(self, market: MarketView, portfolio: PortfolioState) -> Sequence[OrderIntent]:
        if self._pending or market.is_stale(self.symbol):
            return []
        held = portfolio.qty(self.symbol)
        if held == 0 and self._trips < self.config.round_trips:
            self._pending = True
            self._entry_ts = market.now_ns
            return [OrderIntent(self.symbol, Side.BUY, self.config.order_qty)]
        if held > 0 and self._entry_ts is not None:
            if market.now_ns - self._entry_ts >= self.config.hold_s * NS_PER_S:
                self._pending = True
                return [OrderIntent(self.symbol, Side.SELL, held)]
        return []

    def on_order_update(self, order: SimOrder) -> None:
        if order.status in _TERMINAL:
            self._pending = False
            if order.intent.side is Side.SELL and order.status is OrderStatus.FILLED:
                self._trips += 1
