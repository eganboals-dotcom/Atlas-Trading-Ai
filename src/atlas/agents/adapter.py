"""Bridges an agent's target positions to the replay engine's order proposals.

The adapter owns the feature engine, throttles decisions, and converts
``target - current`` into bounded market orders. It enforces hard limits and records
any illegal action (negative/non-integer/unknown-symbol targets) as a violation.
"""

from __future__ import annotations

import math
from typing import Sequence

from atlas.agents.base import Agent, AgentRuntime
from atlas.agents.features import FeatureEngine
from atlas.config import AgentLimits
from atlas.environment.market_state import MarketView, PortfolioState
from atlas.environment.orders import OrderIntent, OrderStatus, Side, SimOrder

_TERMINAL = {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED}


class AgentStrategy:
    def __init__(
        self,
        agent: Agent,
        symbols: list[str],
        limits: AgentLimits,
        decision_interval_ns: int,
        stale_ns: int,
        seed: int,
    ) -> None:
        self.agent = agent
        self.symbols = list(symbols)
        self.limits = limits
        self.interval_ns = decision_interval_ns
        self.features = FeatureEngine(symbols, stale_ns)
        self._pending: set[str] = set()
        self._next_decision: int | None = None
        self.violations: list[str] = []
        self.decisions = 0
        self.clipped = 0
        agent.initialize(AgentRuntime(tuple(symbols), limits), seed)

    def on_market(self, market: MarketView, portfolio: PortfolioState) -> Sequence[OrderIntent]:
        self.features.update(market)
        now = market.now_ns
        if self._next_decision is not None and now < self._next_decision:
            return []
        self._next_decision = now + self.interval_ns
        self.decisions += 1

        snapshot = self.features.snapshot(now)
        targets = self.agent.act(self.agent.observe(snapshot, portfolio, now))
        intents: list[OrderIntent] = []
        for symbol in sorted(targets):
            target = targets[symbol]
            if symbol not in snapshot:
                self.violations.append(f"unknown symbol {symbol!r}")
                continue
            if not isinstance(target, int) or isinstance(target, bool) or target < 0:
                self.violations.append(f"illegal target {target!r} for {symbol}")
                continue
            f = snapshot[symbol]
            cap = self.limits.max_position_shares
            if f.valid and math.isfinite(f.ask) and f.ask > 0:
                cap = min(cap, int(self.limits.max_position_frac * portfolio.equity / f.ask))
            if target > cap:
                target = cap
                self.clipped += 1
            if symbol in self._pending or not f.valid:
                continue  # one working order per symbol; never act on stale data
            delta = target - portfolio.qty(symbol)
            if delta == 0:
                continue
            qty = min(abs(delta), self.limits.max_order_shares)
            intents.append(OrderIntent(symbol, Side.BUY if delta > 0 else Side.SELL, qty))
            self._pending.add(symbol)
        return intents

    def on_order_update(self, order: SimOrder) -> None:
        if order.status in _TERMINAL:
            self._pending.discard(order.intent.symbol)
