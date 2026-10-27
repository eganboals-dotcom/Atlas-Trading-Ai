"""Bridges an agent's target positions to the replay engine's order proposals.

The adapter owns the feature engine, throttles decisions, and converts
``target - current`` into bounded market orders. It enforces hard limits and records
any illegal action (negative/non-integer/unknown-symbol targets) as a violation.

The decision is split into ``poll`` -> ``snapshot`` -> ``intents_for`` so that an external
driver (the Gymnasium environment) can supply targets itself; ``on_market`` composes the
same steps for a normal agent, so both paths share identical order logic.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from atlas.agents.base import Agent, AgentRuntime, TargetPositions
from atlas.agents.features import FeatureEngine, SymbolFeatures
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

    # -- decision steps ------------------------------------------------------

    def poll(self, market: MarketView) -> bool:
        """Update features with newly visible events; True when a decision is due."""
        self.features.update(market)
        now = market.now_ns
        if self._next_decision is not None and now < self._next_decision:
            return False
        self._next_decision = now + self.interval_ns
        self.decisions += 1
        return True

    def snapshot(self, now_ns: int) -> dict[str, SymbolFeatures]:
        return self.features.snapshot(now_ns)

    def intents_for(
        self, targets: TargetPositions, snapshot: Mapping[str, SymbolFeatures], portfolio: PortfolioState
    ) -> list[OrderIntent]:
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

    # -- Strategy protocol -----------------------------------------------------

    def on_market(self, market: MarketView, portfolio: PortfolioState) -> Sequence[OrderIntent]:
        if not self.poll(market):
            return []
        snapshot = self.snapshot(market.now_ns)
        targets = self.agent.act(self.agent.observe(snapshot, portfolio, market.now_ns))
        return self.intents_for(targets, snapshot, portfolio)

    def on_order_update(self, order: SimOrder) -> None:
        if order.status in _TERMINAL:
            self._pending.discard(order.intent.symbol)

    def release(self, symbol: str) -> None:
        """Clear the working-order flag when a proposal never became an order (e.g. risk rejected it)."""
        self._pending.discard(symbol)
