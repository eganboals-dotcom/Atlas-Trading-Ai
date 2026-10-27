"""Deterministic, event-driven chronological replay engine.

Two clocks are modeled explicitly:

* **Visibility clock** — a market event becomes visible to the strategy at its receive
  time (or ``event_ts`` + sampled feed latency when no receive time exists).
* **Venue clock** — the simulated venue knows the true market by ``event_ts``.

A decision made at time ``t`` produces an order that reaches the venue at
``t + decision_latency + order_latency``. It executes against the venue quote
prevailing *at arrival*, so prices can move against the order in between. The strategy
never sees anything that had not been received by ``t``.

All randomness comes from seeded generators and all ties are broken by fixed keys,
so identical data + config + seed always yields an identical run.
"""

from __future__ import annotations

import bisect
import hashlib
import heapq
import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Generator, Protocol, Sequence

from atlas.config import SimulatorConfig
from atlas.data.events import MarketEvent, Quote, Trade, visible_ts
from atlas.environment.costs import CostModel
from atlas.environment.latency import LatencySampler
from atlas.environment.market_state import MarketView, PortfolioState
from atlas.environment.orders import Fill, OrderIntent, OrderStatus, OrderType, Side, SimOrder
from atlas.environment.portfolio import Portfolio
from atlas.logging_utils import get_logger
from atlas.timeutils import NS_PER_MS, NS_PER_S

log = get_logger(__name__)

_ARRIVAL, _MARKET = 0, 1  # at equal timestamps, order arrivals are processed first


class Strategy(Protocol):
    def on_market(self, market: MarketView, portfolio: PortfolioState) -> Sequence[OrderIntent]: ...

    def on_order_update(self, order: SimOrder) -> None: ...


@dataclass
class RunResult:
    orders: list[SimOrder]
    fills: list[Fill]
    audit: list[dict[str, Any]]
    equity_curve: list[tuple[int, float]]
    final: dict[str, Any]
    stats: dict[str, int] = field(default_factory=dict)
    exposure_curve: list[tuple[int, float]] = field(default_factory=list)  # market value at curve samples
    # Wall-clock nanoseconds spent in each strategy decision. Measurement only: not part of the digest.
    decision_wall_ns: list[int] = field(default_factory=list)

    @property
    def digest(self) -> str:
        """Content hash of the run's outcome, used to prove deterministic replay."""
        payload = {
            "fills": [asdict(f) for f in self.fills],
            "orders": [(o.order_id, str(o.status), o.filled_qty, o.reject_reason) for o in self.orders],
            "final": self.final,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class _VenueBook:
    """Per-symbol quotes ordered by source time, for 'what was the market at T' lookups."""

    def __init__(self, events: Sequence[MarketEvent]) -> None:
        self._ts: dict[str, list[int]] = {}
        self._quotes: dict[str, list[Quote]] = {}
        for e in events:
            if isinstance(e, Quote):
                self._ts.setdefault(e.symbol, []).append(e.event_ts)
                self._quotes.setdefault(e.symbol, []).append(e)
        for sym, ts in self._ts.items():
            if any(b < a for a, b in zip(ts, ts[1:])):
                raise ValueError(f"venue quotes for {sym} are not in event-time order; validate data first")

    def quote_at(self, symbol: str, ts: int) -> Quote | None:
        times = self._ts.get(symbol)
        if not times:
            return None
        i = bisect.bisect_right(times, ts) - 1
        return self._quotes[symbol][i] if i >= 0 else None


class ReplayEngine:
    def __init__(self, config: SimulatorConfig, symbols: list[str]) -> None:
        self.config = config
        self.symbols = list(symbols)
        self.costs = CostModel(config.fees, config.slippage, config.spread)
        self._stale_ns = int(config.stale_quote_ms * NS_PER_MS)

    def run(self, events: Sequence[MarketEvent], strategy: Strategy) -> RunResult:
        """Replay ``events`` with a callback strategy."""
        steps = self.iter_run(events, strategy.on_order_update)
        wall: list[int] = []
        try:
            market, state = next(steps)
            while True:
                t0 = time.perf_counter_ns()
                intents = strategy.on_market(market, state)
                wall.append(time.perf_counter_ns() - t0)
                market, state = steps.send(intents)
        except StopIteration as done:
            result: RunResult = done.value
        result.decision_wall_ns = wall
        return result

    def iter_run(
        self, events: Sequence[MarketEvent], on_order_update: Callable[[SimOrder], None]
    ) -> Generator[tuple[MarketView, PortfolioState], Sequence[OrderIntent], RunResult]:
        """Step-wise replay: yields ``(market, portfolio_state)`` at every decision opportunity and
        receives the order intents for it via ``send``. Returns the ``RunResult``.

        This lets an external driver (e.g. a Gymnasium environment) control decisions while the
        engine keeps exactly the same chronology, latency, and execution semantics as ``run``.
        """
        cfg = self.config
        latency = LatencySampler(cfg.seed)
        portfolio = Portfolio(cfg.initial_cash)
        market = MarketView(self.symbols, self._stale_ns)
        venue = _VenueBook(events)
        orders: list[SimOrder] = []
        audit: list[dict[str, Any]] = []
        equity_curve: list[tuple[int, float]] = []
        exposure_curve: list[tuple[int, float]] = []
        stats = {"market_events": 0, "late_quotes": 0, "decisions": 0}

        timeline: list[tuple[int, int, int, Any]] = []
        for i, event in enumerate(events):
            seen_at = visible_ts(event, latency.sample_ns(cfg.latency.feed))
            timeline.append((seen_at, _MARKET, i, event))
        heapq.heapify(timeline)

        next_curve_ts: int | None = None
        while timeline:
            now, kind, _, payload = heapq.heappop(timeline)
            market._advance(now)

            if kind == _ARRIVAL:
                self._execute(payload, now, venue, portfolio, audit)
                on_order_update(payload)
                continue

            event: MarketEvent = payload
            stats["market_events"] += 1
            if isinstance(event, Quote):
                if market._receive_quote(event):
                    portfolio.mark(event.symbol, event.mid)
                else:
                    stats["late_quotes"] += 1
            elif isinstance(event, Trade):
                market._receive_trade(event)

            if next_curve_ts is None or now >= next_curve_ts:
                equity_curve.append((now, portfolio.equity))
                exposure_curve.append((now, portfolio.gross_exposure))
                next_curve_ts = now + NS_PER_S

            state = PortfolioState.of(
                portfolio.cash,
                portfolio.equity,
                {s: p.quantity for s, p in portfolio.positions.items() if p.quantity},
            )
            intents = (yield market, state) or ()
            stats["decisions"] += 1
            for intent in intents:
                order = self._submit(intent, now, market, latency, len(orders), audit)
                orders.append(order)
                if order.status is OrderStatus.ACCEPTED:
                    heapq.heappush(timeline, (order.arrival_ts, _ARRIVAL, order.order_id, order))
                else:
                    on_order_update(order)

        if equity_curve:
            equity_curve.append((market.now_ns, portfolio.equity))
            exposure_curve.append((market.now_ns, portfolio.gross_exposure))
        portfolio.assert_reconciled()
        final = portfolio.snapshot()
        log.info("replay complete", extra={"fills": len(portfolio.fills), "orders": len(orders), **stats})
        return RunResult(orders, list(portfolio.fills), audit, equity_curve, final, stats, exposure_curve)

    # ------------------------------------------------------------------ helpers

    def _submit(
        self,
        intent: OrderIntent,
        now: int,
        market: MarketView,
        latency: LatencySampler,
        order_id: int,
        audit: list[dict[str, Any]],
    ) -> SimOrder:
        delay = latency.sample_ns(self.config.latency.decision) + latency.sample_ns(self.config.latency.order)
        order = SimOrder(order_id=order_id, intent=intent, decision_ts=now, arrival_ts=now + delay)
        order.transition(now, OrderStatus.PROPOSED)
        visible = market.quote(intent.symbol)
        record: dict[str, Any] = {
            "type": "order",
            "ts": now,
            "order_id": order_id,
            "symbol": intent.symbol,
            "side": str(intent.side),
            "qty": intent.quantity,
            "order_type": str(intent.order_type),
            "limit_price": intent.limit_price,
            "visible_bid": visible.bid if visible else None,
            "visible_ask": visible.ask if visible else None,
            "quote_age_ms": None if visible is None else (now - visible.event_ts) / NS_PER_MS,
        }
        if intent.symbol not in self.symbols:
            self._reject(order, now, f"unknown symbol {intent.symbol}")
        elif market.is_stale(intent.symbol):
            # Never trade off a stale or missing quote; there is no forward-filling.
            self._reject(order, now, "stale or missing quote at decision")
        else:
            order.transition(now, OrderStatus.ACCEPTED)
        record.update(status=str(order.status), reason=order.reject_reason, arrival_ts=order.arrival_ts)
        audit.append(record)
        return order

    @staticmethod
    def _reject(order: SimOrder, ts: int, reason: str) -> None:
        order.reject_reason = reason
        order.transition(ts, OrderStatus.REJECTED)

    def _execute(
        self,
        order: SimOrder,
        now: int,
        venue: _VenueBook,
        portfolio: Portfolio,
        audit: list[dict[str, Any]],
    ) -> None:
        intent = order.intent
        quote = venue.quote_at(intent.symbol, now)
        record: dict[str, Any] = {"type": "execution", "ts": now, "order_id": order.order_id}

        if quote is None or now - quote.event_ts > self._stale_ns:
            self._reject(order, now, "no fresh venue quote at arrival")
        else:
            assert quote.bid is not None and quote.ask is not None
            assert quote.bid_size is not None and quote.ask_size is not None
            record.update(venue_bid=quote.bid, venue_ask=quote.ask, venue_quote_ts=quote.event_ts)
            price = self.costs.execution_price(intent.side, quote.bid, quote.ask)
            displayed = quote.ask_size if intent.side is Side.BUY else quote.bid_size
            available = max(0, math.floor(displayed * self.config.fill.max_participation))

            if intent.order_type is OrderType.LIMIT and not self._limit_ok(intent, price):
                # IOC limit that does not cross the (slipped) far side expires unfilled.
                # Touching the last trade price is never enough to fill.
                order.transition(now, OrderStatus.EXPIRED)
                order.reject_reason = "limit not marketable at arrival"
            else:
                if intent.side is Side.BUY:
                    cap = portfolio.max_affordable_qty(price, lambda q: self.costs.fee(Side.BUY, q, price))
                else:
                    cap = portfolio.position_qty(intent.symbol)
                qty = min(intent.quantity, available, cap)
                if qty <= 0:
                    reason = "insufficient cash" if intent.side is Side.BUY and cap <= 0 else (
                        "no position to sell (long-only)" if cap <= 0 else "no displayed liquidity"
                    )
                    self._reject(order, now, reason)
                else:
                    fill = Fill(
                        order_id=order.order_id,
                        symbol=intent.symbol,
                        side=intent.side,
                        quantity=qty,
                        price=price,
                        fee=self.costs.fee(intent.side, qty, price),
                        ts=now,
                        quote_bid=quote.bid,
                        quote_ask=quote.ask,
                        quote_event_ts=quote.event_ts,
                    )
                    portfolio.apply_fill(fill)
                    order.filled_qty = qty
                    if qty == intent.quantity:
                        order.transition(now, OrderStatus.FILLED)
                    else:
                        order.transition(now, OrderStatus.PARTIALLY_FILLED)
                        order.transition(now, OrderStatus.CANCELLED)  # IOC remainder
                    record.update(fill_qty=qty, fill_price=price, fee=fill.fee, simulated=True)

        record.update(status=str(order.status), reason=order.reject_reason, portfolio_equity=portfolio.equity)
        audit.append(record)

    @staticmethod
    def _limit_ok(intent: OrderIntent, price: float) -> bool:
        assert intent.limit_price is not None
        if intent.side is Side.BUY:
            return price <= intent.limit_price
        return price >= intent.limit_price
