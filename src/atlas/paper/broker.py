"""Internal paper broker. It simulates order lifecycle and fills, and never contacts a real broker.

Fill model: identical to the backtest (``execute_ioc``). Orders are immediate-or-cancel
against the top of book prevailing when the order *arrives*, after decision and order
latency, capped by displayed size times ``max_participation``, cash, and position.

In live paper trading the "market at arrival" is the most recent quote *received
locally* whose source time is at or before the arrival time. It is therefore subject to
feed latency, a documented approximation. Every fill is marked ``simulated: true`` and
is never presented as an actual execution.

Lifecycle: proposed -> accepted | rejected; accepted -> filled | partially filled + cancelled
(IOC remainder) | expired (non-marketable limit) | rejected (no fresh quote, no cash or
position) | cancelled (kill switch, safe state, or restart).
"""

from __future__ import annotations

import heapq
from typing import Any, Callable, ClassVar

from atlas.brokers.base import BrokerExecution, QuoteLookup
from atlas.config import SimulatorConfig
from atlas.data.events import Quote
from atlas.environment.costs import CostModel
from atlas.environment.latency import LatencySampler
from atlas.environment.orders import OrderIntent, OrderStatus, OrderType, Side, SimOrder, TimeInForce
from atlas.environment.portfolio import Portfolio
from atlas.environment.replay import execute_ioc, reject
from atlas.timeutils import NS_PER_MS

AuditSink = Callable[[dict[str, Any]], None]


class PaperBroker(BrokerExecution):
    is_live: ClassVar[bool] = False

    def __init__(self, account_id: str, sim: SimulatorConfig, seed: int, audit: AuditSink,
                 portfolio: Portfolio | None = None) -> None:
        self.account_id = account_id
        self.sim = sim
        self.costs = CostModel(sim.fees, sim.slippage, sim.spread)
        self.latency = LatencySampler(seed)
        self.portfolio = portfolio or Portfolio(sim.initial_cash)
        self.stale_ns = int(sim.stale_quote_ms * NS_PER_MS)
        self._audit = audit
        self._working: list[tuple[int, int, SimOrder]] = []
        self.orders: list[SimOrder] = []
        self.next_order_id = 0

    def _emit(self, record: dict[str, Any]) -> None:
        self._audit({"account": self.account_id, "simulated": True, **record})

    def submit(self, intent: OrderIntent, now_ns: int, visible_quote: Quote | None) -> SimOrder:
        delay = self.latency.sample_ns(self.sim.latency.decision) + self.latency.sample_ns(self.sim.latency.order)
        order = SimOrder(order_id=self.next_order_id, intent=intent, decision_ts=now_ns, arrival_ts=now_ns + delay)
        self.next_order_id += 1
        order.transition(now_ns, OrderStatus.PROPOSED)
        if visible_quote is None or now_ns - visible_quote.event_ts > self.stale_ns:
            reject(order, now_ns, "stale or missing quote at decision")
        else:
            order.transition(now_ns, OrderStatus.ACCEPTED)
            heapq.heappush(self._working, (order.arrival_ts, order.order_id, order))
        self.orders.append(order)
        self._emit({
            "type": "order", "ts": now_ns, "order_id": order.order_id, "symbol": intent.symbol,
            "side": str(intent.side), "qty": intent.quantity, "order_type": str(intent.order_type),
            "visible_bid": visible_quote.bid if visible_quote else None,
            "visible_ask": visible_quote.ask if visible_quote else None,
            "quote_age_ms": None if visible_quote is None else (now_ns - visible_quote.event_ts) / NS_PER_MS,
            "status": str(order.status), "reason": order.reject_reason, "arrival_ts": order.arrival_ts,
        })
        return order

    def process_arrivals(self, now_ns: int, quote_at: QuoteLookup) -> list[SimOrder]:
        done = []
        while self._working and self._working[0][0] <= now_ns:
            arrival, _, order = heapq.heappop(self._working)
            record = execute_ioc(order, quote_at(order.intent.symbol, arrival), arrival, self.costs,
                                 self.sim.fill.max_participation, self.stale_ns, self.portfolio)
            self._emit({**record, "fill_assumption": "IOC top-of-book at arrival, latest received quote",
                        "cash": self.portfolio.cash,
                        "positions": {s: p.quantity for s, p in self.portfolio.positions.items() if p.quantity}})
            done.append(order)
        return done

    def cancel_all(self, now_ns: int, reason: str) -> list[SimOrder]:
        cancelled = []
        while self._working:
            _, _, order = heapq.heappop(self._working)
            order.reject_reason = reason
            order.transition(now_ns, OrderStatus.CANCELLED)
            self._emit({"type": "cancel", "ts": now_ns, "order_id": order.order_id, "reason": reason,
                        "status": str(order.status)})
            cancelled.append(order)
        return cancelled

    @property
    def working_orders(self) -> list[SimOrder]:
        return [o for _, _, o in sorted(self._working)]

    def account(self) -> dict[str, Any]:
        return {"account": self.account_id, "simulated": True, **self.portfolio.snapshot()}

    # -- persistence -----------------------------------------------------------

    def to_state(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "portfolio": self.portfolio.to_state(),
            "next_order_id": self.next_order_id,
            "latency_rng": self.latency._rng.bit_generator.state,
            "working": [
                {"order_id": o.order_id, "symbol": o.intent.symbol, "side": str(o.intent.side),
                 "qty": o.intent.quantity, "arrival_ts": o.arrival_ts} for o in self.working_orders
            ],
            "order_counts": _count(str(o.status) for o in self.orders),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any], sim: SimulatorConfig, audit: AuditSink, now_ns: int) -> PaperBroker:
        """Restore an account. Orders that were working at shutdown are recorded as cancelled,
        never assumed filled."""
        broker = cls(state["account_id"], sim, 0, audit, Portfolio.from_state(state["portfolio"]))
        broker.next_order_id = state["next_order_id"]
        broker.latency._rng.bit_generator.state = state["latency_rng"]
        for w in state["working"]:
            order = SimOrder(w["order_id"], OrderIntent(w["symbol"], Side(w["side"]), w["qty"], OrderType.MARKET,
                                                        None, TimeInForce.IOC), now_ns, w["arrival_ts"])
            order.reject_reason = "process restart: working order cancelled, not assumed filled"
            order.transition(now_ns, OrderStatus.CANCELLED)
            broker.orders.append(order)
            broker._emit({"type": "cancel", "ts": now_ns, "order_id": order.order_id,
                          "reason": order.reject_reason, "status": str(order.status)})
        return broker


def _count(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out
