"""Order intents, lifecycle states, and fill records for the simulator.

These objects never leave the process: there is no broker integration in Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"


class TimeInForce(StrEnum):
    IOC = "ioc"  # the only supported TIF in Phase 1


class OrderStatus(StrEnum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """What a strategy proposes. Size, type, and time-in-force are always explicit."""

    symbol: str
    side: Side
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.IOC

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"order quantity must be positive, got {self.quantity}")
        if self.order_type is OrderType.LIMIT and (self.limit_price is None or self.limit_price <= 0):
            raise ValueError("limit orders require a positive limit_price")
        if self.order_type is OrderType.MARKET and self.limit_price is not None:
            raise ValueError("market orders must not carry a limit_price")


@dataclass(slots=True)
class SimOrder:
    order_id: int
    intent: OrderIntent
    decision_ts: int
    arrival_ts: int
    status: OrderStatus = OrderStatus.PROPOSED
    filled_qty: int = 0
    reject_reason: str | None = None
    history: list[tuple[int, OrderStatus]] = field(default_factory=list)

    def transition(self, ts: int, status: OrderStatus) -> None:
        self.status = status
        self.history.append((ts, status))


@dataclass(frozen=True, slots=True)
class Fill:
    """A *simulated* fill. Never an actual execution."""

    order_id: int
    symbol: str
    side: Side
    quantity: int
    price: float
    fee: float
    ts: int
    quote_bid: float
    quote_ask: float
    quote_event_ts: int
    simulated: bool = True
