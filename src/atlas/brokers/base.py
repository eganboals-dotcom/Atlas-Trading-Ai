"""Broker execution interface.

Only ``PaperBroker`` (``atlas.paper.broker``) implements this interface in the current
codebase. A future real broker adapter (Phase 7) must live in its own module, ship
disabled, and must not be reachable from paper-mode code. The paper session refuses any
broker whose ``is_live`` is not exactly ``False``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar

from atlas.data.events import Quote
from atlas.environment.orders import OrderIntent, SimOrder

QuoteLookup = Callable[[str, int], Quote | None]  # (symbol, ts) -> market quote at ts, if known


class LiveTradingDisabledError(RuntimeError):
    """Raised whenever anything would route an order toward a real broker."""


class BrokerExecution(ABC):
    #: True only for an adapter that can reach a real broker. None exists in this codebase.
    is_live: ClassVar[bool]

    @abstractmethod
    def submit(self, intent: OrderIntent, now_ns: int, visible_quote: Quote | None) -> SimOrder:
        """Accept or reject an order proposal; accepted orders arrive after simulated latency."""

    @abstractmethod
    def process_arrivals(self, now_ns: int, quote_at: QuoteLookup) -> list[SimOrder]:
        """Execute every working order whose arrival time has passed; returns them."""

    @abstractmethod
    def cancel_all(self, now_ns: int, reason: str) -> list[SimOrder]:
        """Cancel every working order (kill switch / safe state)."""

    @abstractmethod
    def account(self) -> dict[str, Any]:
        """Snapshot of cash, positions, equity, and P&L."""
