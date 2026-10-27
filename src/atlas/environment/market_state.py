"""What a strategy is allowed to see.

``MarketView`` holds only events that have already been *received* at the current
simulation time. It holds no reference to the engine or the venue's future quotes,
so a strategy cannot reach future data through it.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from atlas.data.events import Quote, Trade


class MarketView:
    def __init__(self, symbols: list[str], stale_ns: int) -> None:
        self.symbols: tuple[str, ...] = tuple(symbols)
        self._stale_ns = stale_ns
        self._quotes: dict[str, Quote] = {}
        self._trades: dict[str, Trade] = {}
        self._now = 0
        self.max_visible_event_ts = 0  # audit hook for no-lookahead checks

    # Engine-side mutation (underscore: not part of the strategy API).
    def _advance(self, now: int) -> None:
        if now < self._now:
            raise RuntimeError(f"simulation clock moved backwards: {now} < {self._now}")
        self._now = now

    def _receive_quote(self, quote: Quote) -> bool:
        """Returns False if the quote is older than the one already visible (late arrival)."""
        current = self._quotes.get(quote.symbol)
        self.max_visible_event_ts = max(self.max_visible_event_ts, quote.event_ts)
        if current is not None and quote.event_ts <= current.event_ts:
            return False
        self._quotes[quote.symbol] = quote
        return True

    def _receive_trade(self, trade: Trade) -> None:
        self.max_visible_event_ts = max(self.max_visible_event_ts, trade.event_ts)
        current = self._trades.get(trade.symbol)
        if current is None or trade.event_ts > current.event_ts:
            self._trades[trade.symbol] = trade

    # Strategy API.
    @property
    def now_ns(self) -> int:
        return self._now

    def quote(self, symbol: str) -> Quote | None:
        return self._quotes.get(symbol)

    def last_trade(self, symbol: str) -> Trade | None:
        return self._trades.get(symbol)

    def quote_age_ns(self, symbol: str) -> int | None:
        q = self._quotes.get(symbol)
        return None if q is None else self._now - q.event_ts

    def is_stale(self, symbol: str) -> bool:
        """True when there is no quote or the visible quote is older than the stale limit."""
        age = self.quote_age_ns(symbol)
        return age is None or age > self._stale_ns


@dataclass(frozen=True, slots=True)
class PortfolioState:
    """Immutable portfolio snapshot handed to strategies."""

    cash: float
    equity: float
    positions: Mapping[str, int]

    @staticmethod
    def of(cash: float, equity: float, positions: dict[str, int]) -> PortfolioState:
        return PortfolioState(cash=cash, equity=equity, positions=MappingProxyType(dict(positions)))

    def qty(self, symbol: str) -> int:
        return self.positions.get(symbol, 0)
