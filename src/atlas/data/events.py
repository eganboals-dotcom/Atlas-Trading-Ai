"""Typed, timestamped market events.

Two timestamps are kept deliberately separate:

* ``event_ts`` — when the event happened at the source (exchange/vendor/synthetic clock).
* ``receive_ts`` — when *we* received it, if known.

An agent may only see an event once it has been received; the simulated venue uses
``event_ts`` to know the true market at any moment. Source timestamps are never
replaced by processing time.

Price/size fields are Optional so that malformed vendor records can be represented
and rejected by validation instead of crashing ingestion.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, ClassVar


class EventType(StrEnum):
    QUOTE = "quote"
    TRADE = "trade"


class TimestampKind(StrEnum):
    """Provenance of ``event_ts`` — the spec requires tracking this explicitly."""

    EXCHANGE = "exchange"
    VENDOR = "vendor"
    LOCAL = "local"
    SYNTHETIC = "synthetic"


@dataclass(frozen=True, slots=True)
class Quote:
    """Level 1 (top-of-book) quote."""

    symbol: str
    event_ts: int
    source: str
    bid: float | None
    ask: float | None
    bid_size: int | None
    ask_size: int | None
    receive_ts: int | None = None
    seq: int | None = None
    timestamp_kind: TimestampKind = TimestampKind.SYNTHETIC
    venue: str | None = None
    conditions: tuple[str, ...] = ()

    event_type: ClassVar[EventType] = EventType.QUOTE

    @property
    def mid(self) -> float:
        assert self.bid is not None and self.ask is not None
        return (self.bid + self.ask) / 2.0

    @property
    def spread_bps(self) -> float:
        return (self.ask - self.bid) / self.mid * 10_000.0  # type: ignore[operator]


@dataclass(frozen=True, slots=True)
class Trade:
    """Last-sale print."""

    symbol: str
    event_ts: int
    source: str
    price: float | None
    size: int | None
    receive_ts: int | None = None
    seq: int | None = None
    timestamp_kind: TimestampKind = TimestampKind.SYNTHETIC
    venue: str | None = None
    conditions: tuple[str, ...] = ()

    event_type: ClassVar[EventType] = EventType.TRADE


MarketEvent = Quote | Trade


def visible_ts(event: MarketEvent, default_feed_latency_ns: int) -> int:
    """Earliest time an agent may observe ``event``.

    When no receive timestamp exists we assume a configured feed latency rather than
    pretending the event was seen instantly at its source time.
    """
    if event.receive_ts is not None:
        return event.receive_ts
    return event.event_ts + default_feed_latency_ns


def is_finite_number(value: float | int | None) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value)) and math.isfinite(value)


def event_to_dict(event: MarketEvent) -> dict[str, Any]:
    out = asdict(event)
    out["event_type"] = str(event.event_type)
    out["timestamp_kind"] = str(event.timestamp_kind)
    out["conditions"] = list(event.conditions)
    return out
