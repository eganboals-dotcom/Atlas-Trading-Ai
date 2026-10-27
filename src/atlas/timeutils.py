"""Time helpers. All internal timestamps are integer nanoseconds since the Unix epoch, UTC.

Integers avoid floating-point drift and make event ordering exact and deterministic.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

NS_PER_US = 1_000
NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000

MARKET_TZ = ZoneInfo("America/New_York")
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)


def ms_to_ns(ms: float) -> int:
    return int(round(ms * NS_PER_MS))


def datetime_to_ns(dt: datetime) -> int:
    """Convert a timezone-aware datetime to epoch nanoseconds."""
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware; naive datetimes are ambiguous")
    delta = dt.astimezone(UTC) - datetime(1970, 1, 1, tzinfo=UTC)
    return (delta.days * 86_400 + delta.seconds) * NS_PER_S + delta.microseconds * NS_PER_US


def ns_to_datetime(ns: int) -> datetime:
    """Convert epoch nanoseconds to a UTC datetime (microsecond precision)."""
    return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=ns // NS_PER_US)


def ns_to_iso(ns: int) -> str:
    return ns_to_datetime(ns).isoformat()


def session_bounds_ns(day: date) -> tuple[int, int]:
    """Regular-session open/close for a trading day, in epoch ns.

    Phase 1 limitation: holidays and early closes are not modeled; a full exchange
    calendar is planned for the historical-data phase.
    """
    open_dt = datetime.combine(day, REGULAR_OPEN, tzinfo=MARKET_TZ)
    close_dt = datetime.combine(day, REGULAR_CLOSE, tzinfo=MARKET_TZ)
    return datetime_to_ns(open_dt), datetime_to_ns(close_dt)


def in_regular_session(ns: int) -> bool:
    local = ns_to_datetime(ns).astimezone(MARKET_TZ)
    if local.weekday() >= 5:
        return False
    start, end = session_bounds_ns(local.date())
    return start <= ns < end
