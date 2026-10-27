"""Market-data validation.

Events are checked in the order they were delivered. Defective events are dropped,
never repaired: we do not reorder out-of-sequence data (that would invent chronology)
and we do not forward-fill missing quotes. Stale gaps are flagged rather than dropped
so downstream consumers can suspend trading for their duration.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from atlas.data.events import MarketEvent, Quote, Trade, is_finite_number
from atlas.timeutils import NS_PER_MS, in_regular_session, ns_to_iso


class IssueKind(StrEnum):
    DUPLICATE = "duplicate"
    OUT_OF_ORDER = "out_of_order"
    MISSING = "missing"
    IMPOSSIBLE = "impossible"
    CLOCK_ANOMALY = "clock_anomaly"
    OUT_OF_SESSION = "out_of_session"
    STALE_GAP = "stale_gap"


#: Issues that cause the event to be excluded from the clean stream.
DROPPING = frozenset(
    {
        IssueKind.DUPLICATE,
        IssueKind.OUT_OF_ORDER,
        IssueKind.MISSING,
        IssueKind.IMPOSSIBLE,
        IssueKind.CLOCK_ANOMALY,
        IssueKind.OUT_OF_SESSION,
    }
)


@dataclass(frozen=True, slots=True)
class Issue:
    kind: IssueKind
    index: int
    symbol: str
    detail: str

    @property
    def drops_event(self) -> bool:
        return self.kind in DROPPING


@dataclass
class ValidationReport:
    total_events: int
    valid_events: list[MarketEvent]
    issues: list[Issue] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        return dict(sorted(Counter(str(i.kind) for i in self.issues).items()))

    @property
    def dropped(self) -> int:
        return self.total_events - len(self.valid_events)

    @property
    def is_clean(self) -> bool:
        return not self.issues

    def summary(self) -> dict[str, Any]:
        return {
            "total_events": self.total_events,
            "valid_events": len(self.valid_events),
            "dropped_events": self.dropped,
            "issue_counts": self.counts,
            "clean": self.is_clean,
        }


class DataValidationError(ValueError):
    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__(f"market data failed validation: {report.counts}")


def _content_problems(event: MarketEvent) -> tuple[IssueKind, str] | None:
    if isinstance(event, Quote):
        fields = {"bid": event.bid, "ask": event.ask, "bid_size": event.bid_size, "ask_size": event.ask_size}
    elif isinstance(event, Trade):
        fields = {"price": event.price, "size": event.size}
    else:  # pragma: no cover - guarded by typing
        return IssueKind.IMPOSSIBLE, f"unknown event type {type(event).__name__}"

    missing = [name for name, value in fields.items() if not is_finite_number(value)]
    if missing:
        return IssueKind.MISSING, f"missing/non-finite fields: {', '.join(missing)}"
    nonpositive = [name for name, value in fields.items() if value <= 0]  # type: ignore[operator]
    if nonpositive:
        return IssueKind.IMPOSSIBLE, f"non-positive fields: {', '.join(nonpositive)}"
    if isinstance(event, Quote) and event.ask < event.bid:  # type: ignore[operator]
        return IssueKind.IMPOSSIBLE, f"crossed quote bid={event.bid} ask={event.ask}"
    return None


def validate_events(
    events: list[MarketEvent],
    allowed_data_age_ms: float,
    check_session: bool = True,
) -> ValidationReport:
    allowed_age_ns = int(allowed_data_age_ms * NS_PER_MS)
    report = ValidationReport(total_events=len(events), valid_events=[])
    seen: set[tuple[Any, ...]] = set()
    last_ts: dict[str, int] = {}
    last_seq: dict[str, int] = {}
    last_quote_ts: dict[str, int] = {}

    for i, event in enumerate(events):
        key = (event.source, event.seq) if event.seq is not None else (event.event_type, event)
        if key in seen:
            report.issues.append(Issue(IssueKind.DUPLICATE, i, event.symbol, f"duplicate of {key}"))
            continue

        problem: tuple[IssueKind, str] | None = None
        if event.receive_ts is not None and event.receive_ts < event.event_ts:
            problem = (IssueKind.CLOCK_ANOMALY, "receive_ts earlier than event_ts")
        elif event.symbol in last_ts and event.event_ts < last_ts[event.symbol]:
            problem = (
                IssueKind.OUT_OF_ORDER,
                f"event_ts {ns_to_iso(event.event_ts)} before previous {ns_to_iso(last_ts[event.symbol])}",
            )
        elif event.seq is not None and event.source in last_seq and event.seq <= last_seq[event.source]:
            problem = (IssueKind.OUT_OF_ORDER, f"seq {event.seq} not after {last_seq[event.source]}")
        elif check_session and not in_regular_session(event.event_ts):
            problem = (IssueKind.OUT_OF_SESSION, f"{ns_to_iso(event.event_ts)} outside regular session")
        else:
            problem = _content_problems(event)

        if problem is not None:
            report.issues.append(Issue(problem[0], i, event.symbol, problem[1]))
            continue

        seen.add(key)
        last_ts[event.symbol] = event.event_ts
        if event.seq is not None:
            last_seq[event.source] = event.seq
        if isinstance(event, Quote):
            prev = last_quote_ts.get(event.symbol)
            if prev is not None and event.event_ts - prev > allowed_age_ns:
                gap_ms = (event.event_ts - prev) / NS_PER_MS
                report.issues.append(
                    Issue(IssueKind.STALE_GAP, i, event.symbol, f"{gap_ms:.0f} ms since previous quote")
                )
            last_quote_ts[event.symbol] = event.event_ts
        report.valid_events.append(event)
    return report
