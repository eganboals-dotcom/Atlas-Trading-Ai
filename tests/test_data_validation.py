from __future__ import annotations

import math

from helpers import data_config, quote, trade, ts

from atlas.data.synthetic import generate_events
from atlas.data.validation import IssueKind, validate_events


def kinds(report) -> list[IssueKind]:
    return [i.kind for i in report.issues]


def test_clean_stream_passes() -> None:
    events = [quote(0, seq=0), trade(10, seq=1), quote(100, seq=2)]
    report = validate_events(events, allowed_data_age_ms=1_000)
    assert report.is_clean
    assert report.valid_events == events


def test_duplicate_is_dropped() -> None:
    q = quote(0, seq=0)
    report = validate_events([q, q, quote(50, seq=1)], 1_000)
    assert kinds(report) == [IssueKind.DUPLICATE]
    assert len(report.valid_events) == 2


def test_duplicate_without_seq_detected_by_content() -> None:
    q = quote(0)
    report = validate_events([q, q], 1_000)
    assert kinds(report) == [IssueKind.DUPLICATE]


def test_out_of_order_event_time_is_dropped_not_reordered() -> None:
    first, late = quote(100), quote(50)
    report = validate_events([first, late], 1_000)
    assert kinds(report) == [IssueKind.OUT_OF_ORDER]
    assert report.valid_events == [first]


def test_out_of_order_sequence_number() -> None:
    report = validate_events([quote(0, seq=5), quote(10, seq=4)], 1_000)
    assert kinds(report) == [IssueKind.OUT_OF_ORDER]


def test_missing_and_nan_fields() -> None:
    report = validate_events(
        [quote(0, bid=None), quote(1, ask=math.nan), trade(2, size=None), quote(3)], 1_000
    )
    assert kinds(report) == [IssueKind.MISSING] * 3
    assert len(report.valid_events) == 1


def test_impossible_values() -> None:
    events = [
        quote(0, bid=100.05, ask=100.00),  # crossed
        quote(1, bid=-1.0),
        quote(2, ask_size=0),
        trade(3, price=0.0),
        trade(4, size=-5),
    ]
    report = validate_events(events, 1_000)
    assert kinds(report) == [IssueKind.IMPOSSIBLE] * 5
    assert report.valid_events == []


def test_locked_quote_is_allowed() -> None:
    assert validate_events([quote(0, bid=100.0, ask=100.0)], 1_000).is_clean


def test_stale_gap_flagged_but_event_kept() -> None:
    events = [quote(0), quote(5_000)]
    report = validate_events(events, allowed_data_age_ms=2_000)
    assert kinds(report) == [IssueKind.STALE_GAP]
    assert report.valid_events == events


def test_receive_before_event_is_clock_anomaly() -> None:
    bad = quote(100, recv_ms=50)
    report = validate_events([bad], 1_000)
    assert kinds(report) == [IssueKind.CLOCK_ANOMALY]


def test_out_of_session_rejected() -> None:
    pre_open = quote(-60_000)  # one minute before the open
    report = validate_events([pre_open], 1_000)
    assert kinds(report) == [IssueKind.OUT_OF_SESSION]
    assert validate_events([pre_open], 1_000, check_session=False).is_clean


def test_synthetic_clean_data_has_no_dropping_issues() -> None:
    report = validate_events(generate_events(data_config()), 2_000)
    assert not [i for i in report.issues if i.drops_event]
    assert report.total_events > 500


def test_injected_anomalies_are_all_detected() -> None:
    cfg = data_config(
        synthetic={
            "seed": 5,
            "initial_prices": {"SPY": 580.0, "QQQ": 500.0},
            "anomalies": {
                "duplicate_rate": 0.02,
                "out_of_order_rate": 0.01,
                "crossed_quote_rate": 0.01,
                "missing_field_rate": 0.01,
                "stale_gaps": 1,
                "stale_gap_s": 5,
            },
        }
    )
    counts = validate_events(generate_events(cfg), 2_000).counts
    for kind in ("duplicate", "out_of_order", "impossible", "missing", "stale_gap"):
        assert counts.get(kind, 0) > 0, f"expected {kind} issues, got {counts}"


def test_validated_stream_is_chronological_per_symbol() -> None:
    cfg = data_config(
        synthetic={"seed": 9, "anomalies": {"out_of_order_rate": 0.05, "duplicate_rate": 0.05}}
    )
    valid = validate_events(generate_events(cfg), 2_000).valid_events
    last: dict[str, int] = {}
    for e in valid:
        assert e.event_ts >= last.get(e.symbol, ts(-10**9))
        last[e.symbol] = e.event_ts
