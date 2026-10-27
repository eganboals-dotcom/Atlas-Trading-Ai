"""Normalized on-disk event format (one Parquet file per session) and conversions.

Schema version 1 — one row per event, in delivered order:

    event_type (str: quote|trade), symbol, event_ts (int64 ns UTC), receive_ts (int64, nullable),
    source, seq (int64, nullable), timestamp_kind, venue (nullable), conditions ('|'-joined),
    bid, ask (float, nullable), bid_size, ask_size (int64, nullable),
    price (float, nullable), size (int64, nullable)

Processed files keep every event, including defective ones; validation runs at load
time so that the same rules apply to every data source.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import polars as pl

from atlas.data.events import MarketEvent, Quote, TimestampKind, Trade

SCHEMA_VERSION = 1
SCHEMA: dict[str, Any] = {
    "event_type": pl.Utf8,
    "symbol": pl.Utf8,
    "event_ts": pl.Int64,
    "receive_ts": pl.Int64,
    "source": pl.Utf8,
    "seq": pl.Int64,
    "timestamp_kind": pl.Utf8,
    "venue": pl.Utf8,
    "conditions": pl.Utf8,
    "bid": pl.Float64,
    "ask": pl.Float64,
    "bid_size": pl.Int64,
    "ask_size": pl.Int64,
    "price": pl.Float64,
    "size": pl.Int64,
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def events_to_frame(events: list[MarketEvent]) -> pl.DataFrame:
    rows = []
    for e in events:
        row: dict[str, Any] = {
            "event_type": str(e.event_type), "symbol": e.symbol, "event_ts": e.event_ts,
            "receive_ts": e.receive_ts, "source": e.source, "seq": e.seq,
            "timestamp_kind": str(e.timestamp_kind), "venue": e.venue, "conditions": "|".join(e.conditions),
            "bid": None, "ask": None, "bid_size": None, "ask_size": None, "price": None, "size": None,
        }
        if isinstance(e, Quote):
            row.update(bid=e.bid, ask=e.ask, bid_size=e.bid_size, ask_size=e.ask_size)
        else:
            row.update(price=e.price, size=e.size)
        rows.append(row)
    return pl.DataFrame(rows, schema=SCHEMA)


def _num(v: Any) -> Any:
    return None if v is None else v


def frame_to_events(df: pl.DataFrame) -> list[MarketEvent]:
    missing = set(SCHEMA) - set(df.columns)
    if missing:
        raise ValueError(f"event frame missing columns: {sorted(missing)}")
    out: list[MarketEvent] = []
    for r in df.select(list(SCHEMA)).iter_rows(named=True):
        common = dict(
            symbol=r["symbol"], event_ts=r["event_ts"], receive_ts=r["receive_ts"], source=r["source"],
            seq=r["seq"], timestamp_kind=TimestampKind(r["timestamp_kind"]), venue=r["venue"],
            conditions=tuple(c for c in (r["conditions"] or "").split("|") if c),
        )
        if r["event_type"] == "quote":
            out.append(Quote(bid=_num(r["bid"]), ask=_num(r["ask"]), bid_size=_num(r["bid_size"]),
                             ask_size=_num(r["ask_size"]), **common))
        elif r["event_type"] == "trade":
            out.append(Trade(price=_num(r["price"]), size=_num(r["size"]), **common))
        else:
            raise ValueError(f"unknown event_type {r['event_type']!r}")
    return out


def write_session(events_or_frame: list[MarketEvent] | pl.DataFrame, path: Path) -> str:
    df = events_or_frame if isinstance(events_or_frame, pl.DataFrame) else events_to_frame(events_or_frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.select(list(SCHEMA)).cast(SCHEMA).write_parquet(path, statistics=False)
    return file_sha256(path)


def read_session(path: Path, expected_sha256: str | None = None) -> list[MarketEvent]:
    """Read a session file, refusing it if its content hash no longer matches the manifest."""
    if expected_sha256 is not None:
        actual = file_sha256(path)
        if actual != expected_sha256:
            raise ValueError(f"{path}: sha256 {actual} does not match manifest {expected_sha256}; "
                             "processed data must be immutable")
    return frame_to_events(pl.read_parquet(path))
