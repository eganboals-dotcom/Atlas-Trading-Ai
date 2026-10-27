"""Build real-market training sessions from Alpaca's historical data, as the live feed would see them.

The paper session polls Alpaca: the latest quote every ~1 s and the latest trade every ~2 s.
Historical data has every tick, so training on it directly would teach strategies that need
data the bots never get live. ``polled_events`` replays the history through the same polling
rule: at each poll time it takes the latest quote (and, every other poll, the latest trade)
at or before that instant, and emits it only if it is new, exactly like ``AlpacaPollingFeed``.

``download_window`` fetches one symbol-window through ``atlas.feeds.alpaca.iter_history`` (the
only network code) and streams it, so millions of raw ticks never sit in memory at once.
``publish_dataset_session`` writes the session to Parquet and registers it in the dataset's
``manifest.json``, so ``ParquetProvider`` and walk-forward use it directly.
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Iterable
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from atlas import __version__
from atlas.data.events import MarketEvent, Quote, TimestampKind, Trade
from atlas.data.storage import SCHEMA_VERSION, write_session
from atlas.timeutils import MARKET_TZ, NS_PER_MS, NS_PER_S, datetime_to_ns

# Notable market days, so the sample always includes crashes, squeezes and shocks.
STRESS_DAYS = [
    date(2018, 2, 5),   # "Volmageddon": VIX spike, S&P -4.1%
    date(2020, 3, 16),  # COVID crash, S&P -12%
    date(2020, 3, 24),  # COVID relief rally, S&P +9.4%
    date(2022, 6, 13),  # bear market, CPI shock, S&P -3.9%
    date(2022, 11, 10),  # soft-CPI rally, Nasdaq +7.4%
    date(2024, 8, 5),   # yen carry-trade unwind
    date(2025, 4, 7),   # tariff crash, extreme intraday swings
    date(2025, 4, 9),   # tariff-pause rally, S&P +9.5%
]


def polled_events(quotes: Iterable[dict[str, Any]], trades: Iterable[dict[str, Any]], symbol: str,
                  start_ns: int, end_ns: int, source: str, poll_ns: int = NS_PER_S, trade_every: int = 2,
                  ) -> list[tuple[int, MarketEvent]]:
    """(poll index, event) pairs for one symbol, as a poller at ``start_ns + k * poll_ns`` would see them.

    Both inputs are raw Alpaca records (``t`` RFC 3339, quotes ``bp/bs/ap/as/c``, trades ``p/s/i/c``),
    oldest first.
    """
    from atlas.feeds.alpaca import parse_rfc3339_ns

    out: list[tuple[int, MarketEvent]] = []
    n_polls = (end_ns - start_ns) // poll_ns

    def sample(records: Iterable[dict[str, Any]], every: int, make) -> None:
        latest: dict[str, Any] | None = None
        latest_t = emitted_t = -1
        k = 0
        it = iter(records)
        nxt = next(it, None)
        while k < n_polls:
            poll_t = start_ns + k * poll_ns
            while nxt is not None and (t := parse_rfc3339_ns(nxt["t"])) <= poll_t:
                latest, latest_t = nxt, t
                nxt = next(it, None)
            if k % every == 0 and latest is not None and latest_t > emitted_t:
                emitted_t = latest_t
                out.append((k, make(latest, latest_t)))
            k += 1

    sample(quotes, 1, lambda q, t: Quote(
        symbol=symbol, event_ts=t, source=source, bid=float(q.get("bp") or 0.0), ask=float(q.get("ap") or 0.0),
        bid_size=int(q.get("bs") or 0), ask_size=int(q.get("as") or 0), timestamp_kind=TimestampKind.VENDOR,
        conditions=tuple(q.get("c") or ())))
    sample(trades, trade_every, lambda r, t: Trade(
        symbol=symbol, event_ts=t, source=source, price=float(r.get("p") or 0.0), size=int(r.get("s") or 0),
        timestamp_kind=TimestampKind.VENDOR, conditions=tuple(r.get("c") or ())))
    return out


def merge_polls(per_symbol: list[list[tuple[int, MarketEvent]]], start_ns: int, poll_ns: int = NS_PER_S,
                latency_ms: float = 80.0) -> list[MarketEvent]:
    """Interleave symbols by poll; one poll = one arrival (``receive_ts`` nudged so it never runs backwards)."""
    import dataclasses

    by_poll: dict[int, list[MarketEvent]] = {}
    for pairs in per_symbol:
        for k, e in pairs:
            by_poll.setdefault(k, []).append(e)
    events: list[MarketEvent] = []
    for k in sorted(by_poll):
        received = start_ns + k * poll_ns + int(latency_ms * NS_PER_MS)
        batch = sorted(by_poll[k], key=lambda e: e.event_ts)
        events.extend(dataclasses.replace(e, receive_ts=received + i) for i, e in enumerate(batch))
    return events


def window_bounds(day: date, start: time, minutes: int) -> tuple[int, int, str, str]:
    a = datetime.combine(day, start, tzinfo=MARKET_TZ)
    b = a + timedelta(minutes=minutes)
    iso = lambda d: d.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return datetime_to_ns(a), datetime_to_ns(b), iso(a), iso(b)


def to_shares(pairs: list[tuple[int, MarketEvent]]) -> tuple[list[tuple[int, MarketEvent]], str]:
    """Alpaca's SIP history reports quote sizes in 100-share round lots until late 2025, shares after.

    Lots give a median quote size of a handful; shares give hundreds. Normalize to shares, which
    is what the live IEX feed (and every agent) uses.
    """
    import dataclasses
    import statistics

    sizes = [x for _, e in pairs if isinstance(e, Quote) for x in (e.bid_size or 0, e.ask_size or 0)]
    if not sizes or statistics.median(sizes) >= 40:
        return pairs, "shares"
    return [(k, dataclasses.replace(e, bid_size=(e.bid_size or 0) * 100, ask_size=(e.ask_size or 0) * 100)
             if isinstance(e, Quote) else e) for k, e in pairs], "round lots (x100)"


def download_window(fetch, symbols: list[str], day: date, start: time, minutes: int, feed: str,
                    min_interval_s: float = 0.35) -> tuple[list[MarketEvent], dict[str, str]]:
    """Polled events for one window, and the quote-size unit found per symbol."""
    from atlas.feeds.alpaca import HISTORY_QUOTES_PATH, HISTORY_TRADES_PATH, iter_history

    a_ns, b_ns, a_iso, b_iso = window_bounds(day, start, minutes)
    per_symbol, units = [], {}
    for sym in symbols:
        quotes = iter_history(fetch, HISTORY_QUOTES_PATH, sym, a_iso, b_iso, feed, min_interval_s=min_interval_s)
        trades = list(iter_history(fetch, HISTORY_TRADES_PATH, sym, a_iso, b_iso, feed,
                                   min_interval_s=min_interval_s))
        pairs, units[sym] = to_shares(polled_events(quotes, trades, sym, a_ns, b_ns, f"alpaca-{feed}"))
        per_symbol.append(pairs)
    return merge_polls(per_symbol, a_ns), units


def sample_days(first: date, last: date, n_random: int, seed: int, exclude: set[date] = frozenset()) -> list[date]:
    """Weekdays spread evenly over [first, last] (one random pick per equal slice), plus the stress days."""
    rng = random.Random(seed)
    span = (last - first).days
    days = {d for d in STRESS_DAYS if first <= d <= last}
    for i in range(n_random):
        lo, hi = first + timedelta(days=span * i // n_random), first + timedelta(days=span * (i + 1) // n_random - 1)
        for _ in range(50):
            d = lo + timedelta(days=rng.randint(0, max(0, (hi - lo).days)))
            if d.weekday() < 5 and d not in days and d not in exclude:
                days.add(d)
                break
    return sorted(days)


def random_window_start(rng: random.Random, minutes: int) -> time:
    """Opens and closes are over-weighted (they behave differently); otherwise any time of day."""
    r = rng.random()
    if r < 0.2:
        return time(9, 30)
    if r < 0.35:
        return (datetime(2000, 1, 1, 16, 0) - timedelta(minutes=minutes)).time()
    latest = 390 - minutes
    m = rng.randint(0, latest // 5) * 5
    return (datetime(2000, 1, 1, 9, 30) + timedelta(minutes=m)).time()


def publish_dataset_session(processed_dir: Path, dataset: str, day: date, events: list[MarketEvent],
                            symbols: list[str], meta: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{day.isoformat()}.parquet"
    sha = write_session(events, processed_dir / file_name)
    manifest_path = processed_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {
        "schema_version": SCHEMA_VERSION, "dataset": dataset, "created_by": f"atlas {__version__}", **meta,
        "raw_files": [], "sessions": []}
    quotes = sum(isinstance(e, Quote) for e in events)
    entry = {"date": day.isoformat(), "file": file_name, "sha256": sha, "rows": len(events), "quotes": quotes,
             "trades": len(events) - quotes, "symbols": sorted(symbols), **extra}
    manifest["sessions"] = sorted([s for s in manifest["sessions"] if s["date"] != entry["date"]] + [entry],
                                  key=lambda s: s["date"])
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, manifest_path)
    return entry
