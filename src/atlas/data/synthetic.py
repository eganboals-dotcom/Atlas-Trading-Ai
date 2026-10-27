"""Deterministic synthetic Level 1 quote/trade generator (no API keys, no network).

Synthetic data exists to test plumbing — chronology, validation, accounting, and
execution. It carries no information about real markets, and nothing measured on it
says anything about strategy profitability.
"""

from __future__ import annotations

import dataclasses
import math
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from atlas.config import AnomalyConfig, DataConfig
from atlas.data.events import MarketEvent, Quote, TimestampKind, Trade
from atlas.timeutils import NS_PER_MS, NS_PER_S, NS_PER_US, datetime_to_ns


def session_start_ns(cfg: DataConfig) -> int:
    return datetime_to_ns(datetime.combine(cfg.session_date, cfg.start_time, tzinfo=ZoneInfo(cfg.timezone)))


def _price(ticks: int, tick_size: float) -> float:
    return round(ticks * tick_size, 8)


def generate_events(cfg: DataConfig) -> list[MarketEvent]:
    """Generate a session of events in source (``event_ts``) order, then inject anomalies."""
    syn = cfg.synthetic
    rng = np.random.default_rng(syn.seed)
    start = session_start_ns(cfg)
    end = start + int(cfg.duration_s * NS_PER_S)
    events: list[MarketEvent] = []

    for symbol in cfg.symbols:  # config order is fixed, so generation is deterministic
        mid = syn.initial_prices.get(symbol, syn.default_initial_price)
        t = start
        while True:
            dt_ns = max(NS_PER_US, int(rng.exponential(syn.mean_quote_interval_ms) * NS_PER_MS))
            t += dt_ns
            if t >= end:
                break
            sigma = syn.volatility_bps_per_sqrt_s * 1e-4 * math.sqrt(dt_ns / NS_PER_S)
            mid *= math.exp(sigma * rng.standard_normal())
            spread_ticks = int(rng.integers(syn.min_spread_ticks, syn.max_spread_ticks + 1))
            bid_ticks = math.floor(mid / syn.tick_size - spread_ticks / 2)
            bid = _price(bid_ticks, syn.tick_size)
            ask = _price(bid_ticks + spread_ticks, syn.tick_size)
            events.append(
                Quote(
                    symbol=symbol,
                    event_ts=t,
                    receive_ts=t + _feed_delay_ns(rng, syn.feed_latency_ms, syn.feed_jitter_ms),
                    source=cfg.feed,
                    bid=bid,
                    ask=ask,
                    bid_size=int(rng.integers(syn.min_size, syn.max_size + 1)),
                    ask_size=int(rng.integers(syn.min_size, syn.max_size + 1)),
                    timestamp_kind=TimestampKind.SYNTHETIC,
                )
            )
            if rng.random() < syn.trade_probability:
                trade_ts = t + NS_PER_US
                buyer_initiated = bool(rng.random() < 0.5)
                events.append(
                    Trade(
                        symbol=symbol,
                        event_ts=trade_ts,
                        receive_ts=trade_ts + _feed_delay_ns(rng, syn.feed_latency_ms, syn.feed_jitter_ms),
                        source=cfg.feed,
                        price=ask if buyer_initiated else bid,
                        size=int(rng.integers(1, syn.min_size + 1)),
                        timestamp_kind=TimestampKind.SYNTHETIC,
                    )
                )

    # Source order: by event time, ties broken by symbol order then type (stable sort).
    events.sort(key=lambda e: e.event_ts)
    events = [dataclasses.replace(e, seq=i) for i, e in enumerate(events)]
    return inject_anomalies(events, syn.anomalies, seed=syn.seed + 1)


def _feed_delay_ns(rng: np.random.Generator, latency_ms: float, jitter_ms: float) -> int:
    return int((latency_ms + abs(rng.normal(0.0, jitter_ms)) if jitter_ms > 0 else latency_ms) * NS_PER_MS)


def inject_anomalies(events: list[MarketEvent], cfg: AnomalyConfig, seed: int) -> list[MarketEvent]:
    """Inject defects that validation must catch. With all rates at zero this is a no-op."""
    if not events:
        return events
    rng = np.random.default_rng(seed)
    out = list(events)

    for _ in range(cfg.stale_gaps):
        victim = out[int(rng.integers(len(out)))]
        gap_end = victim.event_ts + int(cfg.stale_gap_s * NS_PER_S)
        out = [e for e in out if not (e.symbol == victim.symbol and victim.event_ts < e.event_ts < gap_end)]

    result: list[MarketEvent] = []
    for event in out:
        if isinstance(event, Quote):
            if rng.random() < cfg.missing_field_rate:
                event = dataclasses.replace(event, bid=None)
            elif rng.random() < cfg.crossed_quote_rate:
                event = dataclasses.replace(event, bid=event.ask, ask=event.bid)
        result.append(event)
        if rng.random() < cfg.duplicate_rate:
            result.append(event)

    if cfg.out_of_order_rate > 0:
        last_index: dict[str, int] = {}
        for i in range(len(result)):
            sym = result[i].symbol
            j = last_index.get(sym)
            if j is not None and rng.random() < cfg.out_of_order_rate:
                result[i], result[j] = result[j], result[i]
            last_index[sym] = i
    return result
