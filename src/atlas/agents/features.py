"""Causal per-symbol features computed incrementally from what a strategy can see.

The engine updates features on every visible event (O(1)-ish work) and builds a
snapshot only when an agent decides. Only events already in the ``MarketView`` are
used, so features inherit the replay engine's no-lookahead guarantee.

Definitions:
* returns are mid-price log returns in basis points over fixed horizons, measured on
  the source clock of received quotes; ``nan`` until enough history exists;
* volatility is an exponentially time-weighted std of mid log returns, expressed in
  bps per sqrt(second), with short (10 s) and long (300 s) half-lives;
* trade signing uses the quote rule (at/above ask = buy, at/below bid = sell), then
  midpoint comparison, then the tick rule — a standard approximation, not ground truth.

Rule-based agents use these raw values (bps, ratios). Fitted normalization is only
needed for learned policies and will be fit on training data only (Phase 4).
"""

from __future__ import annotations

import bisect
import math
from collections import deque
from dataclasses import dataclass
from typing import Mapping

from atlas.environment.market_state import MarketView
from atlas.timeutils import NS_PER_MS, NS_PER_S

RETURN_HORIZONS_S: tuple[float, ...] = (0.5, 1.0, 5.0, 30.0)
FLOW_WINDOWS_S: tuple[float, ...] = (1.0, 5.0, 30.0)
_SHORT_HALFLIFE_S = 10.0
_LONG_HALFLIFE_S = 300.0
_HISTORY_S = max(RETURN_HORIZONS_S) + 1.0
NAN = float("nan")


@dataclass(frozen=True, slots=True)
class SymbolFeatures:
    symbol: str
    valid: bool  # a fresh, non-stale quote is visible
    mid: float
    bid: float
    ask: float
    spread_bps: float
    bid_size: int
    ask_size: int
    quote_age_ms: float
    l1_imbalance: float  # (bid_size - ask_size) / (bid_size + ask_size), in [-1, 1]
    returns_bps: Mapping[float, float]
    vol_bps: float
    vol_ratio: float  # short-horizon vol / long-horizon vol; 1.0 until both exist
    flow_imbalance: Mapping[float, float]  # signed / total trade volume, in [-1, 1]
    trade_volume: Mapping[float, int]

    @staticmethod
    def invalid(symbol: str) -> SymbolFeatures:
        empty_r = {h: NAN for h in RETURN_HORIZONS_S}
        return SymbolFeatures(
            symbol, False, NAN, NAN, NAN, NAN, 0, 0, NAN, 0.0, empty_r, NAN, 1.0,
            {w: 0.0 for w in FLOW_WINDOWS_S}, {w: 0 for w in FLOW_WINDOWS_S},
        )


class _SymbolState:
    __slots__ = (
        "quote_ts", "quote", "hist_ts", "hist_mid", "offset", "var_short", "var_long",
        "last_trade_ts", "last_trade_price", "last_sign", "flows",
    )

    def __init__(self) -> None:
        self.quote_ts = -1
        self.quote = None
        self.hist_ts: list[int] = []
        self.hist_mid: list[float] = []
        self.offset = 0
        self.var_short: float | None = None
        self.var_long: float | None = None
        self.last_trade_ts = -1
        self.last_trade_price: float | None = None
        self.last_sign = 0
        # per window: deque of (ts, signed_size, size) plus running sums [signed, total]
        self.flows = {w: (deque(), [0, 0]) for w in FLOW_WINDOWS_S}


class FeatureEngine:
    def __init__(self, symbols: list[str], stale_ns: int) -> None:
        self.symbols = list(symbols)
        self._stale_ns = stale_ns
        self._state = {s: _SymbolState() for s in symbols}

    def update(self, market: MarketView) -> None:
        for symbol in self.symbols:
            st = self._state[symbol]
            q = market.quote(symbol)
            if q is not None and q.event_ts > st.quote_ts:
                self._on_quote(st, q)
            t = market.last_trade(symbol)
            if t is not None and t.event_ts > st.last_trade_ts:
                self._on_trade(st, t)

    def _on_quote(self, st: _SymbolState, q) -> None:
        mid = q.mid
        if st.hist_ts and mid > 0 and st.hist_mid[-1] > 0:
            dt_s = max((q.event_ts - st.hist_ts[-1]) / NS_PER_S, 1e-6)
            r2_per_s = math.log(mid / st.hist_mid[-1]) ** 2 / dt_s
            a_s = 1.0 - math.exp(-dt_s * math.log(2) / _SHORT_HALFLIFE_S)
            a_l = 1.0 - math.exp(-dt_s * math.log(2) / _LONG_HALFLIFE_S)
            st.var_short = r2_per_s if st.var_short is None else st.var_short + a_s * (r2_per_s - st.var_short)
            st.var_long = r2_per_s if st.var_long is None else st.var_long + a_l * (r2_per_s - st.var_long)
        st.quote_ts = q.event_ts
        st.quote = q
        st.hist_ts.append(q.event_ts)
        st.hist_mid.append(mid)
        cutoff = q.event_ts - int(_HISTORY_S * NS_PER_S)
        # Keep one sample at or before the cutoff so the longest horizon stays answerable.
        while st.offset + 1 < len(st.hist_ts) and st.hist_ts[st.offset + 1] <= cutoff:
            st.offset += 1
        if st.offset > 4096:
            del st.hist_ts[: st.offset], st.hist_mid[: st.offset]
            st.offset = 0

    def _on_trade(self, st: _SymbolState, t) -> None:
        q = st.quote
        sign = 0
        if q is not None:
            if t.price >= q.ask:
                sign = 1
            elif t.price <= q.bid:
                sign = -1
            elif t.price != q.mid:
                sign = 1 if t.price > q.mid else -1
        if sign == 0 and st.last_trade_price is not None and t.price != st.last_trade_price:
            sign = 1 if t.price > st.last_trade_price else -1
        if sign == 0:
            sign = st.last_sign
        st.last_sign = sign
        st.last_trade_ts = t.event_ts
        st.last_trade_price = t.price
        for window, (dq, sums) in st.flows.items():
            dq.append((t.event_ts, sign * t.size, t.size))
            sums[0] += sign * t.size
            sums[1] += t.size
            self._expire(dq, sums, t.event_ts - int(window * NS_PER_S))

    @staticmethod
    def _expire(dq: deque, sums: list[int], cutoff: int) -> None:
        while dq and dq[0][0] <= cutoff:
            _, signed, size = dq.popleft()
            sums[0] -= signed
            sums[1] -= size

    def snapshot(self, now_ns: int) -> dict[str, SymbolFeatures]:
        return {s: self._snapshot_symbol(s, now_ns) for s in self.symbols}

    def _snapshot_symbol(self, symbol: str, now_ns: int) -> SymbolFeatures:
        st = self._state[symbol]
        q = st.quote
        if q is None:
            return SymbolFeatures.invalid(symbol)
        age_ns = now_ns - q.event_ts
        mid = q.mid
        returns: dict[float, float] = {}
        for h in RETURN_HORIZONS_S:
            target = q.event_ts - int(h * NS_PER_S)
            i = bisect.bisect_right(st.hist_ts, target, lo=st.offset) - 1
            if i >= st.offset and st.hist_mid[i] > 0:
                returns[h] = math.log(mid / st.hist_mid[i]) * 10_000.0
            else:
                returns[h] = NAN
        vol = NAN if st.var_short is None else math.sqrt(st.var_short) * 10_000.0
        ratio = 1.0
        if st.var_short is not None and st.var_long and st.var_long > 0:
            ratio = math.sqrt(st.var_short / st.var_long)
        flow: dict[float, float] = {}
        volume: dict[float, int] = {}
        for window, (dq, sums) in st.flows.items():
            self._expire(dq, sums, now_ns - int(window * NS_PER_S))
            volume[window] = sums[1]
            flow[window] = sums[0] / sums[1] if sums[1] > 0 else 0.0
        depth = q.bid_size + q.ask_size
        return SymbolFeatures(
            symbol=symbol,
            valid=age_ns <= self._stale_ns,
            mid=mid,
            bid=q.bid,
            ask=q.ask,
            spread_bps=q.spread_bps,
            bid_size=q.bid_size,
            ask_size=q.ask_size,
            quote_age_ms=age_ns / NS_PER_MS,
            l1_imbalance=(q.bid_size - q.ask_size) / depth if depth > 0 else 0.0,
            returns_bps=returns,
            vol_bps=vol,
            vol_ratio=ratio,
            flow_imbalance=flow,
            trade_volume=volume,
        )
