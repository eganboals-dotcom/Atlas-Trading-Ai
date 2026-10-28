"""Slow strategy families: hold for minutes to hours, never overnight.

The fast families react to second-scale signals, and on real data the spread eats them:
evolution learned to barely trade. These families act on minute-scale structure instead,
so each trade has room to earn more than it costs. Each one is a published or widely
traded intraday idea, as a hypothesis to test, not a claim that it works:

* ``intraday_momentum`` — the morning's move tends to continue into the afternoon
  (documented for SPY: the first half-hour predicts the last half-hour);
* ``opening_range`` — buy a breakout above the first minutes' high (opening-range breakout);
* ``trend_follow`` — ride a fast/slow moving-average uptrend for minutes to hours;
* ``pullback`` — buy a sharp dip (from a recent high, or below the session's average price)
  and sell the rebound;
* ``smart_hold`` — be long like Hold, but step aside when the day turns dangerous
  (a drawdown from the session high, or a volatility spike), and come back once it calms.

Every family is long-only and time-based: it samples its own 1-minute price history from
whatever quotes arrive (1-second live polls or 1-minute training bars), so the same genome
behaves the same in both. Shared parameters:

* ``size_frac`` — position value as a fraction of equity, per symbol (clipped by limits);
* ``stop_bps`` — exit when the price falls this far below the entry;
* ``max_hold_min`` — exit after this long regardless of signal;
* ``flatten_min`` — no position is held into the last minutes of the session.
"""

from __future__ import annotations

import math
from collections import deque
from datetime import UTC, datetime, time
from typing import ClassVar

from atlas.agents.base import Agent, Observation, TargetPositions
from atlas.agents.features import SymbolFeatures
from atlas.agents.genome import ParamSpec
from atlas.timeutils import MARKET_TZ, NS_PER_S, datetime_to_ns

NS_PER_MIN = 60 * NS_PER_S
SESSION_MIN = 390  # 09:30-16:00 ET
MAX_ENTRY_SPREAD_BPS = 20.0  # fixed guard against broken or stale quotes (single stocks often quote 2-8 bps wide);
#                              training spreads are modeled, so evolution cannot tune this
_DAY_NS = 86_400 * NS_PER_S
_PENDING_NS = 30 * NS_PER_S  # an entry order that has not filled by then is treated as missed

SLOW_COMMON_SPACE: dict[str, ParamSpec] = {
    "size_frac": ParamSpec("float", 0.1, 0.5),
    "stop_bps": ParamSpec("float", 10.0, 400.0, log=True),
    "max_hold_min": ParamSpec("float", 5.0, 390.0, log=True),
    "flatten_min": ParamSpec("int", 1, 15),
}
SLOW_COMMON_DEFAULT = {"size_frac": 0.3, "stop_bps": 60.0, "max_hold_min": 120.0, "flatten_min": 5}


class _Book:
    """Per-symbol minute history and position bookkeeping for one session."""

    __slots__ = ("open_px", "minutes", "high", "twap_sum", "twap_n", "entry_ts", "entry_px", "last_exit_min",
                 "entries", "range_high", "ema_fast", "ema_slow", "peak_since_entry", "pending", "exiting", "saw_open")

    def __init__(self) -> None:
        self.open_px: float | None = None
        self.minutes: deque[tuple[int, float]] = deque(maxlen=400)  # (minute since open, mid)
        self.high = -math.inf
        self.twap_sum = 0.0
        self.twap_n = 0
        self.entry_ts: int | None = None
        self.entry_px = math.nan
        self.last_exit_min = -10_000
        self.entries = 0
        self.range_high = -math.inf
        self.ema_fast: float | None = None
        self.ema_slow: float | None = None
        self.peak_since_entry = -math.inf
        self.pending = 0  # shares of an entry order not filled yet
        self.exiting = False  # an exit was decided; keep the target at 0 until flat
        self.saw_open = False  # first sample within 2 minutes of 09:30 (not, e.g., after a mid-day restart)

    def price_at(self, minute: int) -> float | None:
        """Mid sampled at the first minute >= ``minute``, if seen."""
        for m, px in self.minutes:
            if m >= minute:
                return px
        return None

    def back(self, n: int) -> float | None:
        """Mid sampled ``n`` minutes before the latest sample (nearest earlier sample)."""
        if not self.minutes:
            return None
        target = self.minutes[-1][0] - n
        best = None
        for m, px in self.minutes:
            if m > target:
                break
            best = px
        return best

    def vol_bps(self, n: int) -> float:
        """Std of the last ``n`` 1-minute log returns, in bps; nan without enough history."""
        pts = list(self.minutes)[-(n + 1):]
        if len(pts) < max(5, n // 2):
            return math.nan
        rets = [math.log(b / a) * 1e4 for (_, a), (_, b) in zip(pts, pts[1:]) if a > 0 and b > 0]
        if len(rets) < 3:
            return math.nan
        mu = sum(rets) / len(rets)
        return math.sqrt(sum((r - mu) ** 2 for r in rets) / (len(rets) - 1))


class SlowAgent(Agent):
    """Shared session clock, minute sampling, sizing and exits; families supply enter/exit signals."""

    def reset(self) -> None:
        self._books: dict[str, _Book] = {}
        self._day: int | None = None
        self._open_ns = 0

    # -- session clock ----------------------------------------------------------------

    def _minute(self, now_ns: int) -> float:
        day = now_ns // _DAY_NS  # 09:30 ET is always on the same UTC date
        if day != self._day:
            d = datetime.fromtimestamp(now_ns / NS_PER_S, UTC).astimezone(MARKET_TZ).date()
            self._open_ns = datetime_to_ns(datetime.combine(d, time(9, 30), tzinfo=MARKET_TZ))
            self._day = day
            self._books = {}  # a new session: no state carries over
        return (now_ns - self._open_ns) / NS_PER_MIN

    def _update(self, book: _Book, f: SymbolFeatures, minute: float) -> bool:
        """Record this decision's mid; True when it starts a new minute sample."""
        mid = f.mid
        if book.open_px is None and minute >= 0:
            book.open_px, book.saw_open = mid, minute <= 2
        book.high = max(book.high, mid)
        m = int(minute)
        if book.minutes and book.minutes[-1][0] >= m:
            return False
        book.minutes.append((m, mid))
        book.twap_sum += mid
        book.twap_n += 1
        self.on_minute(book, m, mid)
        return True

    # -- family hooks -------------------------------------------------------------------

    def on_minute(self, book: _Book, minute: int, mid: float) -> None:  # noqa: B027 - optional hook
        """Called once per new 1-minute sample (indicators update here)."""

    def should_enter(self, book: _Book, f: SymbolFeatures, minute: float) -> bool:
        raise NotImplementedError

    def should_exit(self, book: _Book, f: SymbolFeatures, minute: float) -> bool:
        return False

    # -- decision -------------------------------------------------------------------------

    def act(self, obs: Observation) -> TargetPositions:
        g = self.genome
        minute = self._minute(obs.now_ns)
        last_call = SESSION_MIN - g["flatten_min"]
        targets: TargetPositions = {}
        for symbol, f in obs.features.items():
            book = self._books.setdefault(symbol, _Book())
            held = obs.qty(symbol)
            if f.valid and minute >= 0:
                self._update(book, f, minute)
            if held:
                book.pending = 0
                if book.exiting:  # the exit order has not filled yet
                    targets[symbol] = 0
                    continue
                if book.entry_ts is None:  # e.g. a position restored after a restart
                    book.entry_ts, book.entry_px = obs.now_ns, f.mid if f.valid else math.nan
                if f.valid:
                    book.peak_since_entry = max(book.peak_since_entry, f.mid)
                stopped = f.valid and f.mid <= book.entry_px * (1 - g["stop_bps"] / 1e4)
                timed_out = obs.now_ns - book.entry_ts >= g["max_hold_min"] * NS_PER_MIN
                if minute >= last_call or stopped or timed_out or (f.valid and self.should_exit(book, f, minute)):
                    targets[symbol] = 0
                    book.exiting, book.last_exit_min = True, int(minute)
                else:
                    targets[symbol] = held
                continue
            book.exiting = False
            if book.pending and book.entry_ts is not None and obs.now_ns - book.entry_ts < _PENDING_NS:
                targets[symbol] = book.pending  # entry order still working: don't re-decide
                continue
            book.pending, book.entry_ts = 0, None
            targets[symbol] = 0
            if (not f.valid or minute < 0 or minute >= last_call - 1 or book.open_px is None
                    or f.spread_bps > MAX_ENTRY_SPREAD_BPS or not self.should_enter(book, f, minute)):
                continue
            size = self.size_for(f, obs.equity, g["size_frac"])
            if size > 0:
                targets[symbol] = book.pending = size
                book.entry_ts, book.entry_px, book.peak_since_entry = obs.now_ns, f.ask, f.mid
                book.entries += 1
        return targets


def _ret_bps(a: float | None, b: float | None) -> float:
    return math.nan if not a or not b else math.log(b / a) * 1e4


class IntradayMomentumAgent(SlowAgent):
    """If the market is up enough by ``signal_min`` after the open, buy at ``entry_min`` and ride it."""

    family: ClassVar[str] = "intraday_momentum"
    param_space: ClassVar[dict[str, ParamSpec]] = {
        **SLOW_COMMON_SPACE,
        "signal_min": ParamSpec("choice", choices=(15, 30, 60, 90, 120, 180)),
        "entry_min": ParamSpec("int", 15, 370),
        "threshold_bps": ParamSpec("float", 0.0, 80.0),
    }
    default_genome: ClassVar[dict[str, float]] = {
        **SLOW_COMMON_DEFAULT, "signal_min": 30, "entry_min": 360, "threshold_bps": 10.0, "max_hold_min": 60.0,
    }

    def should_enter(self, book: _Book, f: SymbolFeatures, minute: float) -> bool:
        g = self.genome
        start = max(g["entry_min"], g["signal_min"])
        if book.entries or not book.saw_open or not start <= minute < start + 10:
            return False
        r = _ret_bps(book.open_px, book.price_at(int(g["signal_min"])))
        return not math.isnan(r) and r >= g["threshold_bps"]


class OpeningRangeAgent(SlowAgent):
    """Buy when the price breaks above the high of the first ``range_min`` minutes; take profit at a target."""

    family: ClassVar[str] = "opening_range"
    param_space: ClassVar[dict[str, ParamSpec]] = {
        **SLOW_COMMON_SPACE,
        "range_min": ParamSpec("choice", choices=(5, 15, 30, 45, 60)),
        "buffer_bps": ParamSpec("float", 0.0, 30.0),
        "target_bps": ParamSpec("float", 10.0, 300.0, log=True),
        "last_entry_min": ParamSpec("int", 60, 360),
    }
    default_genome: ClassVar[dict[str, float]] = {
        **SLOW_COMMON_DEFAULT, "range_min": 30, "buffer_bps": 3.0, "target_bps": 50.0, "last_entry_min": 240,
    }

    def on_minute(self, book: _Book, minute: int, mid: float) -> None:
        if minute < self.genome["range_min"]:
            book.range_high = max(book.range_high, mid)

    def should_enter(self, book: _Book, f: SymbolFeatures, minute: float) -> bool:
        g = self.genome
        return (not book.entries and book.saw_open and g["range_min"] <= minute <= g["last_entry_min"]
                and math.isfinite(book.range_high) and f.mid > book.range_high * (1 + g["buffer_bps"] / 1e4))

    def should_exit(self, book: _Book, f: SymbolFeatures, minute: float) -> bool:
        return f.mid >= book.entry_px * (1 + self.genome["target_bps"] / 1e4)


class TrendFollowAgent(SlowAgent):
    """Long while a fast moving average of 1-minute prices is above a slow one; out when it crosses back."""

    family: ClassVar[str] = "trend_follow"
    param_space: ClassVar[dict[str, ParamSpec]] = {
        **SLOW_COMMON_SPACE,
        "fast_min": ParamSpec("float", 3.0, 60.0, log=True),
        "slow_ratio": ParamSpec("float", 2.0, 8.0),
        "entry_bps": ParamSpec("float", 0.0, 20.0),
        "exit_bps": ParamSpec("float", 0.0, 20.0),
        "cooldown_min": ParamSpec("int", 0, 60),
    }
    default_genome: ClassVar[dict[str, float]] = {
        **SLOW_COMMON_DEFAULT, "fast_min": 10.0, "slow_ratio": 4.0, "entry_bps": 3.0, "exit_bps": 2.0,
        "cooldown_min": 10, "max_hold_min": 240.0,
    }

    def on_minute(self, book: _Book, minute: int, mid: float) -> None:
        g = self.genome
        a_f = 2 / (g["fast_min"] + 1)
        a_s = 2 / (g["fast_min"] * g["slow_ratio"] + 1)
        book.ema_fast = mid if book.ema_fast is None else book.ema_fast + a_f * (mid - book.ema_fast)
        book.ema_slow = mid if book.ema_slow is None else book.ema_slow + a_s * (mid - book.ema_slow)

    def _gap_bps(self, book: _Book) -> float:
        return math.nan if not book.ema_slow else (book.ema_fast / book.ema_slow - 1) * 1e4

    def should_enter(self, book: _Book, f: SymbolFeatures, minute: float) -> bool:
        g = self.genome
        warm = min(120.0, g["fast_min"] * g["slow_ratio"])
        return (len(book.minutes) >= warm and minute - book.last_exit_min >= g["cooldown_min"]
                and self._gap_bps(book) >= g["entry_bps"])

    def should_exit(self, book: _Book, f: SymbolFeatures, minute: float) -> bool:
        return self._gap_bps(book) <= -self.genome["exit_bps"]


class PullbackAgent(SlowAgent):
    """Buy a sharp dip and sell the rebound.

    ``mode`` 0: the dip is measured from the highest price of the last ``lookback_min`` minutes;
    ``mode`` 1: from the session's average price so far (a time-weighted VWAP stand-in).
    ``trend_filter`` 1 only buys dips on days that are up since the open.
    """

    family: ClassVar[str] = "pullback"
    param_space: ClassVar[dict[str, ParamSpec]] = {
        **SLOW_COMMON_SPACE,
        "mode": ParamSpec("choice", choices=(0, 1)),
        "lookback_min": ParamSpec("choice", choices=(5, 15, 30, 60, 120)),
        "drop_bps": ParamSpec("float", 5.0, 200.0, log=True),
        "target_bps": ParamSpec("float", 5.0, 200.0, log=True),
        "trend_filter": ParamSpec("choice", choices=(0, 1)),
        "cooldown_min": ParamSpec("int", 0, 60),
    }
    default_genome: ClassVar[dict[str, float]] = {
        **SLOW_COMMON_DEFAULT, "mode": 0, "lookback_min": 30, "drop_bps": 30.0, "target_bps": 20.0,
        "trend_filter": 0, "cooldown_min": 15, "max_hold_min": 60.0,
    }

    def _anchor(self, book: _Book) -> float | None:
        if self.genome["mode"] == 1:
            return book.twap_sum / book.twap_n if book.twap_n >= 15 else None
        n = int(self.genome["lookback_min"])
        recent = [px for m, px in book.minutes if book.minutes and m >= book.minutes[-1][0] - n]
        return max(recent) if len(recent) >= max(3, n // 2) else None

    def should_enter(self, book: _Book, f: SymbolFeatures, minute: float) -> bool:
        g = self.genome
        if minute - book.last_exit_min < g["cooldown_min"]:
            return False
        if g["trend_filter"] and not (book.saw_open and f.mid > book.open_px):
            return False
        anchor = self._anchor(book)
        return anchor is not None and _ret_bps(anchor, f.mid) <= -g["drop_bps"]

    def should_exit(self, book: _Book, f: SymbolFeatures, minute: float) -> bool:
        return f.mid >= book.entry_px * (1 + self.genome["target_bps"] / 1e4)


class SmartHoldAgent(SlowAgent):
    """Hold, but step aside on dangerous days.

    Long from ``entry_min`` after the open. Out when the price is ``dd_bps`` below the session
    high or 1-minute volatility over ``vol_window_min`` exceeds ``vol_limit_bps``. Back in after
    ``calm_min`` minutes once volatility is under the limit again.
    """

    family: ClassVar[str] = "smart_hold"
    param_space: ClassVar[dict[str, ParamSpec]] = {
        **SLOW_COMMON_SPACE,
        "entry_min": ParamSpec("int", 0, 60),
        "dd_bps": ParamSpec("float", 20.0, 400.0, log=True),
        "vol_window_min": ParamSpec("choice", choices=(5, 15, 30, 60)),
        "vol_limit_bps": ParamSpec("float", 2.0, 40.0, log=True),
        "calm_min": ParamSpec("int", 5, 180),
    }
    default_genome: ClassVar[dict[str, float]] = {
        **SLOW_COMMON_DEFAULT, "size_frac": 0.45, "stop_bps": 150.0, "max_hold_min": 390.0, "entry_min": 1,
        "dd_bps": 80.0, "vol_window_min": 15, "vol_limit_bps": 8.0, "calm_min": 30,
    }

    def _dangerous(self, book: _Book, f: SymbolFeatures) -> bool:
        g = self.genome
        vol = book.vol_bps(int(g["vol_window_min"]))
        return f.mid <= book.high * (1 - g["dd_bps"] / 1e4) or (not math.isnan(vol) and vol > g["vol_limit_bps"])

    def should_enter(self, book: _Book, f: SymbolFeatures, minute: float) -> bool:
        g = self.genome
        if minute < g["entry_min"]:
            return False
        if book.entries and minute - book.last_exit_min < g["calm_min"]:
            return False
        vol = book.vol_bps(int(g["vol_window_min"]))
        calm = math.isnan(vol) or vol <= g["vol_limit_bps"]
        return calm and (not book.entries or f.mid > book.high * (1 - g["dd_bps"] / 1e4))

    def should_exit(self, book: _Book, f: SymbolFeatures, minute: float) -> bool:
        return self._dangerous(book, f)


SLOW_FAMILIES = (IntradayMomentumAgent, OpeningRangeAgent, TrendFollowAgent, PullbackAgent, SmartHoldAgent)
