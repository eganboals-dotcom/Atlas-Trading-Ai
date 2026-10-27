"""Alpaca market-data feed (read-only). This is the only module in Atlas allowed to use the network.

It calls exactly two endpoints, both HTTP GET on the market-data host (verified against
Alpaca's official docs on 2026-10-26):

    GET https://data.alpaca.markets/v2/stocks/quotes/latest?symbols=...&feed=iex
    GET https://data.alpaca.markets/v2/stocks/trades/latest?symbols=...&feed=iex

Response fields used: quotes ``bp, bs, ap, as, t``. Sizes are in shares; a price of 0
means no active bid or ask, and validation rejects it. Trades use ``p, s, t, i, c``.
Timestamps are RFC 3339 with nanosecond precision.

It has **no trading endpoint** and no way to send an order: Atlas's paper orders go only
to the internal ``PaperBroker``. Use API keys from an Alpaca *paper* account. Alpaca
gives paper accounts their own keys, separate from live ones.

Polling: latest quotes every ``poll_interval_ms`` and latest trades every
``trade_poll_every`` polls. The default is 90 calls/min, under the free plan's 200/min.
Only changed quotes and new trades are emitted. Each event keeps Alpaca's source
timestamp as ``event_ts``, and ``receive_ts`` is the local clock when the response
arrived. The free plan's real-time data is IEX only: IEX's own top of book, usually
wider than the national best bid/offer, which makes paper fills conservative.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime
from pathlib import Path
from typing import Any, Callable

from atlas.data.events import MarketEvent, Quote, TimestampKind, Trade
from atlas.paper.feed import FeedDisconnected, LiveFeed
from atlas.timeutils import MARKET_TZ, NS_PER_MS, NS_PER_S, datetime_to_ns

DATA_HOST = "https://data.alpaca.markets"
QUOTES_PATH = "/v2/stocks/quotes/latest"
TRADES_PATH = "/v2/stocks/trades/latest"
ALLOWED_PATHS = frozenset({QUOTES_PATH, TRADES_PATH})
KEY_ENV, SECRET_ENV = "APCA_API_KEY_ID", "APCA_API_SECRET_KEY"

Fetch = Callable[[str, dict[str, str]], dict[str, Any]]


class AlpacaAuthError(PermissionError):
    """Missing or rejected market-data credentials; the session must not start."""


@dataclass(frozen=True)
class AlpacaCredentials:
    key_id: str
    secret: str = field(repr=False)

    def __repr__(self) -> str:  # never print secrets
        return f"AlpacaCredentials(key_id={self.key_id[:4]}…)"


def load_credentials(env_file: Path | None = Path(".env")) -> AlpacaCredentials:
    """Read market-data keys from the environment, or from a local .env file (git-ignored)."""
    values = {KEY_ENV: os.environ.get(KEY_ENV, ""), SECRET_ENV: os.environ.get(SECRET_ENV, "")}
    if env_file is not None and env_file.is_file() and not all(values.values()):
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = (x.strip() for x in line.split("=", 1))
            if k in values and not values[k]:
                values[k] = v.strip('"').strip("'")
    if not all(values.values()):
        raise AlpacaAuthError(f"set {KEY_ENV} and {SECRET_ENV} (environment or .env) to your Alpaca paper-account "
                              "market-data keys")
    return AlpacaCredentials(values[KEY_ENV], values[SECRET_ENV])


def http_fetch(credentials: AlpacaCredentials, timeout_s: float = 5.0) -> Fetch:
    def fetch(path: str, params: dict[str, str]) -> dict[str, Any]:
        if path not in ALLOWED_PATHS:  # hard guard: only the two read-only market-data endpoints
            raise PermissionError(f"refusing non-market-data path {path!r}")
        url = f"{DATA_HOST}{path}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, method="GET", headers={
            "APCA-API-KEY-ID": credentials.key_id, "APCA-API-SECRET-KEY": credentials.secret,
            "Accept": "application/json", "User-Agent": "atlas-paper/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise AlpacaAuthError(f"Alpaca rejected the market-data credentials (HTTP {exc.code})") from exc
            raise FeedDisconnected(f"Alpaca HTTP {exc.code}" + (" (rate limited)" if exc.code == 429 else "")) \
                from exc
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise FeedDisconnected(f"Alpaca request failed: {exc}") from exc

    return fetch


def parse_rfc3339_ns(ts: str) -> int:
    """RFC 3339 timestamp with up to nanosecond precision -> epoch ns (exact, no float rounding)."""
    s = ts.strip().replace("Z", "+00:00")
    main, frac, offset = s, "", ""
    if "." in s:
        main, rest = s.split(".", 1)
        i = next((k for k, ch in enumerate(rest) if ch in "+-"), len(rest))
        frac, offset = rest[:i], rest[i:]
    else:
        i = max(s.rfind("+"), s.rfind("-"))
        if i > 10:
            main, offset = s[:i], s[i:]
    dt = datetime.fromisoformat(main + (offset or "+00:00"))
    return datetime_to_ns(dt) + int((frac + "000000000")[:9])


class AlpacaPollingFeed(LiveFeed):
    source = "alpaca-iex"

    def __init__(self, symbols: list[str], session_date: date, fetch: Fetch, feed: str = "iex",
                 poll_interval_ms: float = 1_000.0, trade_poll_every: int = 2, start: dtime = dtime(9, 30),
                 end: dtime = dtime(16, 0), wait_for_open: bool = True,
                 clock: Callable[[], int] = time.time_ns, sleep: Callable[[float], None] = time.sleep) -> None:
        self.symbols = tuple(symbols)
        self.session_date = session_date
        self._fetch = fetch
        self.feed = feed
        self.poll_ns = int(poll_interval_ms * NS_PER_MS)
        self.trade_every = max(1, trade_poll_every)
        self.start_ns = datetime_to_ns(datetime.combine(session_date, start, tzinfo=MARKET_TZ))
        self.end_ns = datetime_to_ns(datetime.combine(session_date, end, tzinfo=MARKET_TZ))
        self.wait_for_open = wait_for_open
        self._clock, self._sleep = clock, sleep
        self._buffer: list[MarketEvent] = []
        self._last_quote_t: dict[str, int] = {}
        self._last_trade_key: dict[str, tuple[int, Any]] = {}
        self._polls = 0
        self._next_poll = 0
        self._delivered = 0
        self._connected = False
        self.clock_offset_ms: float | None = None  # latest source timestamp minus local clock, at connect

    # -- LiveFeed ------------------------------------------------------------------------

    def connect(self) -> None:
        data = self._fetch(QUOTES_PATH, self._params())  # validates credentials; raises on auth failure
        self.connect_quotes = {s: {"bid": q.get("bp"), "ask": q.get("ap"), "t": q.get("t")}
                               for s, q in (data.get("quotes") or {}).items()}
        stamps = [parse_rfc3339_ns(q["t"]) for q in data.get("quotes", {}).values() if q.get("t")]
        if stamps:
            self.clock_offset_ms = (max(stamps) - self._clock()) / NS_PER_MS
        self._connected = True

    def reconnect(self) -> None:
        self._fetch(QUOTES_PATH, self._params())
        self._connected = True

    def next_event(self) -> MarketEvent | None:
        while not self._buffer:
            now = self._clock()
            if now >= self.end_ns:
                return None
            if now < self.start_ns and self.wait_for_open:
                self._sleep(min(1.0, (self.start_ns - now) / NS_PER_S))
                continue
            if now < self._next_poll:
                self._sleep((self._next_poll - now) / NS_PER_S)
                continue
            self._next_poll = now + self.poll_ns
            self._poll()
        self._delivered += 1
        return self._buffer.pop(0)

    def clock_ns(self) -> int:
        return self._clock()

    @property
    def position(self) -> int:
        return self._delivered

    def seek(self, position: int) -> None:
        """A live stream cannot be rewound: after a restart it simply continues from the present."""
        self._delivered = position

    # -- polling ---------------------------------------------------------------------------

    def _params(self) -> dict[str, str]:
        return {"symbols": ",".join(self.symbols), "feed": self.feed}

    def _poll(self) -> None:
        if not self._connected:
            raise FeedDisconnected("feed not connected")
        try:
            raw_quotes = self._fetch(QUOTES_PATH, self._params())
            raw_trades: Any = {}
            if self._polls % self.trade_every == 0:
                raw_trades = self._fetch(TRADES_PATH, self._params())
        except FeedDisconnected:
            self._connected = False
            raise
        self._polls += 1
        received = self._clock()
        try:
            quotes = raw_quotes.get("quotes") or {}
            trades = (raw_trades.get("trades") or {}) if raw_trades else {}
            events = self._parse(quotes, trades, received)
        except (KeyError, ValueError, TypeError, AttributeError) as exc:  # malformed response: transient
            raise FeedDisconnected(f"malformed Alpaca response: {exc!r}") from exc
        # One response = one arrival instant; keep source order within it, nudging receive_ts so the
        # local clock never runs backwards between events.
        events.sort(key=lambda e: e.event_ts)
        self._buffer.extend(dataclasses.replace(e, receive_ts=received + k) for k, e in enumerate(events))

    def _parse(self, quotes: dict[str, Any], trades: dict[str, Any], received: int) -> list[MarketEvent]:
        events: list[MarketEvent] = []
        for sym, q in quotes.items():
            if sym not in self.symbols or not q.get("t"):
                continue
            t = parse_rfc3339_ns(q["t"])
            if t <= self._last_quote_t.get(sym, -1):
                continue  # unchanged quote: nothing new happened
            self._last_quote_t[sym] = t
            events.append(Quote(symbol=sym, event_ts=t, receive_ts=received,
                                source=self.source, bid=float(q.get("bp") or 0.0), ask=float(q.get("ap") or 0.0),
                                bid_size=int(q.get("bs") or 0), ask_size=int(q.get("as") or 0),
                                timestamp_kind=TimestampKind.VENDOR, conditions=tuple(q.get("c") or ())))
        for sym, tr in trades.items():
            if sym not in self.symbols or not tr.get("t"):
                continue
            t = parse_rfc3339_ns(tr["t"])
            key = (t, tr.get("i"))
            prev = self._last_trade_key.get(sym)
            if prev is not None and (key == prev or t < prev[0]):
                continue  # same latest trade as last poll, or older
            self._last_trade_key[sym] = key
            events.append(Trade(symbol=sym, event_ts=t, receive_ts=received, source=self.source,
                                price=float(tr.get("p") or 0.0), size=int(tr.get("s") or 0),
                                timestamp_kind=TimestampKind.VENDOR, conditions=tuple(tr.get("c") or ())))
        return events
