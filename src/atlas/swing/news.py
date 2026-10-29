"""News features for swing research: article counts per symbol and trading day, split at the open.

Each article is bucketed by its ``created_at`` time in New York:

* ``news_pre[t]``: from 16:00 on the previous trading day to 09:15 on day t (weekends and holidays included),
  so it is known before the 09:30 open of t. The 15 minutes before the open are left out as a margin.
* ``news_day[t]``: from 09:15 to 16:00 on day t; usable for decisions from the open of t+1 on.

``earn_pre`` / ``earn_day`` count the subset whose headline looks like an earnings report. The tagged counts
(``up``/``down``: an analyst upgrade or downgrade; ``beat``/``miss``: reported EPS above or below the estimate,
e.g. "Q4 EPS $1.46 Beats $1.37 Estimate") use only articles about one or two companies, so market roundups
("Top Upgrades, Downgrades For ...") don't count.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import polars as pl

from atlas.swing.data import Panel
from atlas.timeutils import MARKET_TZ, NS_PER_S

DEFAULT_DIR = Path("data/raw/alpaca_news")
PRE_OPEN_CUTOFF = time(9, 15)
CLOSE = time(16, 0)
EARNINGS = re.compile(r"\b(earnings|EPS|Q[1-4]|quarterly (results|report)|(first|second|third|fourth)[- ]quarter "
                      r"(results|profit|revenue|sales)|beats? estimates|miss(es)? estimates|tops estimates)\b",
                      re.IGNORECASE)
TAGS = {"up": re.compile(r"\bupgrades?\b", re.IGNORECASE),
        "down": re.compile(r"\bdowngrades?\b", re.IGNORECASE),
        "beat": re.compile(r"\beps\s+[-$()]*[\d.]+\)?\s+beats?\b", re.IGNORECASE),
        "miss": re.compile(r"\beps\s+[-$()]*[\d.]+\)?\s+miss(es)?\b", re.IGNORECASE)}
TAG_MAX_SYMBOLS = 2
FEATURES = ("news_pre", "news_day", "earn_pre", "earn_day", *(f"{t}_{part}" for t in TAGS for part in ("pre", "day")))
# Reported EPS against the analysts' estimate, both numbers in the headline: "Q4 Adj. EPS $1.46 Beats $1.37
# Estimate", "Q2 EPS $(0.05) Misses $0.02 Estimate"; "EPS $0.63, Inline" counts as no surprise.
_NUM = r"(-?\$?\s?\(?-?\$?\s?[\d.]+\)?)"
SURPRISE = re.compile(r"\bEPS\s+" + _NUM + r"\s+(Beats?|Miss(?:es)?)\s+" + _NUM + r"\s+Estimate", re.IGNORECASE)
INLINE = re.compile(r"\bEPS\s+" + _NUM + r",?\s+(?:In-?Line|Inline)\b", re.IGNORECASE)
SURPRISE_FRESH_DAYS = 63  # trading days, about the 92 calendar days of the filing-based surprise


def _num(s: str) -> float:
    neg = "(" in s or "-" in s
    v = float(re.sub(r"[^\d.]", "", s) or "nan")
    return -v if neg else v


def parse_surprise(head: str) -> tuple[float, float] | None:
    """(reported EPS, estimate) from a Benzinga-style earnings headline, or None if it has no such numbers."""
    m = SURPRISE.search(head)
    if m:
        try:
            return _num(m.group(1)), _num(m.group(3))
        except ValueError:
            return None
    m = INLINE.search(head)
    if m:
        try:
            v = _num(m.group(1))
        except ValueError:
            return None
        return v, v
    return None


def surprise_features(p: Panel, articles: pl.DataFrame, price: np.ndarray,
                      fresh_days: int = SURPRISE_FRESH_DAYS) -> dict[str, np.ndarray]:
    """``news_surprise[t, j]``: the latest (reported - estimated EPS) / share price for symbol j known at the open
    of t, for ``fresh_days`` trading days after it became known (NaN otherwise). A headline before 09:15 counts
    from that day's open, later ones from the next open (see ``bucket``). Only headlines naming exactly one of the
    panel's symbols count. ``price`` is the unadjusted close (EPS is per share as reported then), and the close
    before the headline is used. Several headlines on one day are averaged."""
    out = np.full(p.close.shape, np.nan)
    col = {s: j for j, s in enumerate(p.symbols)}
    days = np.array(p.dates, dtype="datetime64[D]")
    events: dict[tuple[int, int], list[float]] = {}
    for ns, syms, head in articles.select("created_ns", "symbols", "headline").iter_rows():
        js = [col[s] for s in syms if s in col]
        if len(js) != 1 or "EPS" not in head.upper():
            continue
        got = parse_surprise(head)
        if got is None:
            continue
        i, part = bucket(ns, p.dates, days)
        i += part == "day"
        if not 1 <= i < len(p.dates):
            continue
        px = price[i - 1, js[0]]
        if not np.isfinite(px) or px <= 0 or not all(np.isfinite(got)):
            continue
        events.setdefault((i, js[0]), []).append((got[0] - got[1]) / px)
    for (i, j), vals in sorted(events.items()):
        out[i:i + fresh_days, j] = float(np.mean(vals))  # later events overwrite: sorted by day
    return {"news_surprise": out}


def news_rows(articles: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Raw Alpaca articles -> the fields kept: id, time, symbols and headline (no bodies, no summaries)."""
    from atlas.feeds.alpaca import parse_rfc3339_ns

    return [{"id": int(a["id"]), "created_ns": parse_rfc3339_ns(a["created_at"]), "symbols": list(a.get("symbols") or []),
             "headline": str(a.get("headline") or ""), "source": str(a.get("source") or "")} for a in articles]


def save_news(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema={"id": pl.Int64, "created_ns": pl.Int64, "symbols": pl.List(pl.Utf8),
                               "headline": pl.Utf8, "source": pl.Utf8}).write_parquet(path)


def load_news(news_dir: Path) -> pl.DataFrame:
    files = sorted(Path(news_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no news files in {news_dir}")
    return pl.concat([pl.read_parquet(f) for f in files]).unique("id")


def bucket(created_ns: int, dates: tuple, days: np.ndarray | None = None) -> tuple[int, str]:
    """(trading-day index, "pre" or "day") for an article time; the index is len(dates) past the last day.
    ``days`` is ``dates`` as datetime64[D], passed in when bucketing many articles."""
    et = datetime.fromtimestamp(created_ns / NS_PER_S, UTC).astimezone(MARKET_TZ)
    d, tm = et.date(), et.time()
    days = np.array(dates, dtype="datetime64[D]") if days is None else days
    i = int(np.searchsorted(days, np.datetime64(d), side="left"))
    if i < len(dates) and dates[i] == d:  # a trading day
        if tm < PRE_OPEN_CUTOFF:
            return i, "pre"
        if tm < CLOSE:
            return i, "day"
        return i + 1, "pre"
    return i, "pre"  # weekend or holiday: before the next trading day's open


def news_features(p: Panel, articles: pl.DataFrame) -> dict[str, np.ndarray]:
    out = {k: np.zeros(p.close.shape) for k in FEATURES}
    col = {s: j for j, s in enumerate(p.symbols)}
    days = np.array(p.dates, dtype="datetime64[D]")
    for ns, syms, head in articles.select("created_ns", "symbols", "headline").iter_rows():
        js = [col[s] for s in syms if s in col]
        if not js:
            continue
        i, part = bucket(ns, p.dates, days)
        if i >= len(p.dates):
            continue  # after the panel's last day (e.g. inside a holdout that was cut off)
        earn = bool(EARNINGS.search(head))
        tags = [t for t, rx in TAGS.items() if rx.search(head)] if len(syms) <= TAG_MAX_SYMBOLS else []
        for j in js:
            out[f"news_{part}"][i, j] += 1
            out[f"earn_{part}"][i, j] += earn
            for t in tags:
                out[f"{t}_{part}"][i, j] += 1
    return out


def write_manifest(news_dir: Path, meta: dict[str, Any]) -> None:
    (Path(news_dir) / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
