"""News desk: readers, notes and the pre-open briefing.

A **reader** reads headlines and writes **notes**: one per (article, company) it thinks matters, with a
direction (-1 bad news .. +1 good news), an importance (0..1) and a category. Readers know nothing about
trading. The **briefing** turns notes into what the trading bots may know at each open: for day t, every
note on an article published since the previous open (from 09:15 on day t-1 to 09:15 on day t; weekends
and holidays roll forward), using the same clock as ``atlas.swing.news.bucket``. Bots see only the briefing,
so a reader can be swapped (keywords, an AI model) without touching them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import polars as pl

from atlas.swing.data import Panel
from atlas.swing.news import EARNINGS, TAG_MAX_SYMBOLS, TAGS, bucket

NOTE_SCHEMA = {"article_id": pl.Int64, "created_ns": pl.Int64, "symbol": pl.Utf8, "reader": pl.Utf8,
               "direction": pl.Float64, "importance": pl.Float64, "category": pl.Utf8}


def empty_notes() -> pl.DataFrame:
    return pl.DataFrame(schema=NOTE_SCHEMA)


class Reader(Protocol):
    name: str

    def read(self, articles: pl.DataFrame) -> pl.DataFrame:
        """Notes (``NOTE_SCHEMA``) for the given articles (columns id, created_ns, symbols, headline)."""
        ...


class KeywordReader:
    """Reader #1: fixed keyword rules. Analyst upgrades/downgrades and EPS beats/misses get a direction;
    other earnings headlines are noted with no direction (attention only). Only articles about one or two
    companies count, so market roundups don't."""

    name = "keyword"
    IMPORTANCE = {"beat": 1.0, "miss": 1.0, "up": 0.6, "down": 0.6}
    SIGN = {"beat": 1.0, "miss": -1.0, "up": 1.0, "down": -1.0}

    def read(self, articles: pl.DataFrame) -> pl.DataFrame:
        rows = []
        for aid, ns, syms, head in articles.select("id", "created_ns", "symbols", "headline").iter_rows():
            if not syms or len(syms) > TAG_MAX_SYMBOLS:
                continue
            tags = [t for t, rx in TAGS.items() if rx.search(head)]
            if tags:
                direction = float(np.clip(sum(self.SIGN[t] for t in tags), -1, 1))
                importance = max(self.IMPORTANCE[t] for t in tags)
                category = "earnings" if {"beat", "miss"} & set(tags) else "analyst"
            elif EARNINGS.search(head):
                direction, importance, category = 0.0, 0.3, "earnings"
            else:
                continue
            rows += [(aid, ns, s, self.name, direction, importance, category) for s in syms]
        return pl.DataFrame(rows, schema=NOTE_SCHEMA, orient="row") if rows else empty_notes()


def save_notes(notes: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    notes.write_parquet(path)


def load_notes(notes_dir: Path) -> pl.DataFrame:
    files = sorted(Path(notes_dir).glob("*.parquet"))
    if not files:
        return empty_notes()
    return pl.concat([pl.read_parquet(f) for f in files]).unique(["article_id", "symbol"], keep="last")


def briefing_features(p: Panel, notes: pl.DataFrame) -> dict[str, np.ndarray]:
    """(days, symbols) arrays of what is known at each open: ``brief_score`` (sum of direction x importance),
    ``brief_count`` (notes) and ``brief_attention`` (sum of importance)."""
    out = {k: np.zeros(p.close.shape) for k in ("brief_score", "brief_count", "brief_attention")}
    col = {s: j for j, s in enumerate(p.symbols)}
    days = np.array(p.dates, dtype="datetime64[D]")
    for ns, sym, d, imp in notes.select("created_ns", "symbol", "direction", "importance").iter_rows():
        j = col.get(sym)
        if j is None:
            continue
        i, part = bucket(ns, p.dates, days)
        i += part == "day"  # published during a session: known from the next open
        if i >= len(p.dates):
            continue
        out["brief_score"][i, j] += d * imp
        out["brief_count"][i, j] += 1
        out["brief_attention"][i, j] += imp
    return out


def render_briefing(notes: pl.DataFrame, p: Panel, day_index: int, top: int = 15) -> str:
    """The briefing for one open as Markdown: one line per company with notes, strongest first."""
    f = briefing_features(p, notes)
    i = day_index
    rows = [(p.symbols[j], f["brief_score"][i, j], int(f["brief_count"][i, j]))
            for j in range(len(p.symbols)) if f["brief_count"][i, j] > 0]
    rows.sort(key=lambda r: -abs(r[1]))
    L = [f"## Briefing for the open of {p.dates[i]}", ""]
    if not rows:
        return "\n".join(L + ["No notes since the previous open.", ""])
    L += ["| Company | Score | Notes |", "|---|---|---|"]
    L += [f"| {s} | {sc:+.2f} | {n} |" for s, sc, n in rows[:top]]
    return "\n".join(L) + "\n"
