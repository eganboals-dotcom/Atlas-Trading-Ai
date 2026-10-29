"""The news desk's daily run: fetch new headlines, let each reader note the ones it hasn't seen, write the
briefing for the next open. Run it after the close (the forward tracker scores the notes) or before the open
(to read the briefing). The network calls are injected, so this module never touches the network itself.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

import polars as pl
from pydantic import Field

from atlas.config import StrictModel
from atlas.news.desk import KeywordReader, Reader, load_notes, save_notes
from atlas.swing.news import PRE_OPEN_CUTOFF, load_news, news_rows, save_news
from atlas.timeutils import MARKET_TZ, NS_PER_S, datetime_to_ns


class DeskConfig(StrictModel):
    start_date: date  # headlines from this day on; the AI reader is never run on anything older
    symbols: list[str]
    news_dir: Path = Path("data/raw/alpaca_news/desk")
    notes_dir: Path = Path("data/processed/news_notes")
    out_dir: Path = Path("artifacts/reports/news_desk")
    readers: list[str] = Field(default_factory=lambda: ["keyword"])
    local_model: str = "gemma3:12b"  # the "local" reader: a free model on this PC, run by Ollama
    claude_model: str = "claude-haiku-5-5"  # the "claude" reader (optional; uses paid API credits)
    batch_size: int = 25
    max_articles_per_run: int = 2000  # cap per run for each AI reader (time for local, cost for claude)


AI_READERS = ("local", "claude")


FetchNews = Callable[[list[str], str, str], Iterable[dict[str, Any]]]


def _stamp(now: datetime) -> str:
    return now.strftime("%Y%m%dT%H%M%SZ")


def fetch_new(cfg: DeskConfig, fetch: FetchNews, now: datetime) -> int:
    """Download headlines published since the last stored one (or the start date) until 20 minutes ago."""
    since = datetime.combine(cfg.start_date, time(0), tzinfo=MARKET_TZ)
    if any(cfg.news_dir.glob("*.parquet")):
        last = load_news(cfg.news_dir)["created_ns"].max()
        since = max(since, datetime.fromtimestamp(last / NS_PER_S + 1, UTC))
    end = now - timedelta(minutes=20)
    if end <= since:
        return 0
    rows = news_rows(fetch(cfg.symbols, since.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                           end.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")))
    if rows:
        save_news(rows, cfg.news_dir / f"{_stamp(now)}.parquet")
    return len(rows)


def _seen_path(cfg: DeskConfig, reader: str) -> Path:
    return cfg.notes_dir / "_seen" / f"{reader}.parquet"


def run_readers(cfg: DeskConfig, readers: dict[str, Reader], now: datetime) -> dict[str, int]:
    """Each reader reads the stored headlines it hasn't seen (oldest first, up to the cap) and saves notes."""
    if not any(cfg.news_dir.glob("*.parquet")):
        return {}
    start_ns = datetime_to_ns(datetime.combine(cfg.start_date, time(0), tzinfo=MARKET_TZ))
    arts = load_news(cfg.news_dir).filter(pl.col("created_ns") >= start_ns).sort("created_ns")
    counts = {}
    for name, reader in readers.items():
        seen_file = _seen_path(cfg, name)
        seen = set(pl.read_parquet(seen_file)["id"].to_list()) if seen_file.exists() else set()
        new = arts.filter(~pl.col("id").is_in(list(seen))).head(cfg.max_articles_per_run)
        if new.height == 0:
            counts[name] = 0
            continue
        notes = reader.read(new)
        save_notes(notes, cfg.notes_dir / name / f"{_stamp(now)}.parquet")
        seen_file.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"id": sorted(seen | set(new["id"].to_list()))}, schema={"id": pl.Int64}).write_parquet(seen_file)
        counts[name] = notes.height
    return counts


def next_open(now: datetime) -> date:
    """The next weekday whose pre-open cutoff (09:15 New York) is still ahead (holidays aren't known here)."""
    et = now.astimezone(MARKET_TZ)
    d = et.date() if et.time() < PRE_OPEN_CUTOFF else et.date() + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def briefing_window(open_day: date) -> tuple[int, int]:
    """[previous weekday 09:15, open_day 09:15) in epoch ns: what the briefing for open_day covers."""
    prev = open_day - timedelta(days=1)
    while prev.weekday() >= 5:
        prev -= timedelta(days=1)
    return (datetime_to_ns(datetime.combine(prev, PRE_OPEN_CUTOFF, tzinfo=MARKET_TZ)),
            datetime_to_ns(datetime.combine(open_day, PRE_OPEN_CUTOFF, tzinfo=MARKET_TZ)))


def render(cfg: DeskConfig, open_day: date, headlines: pl.DataFrame) -> str:
    lo, hi = briefing_window(open_day)
    L = [f"# News desk briefing for the open of {open_day}", "",
         "Notes on headlines published since the previous open's 09:15 cutoff, by reader. Score = sum of "
         "direction x importance. Research only: no bot trades on this in the live session.", ""]
    heads = dict(zip(headlines["id"].to_list(), headlines["headline"].to_list())) if headlines.height else {}
    for name in cfg.readers:
        notes = load_notes(cfg.notes_dir / name).filter(pl.col("created_ns").is_between(lo, hi, closed="left"))
        L += [f"## Reader: {name}", ""]
        if notes.height == 0:
            L += ["No notes in this window.", ""]
            continue
        by = (notes.with_columns((pl.col("direction") * pl.col("importance")).alias("s"))
              .group_by("symbol").agg(pl.col("s").sum().alias("score"), pl.len().alias("n"),
                                      pl.col("article_id").sort_by("importance", descending=True).first().alias("top"))
              .sort(pl.col("score").abs(), descending=True))
        L += ["| Company | Score | Notes | Most important headline |", "|---|---|---|---|"]
        for sym, score, n, top in by.head(20).iter_rows():
            L.append(f"| {sym} | {score:+.2f} | {n} | {heads.get(top, '').replace('|', '/')[:110]} |")
        L.append("")
    return "\n".join(L) + "\n"


def run_desk(cfg: DeskConfig, fetch: FetchNews | None, classifiers: dict[str, Callable[[], Any] | str],
             now: datetime | None = None, log: Callable[[str], None] = print) -> Path:
    """``classifiers``: for each AI reader, a factory for its classify function, or why it can't run today
    (it is then skipped and reads the missed headlines on a later run)."""
    now = now or datetime.now(UTC)
    if fetch is not None:
        log(f"news: {fetch_new(cfg, fetch, now)} new headlines")
    readers: dict[str, Reader] = {}
    for name in cfg.readers:
        if name == "keyword":
            readers[name] = KeywordReader()
        elif name in AI_READERS:
            factory = classifiers.get(name, "not set up")
            if isinstance(factory, str):
                log(f"{name} reader skipped: {factory}")
                continue
            from atlas.news.llm_reader import LlmReader

            readers[name] = LlmReader(factory(), name, cfg.batch_size, set(cfg.symbols))
        else:
            raise ValueError(f"unknown reader {name!r}")
    for name, n in run_readers(cfg, readers, now).items():
        log(f"reader {name}: {n} new notes")
    day = next_open(now)
    heads = load_news(cfg.news_dir) if any(cfg.news_dir.glob("*.parquet")) else pl.DataFrame()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    text = render(cfg, day, heads)
    (cfg.out_dir / f"briefing_{day}.md").write_text(text, encoding="utf-8")
    (cfg.out_dir / "latest.md").write_text(text, encoding="utf-8")
    return cfg.out_dir / "latest.md"
