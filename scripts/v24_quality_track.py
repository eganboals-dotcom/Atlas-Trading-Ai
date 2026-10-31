"""Forward tracker for earnings quality (see configs/quality_track_v24.yaml, which holds the frozen rule and its verdict).

Reads the 300 companies' daily bars, which the 300-company tracker downloads first. When a quarter ends it downloads their SEC
filings (read-only) and records the next book, then scores each book against Hold and writes
artifacts/reports/quality_track/tracker.md. Research only: no order is placed anywhere.

    .venv/Scripts/python.exe scripts/v24_quality_track.py
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import yaml

import v23_filings as f
from atlas.config import load_config
from atlas.feeds import sec
from atlas.swing.backtest import run as run_book, summary
from atlas.swing.data import Panel
from atlas.swing.families import HOLD, exposures
from atlas.swing.fundamentals import SCHEMA, extract, fundamental_features
from atlas.swing.walkforward import SwingConfig

CONFIG = Path("configs/quality_track_v24.yaml")
ET = ZoneInfo("America/New_York")
BOOKS = ("top", "bottom", "profitable")
LONG = 60  # annual figures and Sharpe are shown only after this many days, as in the other trackers


@dataclass(frozen=True)
class Market:
    p: Panel  # row 0 is the session before first_day, and a gone company keeps its last close
    cash: np.ndarray
    traded: np.ndarray  # False where a company had no bar that day
    spy: np.ndarray  # SPY's daily return, dividends included: a yardstick in the report, and no part of the verdict


def universe(rule: dict) -> list[str]:
    return list(load_config(Path(rule["universe_config"]), SwingConfig).symbols)


def fingerprint(rule: dict, symbols: list[str]) -> str:
    code = [inspect.getsource(x) for x in (f.filing_scores, f.latest_two, f.top)]
    frozen = dict(rule=rule, symbols=symbols, filings=[f.cfg["max_age_days"], f.cfg["prior_year_days"]], code=code)
    return hashlib.sha256(json.dumps(frozen, sort_keys=True).encode("utf-8")).hexdigest()


def check_state(cfg: dict, symbols: list[str], state_path: Path) -> dict:
    """Refuses a changed rule."""
    fp = fingerprint(cfg["rule"], symbols)
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else dict(fingerprint=fp, frozen=str(date.today()), books=[])
    if state["fingerprint"] != fp:
        sys.exit("refused: the frozen rule in configs/quality_track_v24.yaml, its companies or its scoring code has changed")
    return state


def market(cfg: dict, symbols: list[str]) -> Market:
    """Every session of the cash ETF from the one before first_day. load_panel would stop the calendar when a company goes."""
    rule = cfg["rule"]
    bars = Path(cfg["bars_dir"])
    first = date.fromisoformat(rule["first_day"])
    bil = pl.read_parquet(bars / f"{rule['cash_symbol']}.parquet").sort("date")
    days = bil["date"].to_list()
    a = max(i for i, d in enumerate(days) if d < first)
    dates, px = days[a:], bil["close"].to_numpy()
    cash = np.r_[0.0, px[a + 1:] / px[a:-1] - 1]
    calendar = pl.DataFrame(dict(date=dates), schema={"date": pl.Date})
    opens, closes = np.empty((len(dates), len(symbols))), np.empty((len(dates), len(symbols)))
    traded = np.zeros((len(dates), len(symbols)), dtype=bool)
    for j, s in enumerate(symbols):
        row = calendar.join(pl.read_parquet(bars / f"{s}.parquet").select("date", "open", "close"), on="date", how="left")
        closes[:, j] = row["close"].forward_fill().to_numpy()
        opens[:, j] = row["open"].fill_null(row["close"].forward_fill()).to_numpy()
        traded[:, j] = row["close"].is_not_null().to_numpy()
    none = np.zeros_like(closes)
    spy = calendar.join(pl.read_parquet(Path(cfg["spy_dir"]) / "SPY.parquet").select("date", "close"), on="date", how="left")["close"].forward_fill().to_numpy()
    return Market(Panel(tuple(dates), tuple(symbols), opens, closes, closes, closes, none), cash, traded, np.r_[0.0, spy[1:] / spy[:-1] - 1])


def due(books: list[dict], sessions: tuple[date, ...], every: int) -> list[date]:
    """Last close before each book not yet chosen: the session before first_day, then each book's last session."""
    return [sessions[k * every] for k in range(len(books), (len(sessions) - 1) // every + 1)]


def refresh(cfg: dict, symbols: list[str]) -> None:
    """Every company's filings. One failure raises before the next book is chosen, so it waits for the next run."""
    out = Path(cfg["filings_dir"])
    out.mkdir(parents=True, exist_ok=True)
    get = sec.http_get(sec.user_agent(Path(".env")))
    ciks = sec.ciks(get)
    for sym in symbols:
        cik = ciks.get(sym, sec.DELISTED_CIKS.get(sym))
        if cik is None:
            print(f"{sym}: not in EDGAR's ticker list, keeps the filings already stored", flush=True)
            continue
        facts = [sec.company_facts(get, c) for c in (*sec.EARLIER_CIKS.get(sym, ()), cik)]
        pl.concat([x for g in facts for x in (extract(g), f.extract_extra(g))]).unique(maintain_order=True).write_parquet(out / f"{sym}.parquet")


def choose(cfg: dict, m: Market, cut: date) -> dict:
    """The book held from the session after cut, from filings filed on or before cut."""
    rule = cfg["rule"]
    symbols, d = m.p.symbols, cut + timedelta(days=1)
    n = len(symbols)
    day = Panel((d,), symbols, *(np.zeros((1, n)) for _ in range(5)))
    gpa = fundamental_features(day, Path(cfg["filings_dir"]), np.zeros((1, n)))["fund_gpa"][0]
    live = m.traded[m.p.dates.index(cut)]
    score, income = np.full(n, np.nan), np.full(n, np.nan)
    for j, s in enumerate(symbols):
        path = Path(cfg["filings_dir"]) / f"{s}.parquet"
        rows = pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=SCHEMA)
        q = f.filing_scores(rows, d, bool(np.isfinite(gpa[j]))).get("earnings_quality")
        if q is not None and live[j]:
            score[j], income[j] = q, f.latest_two(rows, "net_income", d)[1]
    eligible = int(np.isfinite(score).sum())
    if eligible < rule["min_eligible"]:
        everyone = list(symbols)
        return dict(cut=str(cut), eligible=eligible, top=everyone, bottom=everyone, profitable=everyone)
    names = lambda idx: [symbols[j] for j in idx]
    return dict(cut=str(cut), eligible=eligible, top=names(f.top(score, rule["top_share"])), bottom=names(f.top(score, rule["top_share"], highest=False)),
                profitable=names(f.top(np.where(income > 0, score, np.nan), rule["top_share"])))


def book_exposure(p: Panel, books: list[dict], key: str, every: int) -> np.ndarray:
    n = len(p.symbols)
    at = {s: j for j, s in enumerate(p.symbols)}
    x = np.zeros((len(p.dates), n))
    for k, b in enumerate(books):
        x[1 + k * every: 1 + (k + 1) * every, [at[s] for s in b[key]]] = n / len(b[key])
    return x


def score(cfg: dict, m: Market, books: list[dict]) -> dict[str, np.ndarray]:
    """Daily returns of Hold and each book, through the last session that has a chosen book. Each is sold at its last close."""
    rule, p = cfg["rule"], m.p
    end = min(len(p.dates), 1 + len(books) * rule["every"])
    if end <= 1:
        return {}
    go = lambda x: run_book(p, x, x, 1, end, rule["cost_bps"], None, m.cash).returns
    x = exposures(p, *HOLD)
    out = dict(Hold=go(x[0]), SPY=m.spy[1:end])
    return out | {k: go(book_exposure(p, books, k, rule["every"])) for k in BOOKS}


def gap_interval(d: np.ndarray, rule: dict) -> tuple[float, float]:
    """95% block bootstrap interval of the mean daily gap."""
    rng = np.random.default_rng(rule["seed"])
    block, count = rule["block"], int(np.ceil(len(d) / rule["block"]))
    starts = rng.integers(0, len(d) - block + 1, (rule["bootstrap_samples"], count))
    ids = (starts[:, :, None] + np.arange(block)).reshape(len(starts), -1)[:, : len(d)]
    lo, hi = np.percentile(d[ids].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def verdict(rule: dict, r: dict[str, np.ndarray], rf: np.ndarray) -> str:
    n = rule["review_sessions"]
    days = len(r["top"]) if r else 0
    if days < n:
        return f"Not reviewed yet: {days} of {n} sessions. Nothing before then is a verdict."
    top, hold, bottom = (summary(r[k][:n], rf[:n]) for k in ("top", "Hold", "bottom"))
    lo = gap_interval(r["top"][:n] - r["Hold"][:n], rule)[0]
    won = top["sharpe"] > hold["sharpe"] and top["ann_return"] >= hold["ann_return"] and top["ann_return"] > bottom["ann_return"] and lo > 0
    return f"Review after {n} sessions: {'PASS' if won else 'FAIL'} (interval's low end {lo * 252:+.1%} a year)."


def report(cfg: dict, state: dict, m: Market, r: dict[str, np.ndarray], waiting: str | None) -> str:
    rule, p = cfg["rule"], m.p
    rf = m.cash[1: 1 + len(r["top"])] if r else m.cash[:0]
    lines = ["# Earnings quality: forward tracker", "", f"Frozen {state['frozen']}, first day {rule['first_day']}. Research only: no order is placed anywhere, and holdings are not advice.", "",
             f"**{verdict(rule, r, rf)}**", ""]
    if waiting:
        lines += [f"- {waiting}", ""]
    if r:
        days = len(r["top"])
        last = p.dates[days]
        lines += [f"Days {p.dates[1]} to {last}. Annual figures and Sharpe are shown after {LONG} days.", "", "| Book | Return | vs Hold | Return/yr | Sharpe | Max DD |", "|---|---|---|---|---|---|"]
        total = {k: float(np.prod(1 + v) - 1) for k, v in r.items()}
        label = dict(Hold="Hold", SPY="SPY (side)", top="Earnings quality", bottom="Bottom fifth (control)", profitable="Profitable only (side)")
        for k in ("top", "Hold", "SPY", "bottom", "profitable"):
            s = summary(r[k], rf)
            lines.append(f"| {label[k]} | {total[k]:+.2%} | {'' if k == 'Hold' else f'{total[k] - total['Hold']:+.2%}'} | {f'{s['ann_return']:+.1%}' if days >= LONG else '-'} | {f'{s['sharpe']:.2f}' if days >= LONG else '-'} | {s['max_drawdown']:.1%} |")
        lines.append("")
    gone = [s for j, s in enumerate(p.symbols) if not m.traded[-1, j]]
    if gone:
        lines += [f"- Stopped trading, held at the last close: {', '.join(gone)}. Check each by hand: a bankruptcy is flattered here.", ""]
    if state["books"]:
        b = state["books"][-1]
        lines += [f"Book {len(state['books'])}, from filings through {b['cut']}, {b['eligible']} eligible, {len(b['top'])} held:", "", ", ".join(sorted(b["top"])), ""]
    return "\n".join(lines)


def run(cfg: dict, today: date) -> str:
    rule = cfg["rule"]
    symbols = universe(rule)
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    state = check_state(cfg, symbols, state_path)
    m = market(cfg, symbols)
    cuts = due(state["books"], m.p.dates, rule["every"])
    waiting = None
    if cuts:
        try:
            refresh(cfg, symbols)
            for cut in cuts:
                state["books"].append(choose(cfg, m, cut))  # a book never changes once recorded
        except (OSError, ValueError) as e:  # EDGAR down: the book must use the filings of its day, so scoring waits for it
            waiting = f"Next book waits for SEC filings, tried {today}: {e}"
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    text = report(cfg, state, m, score(cfg, m, state["books"]), waiting)
    (out / "tracker.md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    print(run(yaml.safe_load(CONFIG.read_text(encoding="utf-8")), datetime.now(ET).date()))
