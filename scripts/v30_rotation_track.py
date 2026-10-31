"""Forward tracker for the four-sleeve rotation (see configs/rotation_track_v30.yaml, which holds the frozen rule and its verdict).

Reads the 300 companies' daily bars, which the 300-company tracker downloads first, and downloads the ETFs the rule needs (read-only). It
records each week's book and each year's list, then scores them against SPY and the list held equally, and writes
artifacts/reports/rotation_track/tracker.md. Research only: no order is placed anywhere.

    .venv/Scripts/python.exe scripts/v30_rotation_track.py
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import yaml

import v22_bottleneck as bottleneck
import v27_finrl_rotation as v
from atlas.config import load_config
from atlas.feeds.alpaca import HISTORY_BARS_PATH, HISTORY_PATHS, history_end, http_fetch, iter_history, load_credentials
from atlas.swing.backtest import run as run_book, summary
from atlas.swing.data import Panel, daily_rows, save_daily
from atlas.swing.track import last_complete_day
from atlas.swing.walkforward import SwingConfig

CONFIG = Path("configs/rotation_track_v30.yaml")
ET = ZoneInfo("America/New_York")
LONG = 60  # annual figures and Sharpe are shown only after this many days, as in the other trackers


@dataclass(frozen=True)
class Market:
    m: v.Rotation  # the 300, then the rule's ETFs. A gone company keeps its last close
    n: int  # how many of the columns are the 300
    sector_etfs: np.ndarray
    spy: np.ndarray  # SPY's close, for v22's price-based sectors
    cash: np.ndarray
    traded: np.ndarray  # False where a company had no bar that day


def universe(rule: dict) -> list[str]:
    return list(load_config(Path(rule["universe_config"]), SwingConfig).symbols)


def rule_etfs() -> list[str]:
    """The ETFs that are columns of the panel, in v27's order."""
    return list(dict.fromkeys([*[s for g in v.cfg["free_fixed"].values() for s in g], *v.cfg["fallback"], v.cfg["market"], v.cfg["benchmark"]]))


def all_etfs() -> list[str]:
    return list(dict.fromkeys([*rule_etfs(), *bottleneck.cfg["etfs"]]))


def closed_weeks(dates: tuple[date, ...], today: date) -> list[int]:
    """v27's week-ends, plus a last session that is a Friday already over, so a book is recorded before the open it trades at."""
    ends = v.week_ends(dates)
    return [*ends, len(dates) - 1] if dates[-1].weekday() == 4 and today > dates[-1] else ends


def sleeve_mix(weights: list[dict[str, float]], sleeves: int) -> list[dict[str, float]]:
    """The combined weights after each book. All sleeves start on book 0, then book k replaces sleeve k mod sleeves."""
    out = []
    for k in range(len(weights)):
        mix: dict[str, float] = {}
        for s in range(sleeves):
            for name, w in weights[max(0, k - (k - s) % sleeves)].items():
                mix[name] = mix.get(name, 0.0) + w / sleeves
        out.append(mix)
    return out


def fingerprint(rule: dict, symbols: list[str]) -> str:
    code = [inspect.getsource(x) for x in (v.week_ends, v.realized_vol, v.rotation, v.robust_z, v.strength, v.budget, v.scores, v.choose, v.exposure, v.yearly_lists,
                                           bottleneck.sectors, closed_weeks, sleeve_mix)]
    settings = {k: v.cfg[k] for k in ("rule", "free_fixed", "fallback", "list_share", "list_etfs", "list_days", "market", "benchmark")}
    frozen = dict(rule=rule, symbols=symbols, rotation=settings, sectors={k: bottleneck.cfg[k] for k in ("etfs", "market", "group_days")}, code=code)
    return hashlib.sha256(json.dumps(frozen, sort_keys=True).encode("utf-8")).hexdigest()


def check_state(cfg: dict, symbols: list[str], state_path: Path) -> dict:
    """Refuses a changed rule."""
    fp = fingerprint(cfg["rule"], symbols)
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else dict(fingerprint=fp, frozen=str(date.today()), lists={}, books=[])
    if state["fingerprint"] != fp:
        sys.exit("refused: the frozen rule in configs/rotation_track_v30.yaml, v27's settings, its companies or its code has changed")
    return state


def refresh(cfg: dict, now: datetime) -> None:
    """Every ETF's whole history, since adjustments change it. One failure raises before any is stored."""
    last, end = last_complete_day(now), history_end(str(now.date()), now)
    fetch = http_fetch(load_credentials(Path(".env")), timeout_s=60, allowed=HISTORY_PATHS)
    rows = {}
    for s in all_etfs():
        got = daily_rows(iter_history(fetch, HISTORY_BARS_PATH, s, f"{cfg['rule']['history_start']}T00:00:00Z", end, "sip", min_interval_s=1.0, extra={"timeframe": "1Day", "adjustment": "all"}))
        rows[s] = [r for r in got if r["date"] <= last]
    save_daily(rows, Path(cfg["etf_dir"]), {"source": "alpaca-sip-1Day", "adjustment": "all (splits and dividends)", "through": str(last),
                                            "license_notes": "Alpaca historical market data, downloaded for personal research; check Alpaca's data terms before sharing"})


def market(cfg: dict, symbols: list[str], today: date) -> Market:
    """Every SPY session with bars for most of the 300. load_panel would stop the calendar when a company goes."""
    rule = cfg["rule"]
    bars, etf = Path(cfg["bars_dir"]), Path(cfg["etf_dir"])
    dates = pl.read_parquet(etf / f"{v.cfg['market']}.parquet").sort("date")["date"].to_list()
    calendar = pl.DataFrame(dict(date=dates), schema={"date": pl.Date})

    def column(path: Path) -> tuple[np.ndarray, ...]:
        row = calendar.join(pl.read_parquet(path).select("date", "open", "close", "volume"), on="date", how="left")
        close = row["close"].forward_fill().backward_fill()
        return row["open"].fill_null(close).to_numpy(), close.to_numpy(), row["volume"].fill_null(0.0).to_numpy(), row["close"].is_not_null().to_numpy()

    names = [*symbols, *rule_etfs()]
    cols = [column(bars / f"{s}.parquet") for s in symbols] + [column(etf / f"{s}.parquet") for s in rule_etfs()]
    opens, closes, volume, traded = (np.column_stack([c[i] for c in cols]) for i in range(4))
    full = np.nonzero(traded[:, : len(symbols)].mean(axis=1) >= rule["min_stock_bars"])[0][-1] + 1
    sector = np.column_stack([column(etf / f"{s}.parquet")[1] for s in bottleneck.cfg["etfs"]])
    px = calendar.join(pl.read_parquet(bars / f"{rule['cash_symbol']}.parquet").select("date", "close"), on="date", how="left")["close"].forward_fill().backward_fill().to_numpy()
    cash = np.r_[0.0, px[1:] / px[:-1] - 1]
    p = Panel(tuple(dates[:full]), tuple(names), opens[:full], closes[:full], closes[:full], closes[:full], volume[:full])
    m = dataclasses.replace(v.rotation(p), ends=closed_weeks(p.dates, today))
    return Market(m, len(symbols), sector[:full], closes[:full, names.index(v.cfg["market"])], cash[:full], traded[:full])


def year_list(mk: Market, year: int) -> dict[str, list[str]]:
    """A year's groups, from data through the prior year's last session."""
    p = mk.m.p
    groups = v.yearly_lists(p, mk.n, mk.sector_etfs, mk.spy, [year])[year]
    return {g: [p.symbols[j] for j in js] for g, js in groups.items()}


def add_lists(state: dict, mk: Market, first: date) -> None:
    for year in sorted({d.year for d in mk.m.p.dates if d >= first}):
        state["lists"].setdefault(str(year), year_list(mk, year))  # a list never changes once recorded


def signals(mk: Market, first: date) -> list[int]:
    """Week-ends from first_signal on: a position in mk.m.ends."""
    return [k for k, t in enumerate(mk.m.ends) if mk.m.p.dates[t] >= first]


def choose(state: dict, mk: Market, k: int) -> dict:
    """The book of one week-end. A company with no bar on that close can't be picked."""
    m, t = mk.m, mk.m.ends[k]
    p = m.p
    at = {s: j for j, s in enumerate(p.symbols)}
    groups = {g: [at[s] for s in names if mk.traded[t, at[s]]] for g, names in state["lists"][str(p.dates[t].year)].items()}
    b = v.choose(m, k, groups)
    return dict(signal=str(p.dates[t]), budget=b.budget, kept=list(b.kept), weights={p.symbols[j]: w for j, w in b.weights.items()})


def score(cfg: dict, mk: Market, state: dict) -> dict[str, np.ndarray]:
    """Daily returns from first_day, through the session before a book that hasn't been chosen would trade. Each is sold at its last close."""
    rule, m = cfg["rule"], mk.m
    p, books = m.p, state["books"]
    a = p.dates.index(date.fromisoformat(rule["first_day"])) if date.fromisoformat(rule["first_day"]) in p.dates else len(p.dates)
    weeks = signals(mk, date.fromisoformat(rule["first_signal"]))
    end = m.ends[weeks[len(books)]] + 1 if len(weeks) > len(books) else len(p.dates)
    if not books or end <= a:
        return {}
    at = {s: j for j, s in enumerate(p.symbols)}
    mix = [{at[s]: w for s, w in day.items()} for day in sleeve_mix([b["weights"] for b in books], rule["sleeves"])]
    x = v.exposure(mix, [p.dates.index(date.fromisoformat(b["signal"])) for b in books], p.close.shape)
    n = p.close.shape[1]
    plain = np.zeros_like(p.close)
    for t in range(a, end):
        names = [at[s] for g in state["lists"][str(p.dates[t].year)].values() for s in g]
        plain[t, names] = n / len(names)
    go = lambda y, cost: run_book(p, y, y, a, end, cost, None, mk.cash).returns
    grow = lambda j: v.daily(p, j)[a:end]
    return dict(rotation=go(x, rule["cost_bps"]), side=go(x, rule["side_cost_bps"]), SPY=grow(m.spy), plain=go(plain, rule["cost_bps"]), QQQ=grow(m.qqq), rf=mk.cash[a:end])


def gap_interval(d: np.ndarray, rule: dict) -> tuple[float, float]:
    """95% block bootstrap interval of the mean daily gap."""
    rng = np.random.default_rng(rule["seed"])
    block, count = rule["block"], int(np.ceil(len(d) / rule["block"]))
    starts = rng.integers(0, len(d) - block + 1, (rule["bootstrap_samples"], count))
    ids = (starts[:, :, None] + np.arange(block)).reshape(len(starts), -1)[:, : len(d)]
    lo, hi = np.percentile(d[ids].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def verdict(rule: dict, r: dict[str, np.ndarray]) -> str:
    n = rule["review_sessions"]
    days = len(r["rotation"]) if r else 0
    if days < n:
        return f"Not reviewed yet: {days} of {n} sessions. Nothing before then is a verdict."
    rf = r["rf"][:n]
    top, spy, plain = (summary(r[k][:n], rf) for k in ("rotation", "SPY", "plain"))
    lo = gap_interval(r["rotation"][:n] - r["SPY"][:n], rule)[0]
    beats = lambda y: top["sharpe"] > y["sharpe"] and top["ann_return"] >= y["ann_return"]
    return f"Review after {n} sessions: {'PASS' if beats(spy) and beats(plain) and lo > 0 else 'FAIL'} (interval's low end {lo * 252:+.1%} a year)."


def report(cfg: dict, state: dict, mk: Market, r: dict[str, np.ndarray], waiting: str | None) -> str:
    rule, p = cfg["rule"], mk.m.p
    lines = ["# Rotation: forward tracker", "", f"Frozen {state['frozen']}, first day {rule['first_day']}. Research only: no order is placed anywhere, and holdings are not advice.", "",
             f"**{verdict(rule, r)}**", ""]
    if waiting:
        lines += [f"- {waiting}", ""]
    if r:
        days = len(r["rotation"])
        first = p.dates.index(date.fromisoformat(rule["first_day"]))
        lines += [f"Days {p.dates[first]} to {p.dates[first + days - 1]}. Annual figures and Sharpe are shown after {LONG} days.", "", "| Book | Return | vs SPY | Return/yr | Sharpe | Max DD |", "|---|---|---|---|---|---|"]
        total = {k: float(np.prod(1 + r[k]) - 1) for k in ("rotation", "side", "SPY", "plain", "QQQ")}
        label = dict(rotation=f"Rotation, {rule['cost_bps']:g} bps", side=f"Rotation, {rule['side_cost_bps']:g} bps (side)", SPY="SPY", plain="The list held equally", QQQ="QQQ (side)")
        for k, name in label.items():
            s = summary(r[k], r["rf"])
            lines.append(f"| {name} | {total[k]:+.2%} | {'' if k == 'SPY' else f'{total[k] - total['SPY']:+.2%}'} | {f'{s['ann_return']:+.1%}' if days >= LONG else '-'} | {f'{s['sharpe']:.2f}' if days >= LONG else '-'} | {s['max_drawdown']:.1%} |")
        lines.append("")
    gone = [s for j, s in enumerate(p.symbols) if not mk.traded[-1, j]]
    if gone:
        lines += [f"- Stopped trading, held at the last close: {', '.join(gone)}. Check each by hand: a bankruptcy is flattered here.", ""]
    if state["books"]:
        b = state["books"][-1]
        mix = sleeve_mix([x["weights"] for x in state["books"]], rule["sleeves"])[-1]
        lines += [f"Book {len(state['books'])}, from the close of {b['signal']}: groups {', '.join(b['kept'])}, invested {b['budget']:.0%}. Replaces sleeve {(len(state['books']) - 1) % rule['sleeves'] + 1} of {rule['sleeves']}.", "",
                  "Combined weights: " + ", ".join(f"{s} {w:.1%}" for s, w in sorted(mix.items(), key=lambda x: -x[1])), ""]
    return "\n".join(lines)


def run(cfg: dict, now: datetime) -> str:
    rule = cfg["rule"]
    symbols = universe(rule)
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    state = check_state(cfg, symbols, state_path)
    waiting = None
    try:
        refresh(cfg, now)
    except (OSError, ValueError) as e:  # Alpaca down: a book must use the closes of its day, so scoring waits for it
        waiting = f"Next book waits for ETF bars, tried {now.date()}: {e}"
    today = now.astimezone(ET).date()
    mk = market(cfg, symbols, today)
    first = date.fromisoformat(rule["first_signal"])
    add_lists(state, mk, first)
    if not waiting:
        for k in signals(mk, first)[len(state["books"]):]:
            state["books"].append(choose(state, mk, k))  # a book never changes once recorded
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    text = report(cfg, state, mk, score(cfg, mk, state), waiting)
    (out / "tracker.md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    print(run(yaml.safe_load(CONFIG.read_text(encoding="utf-8")), datetime.now(UTC)))
