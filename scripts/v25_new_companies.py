"""Earnings quality on companies outside the 300, and against SPY (see configs/new_companies_v25.yaml, which holds the pre-registered
tests). fetch reads the companies' SEC filings, then daily bars for those with the figures the rule needs (read-only, resumable).
check prints eligible counts and the largest daily moves, and no rule's return. run writes artifacts/reports/swing/v25_new_companies.json.

    .venv/Scripts/python.exe scripts/v25_new_companies.py fetch
    .venv/Scripts/python.exe scripts/v25_new_companies.py check
    .venv/Scripts/python.exe scripts/v25_new_companies.py run
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import yaml

import v22_bottleneck as bottleneck
import v23_filings as f
from atlas.config import load_config
from atlas.feeds import sec
from atlas.feeds.alpaca import HISTORY_BARS_PATH, HISTORY_PATHS, http_fetch, iter_history, load_credentials
from atlas.swing.backtest import run
from atlas.swing.data import Panel, daily_rows, load_panel
from atlas.swing.fundamentals import extract, fundamental_features
from atlas.swing.walkforward import SwingConfig

cfg = yaml.safe_load(Path("configs/new_companies_v25.yaml").read_text(encoding="utf-8"))
OUT = Path("artifacts/reports/swing/v25_new_companies.json")
FIRST, LAST = date.fromisoformat(cfg["first_scored"]), date.fromisoformat(cfg["last_scored"])
EVERY = cfg["every"]
NEEDED = {"net_income", "cfo", "assets"}


@dataclass(frozen=True)
class Market:
    p: Panel  # a company keeps its first price before it lists and its last after it stops, so those days earn nothing
    cash: np.ndarray
    spy: np.ndarray  # SPY's daily return, dividends included
    dollars: np.ndarray  # close times volume, zero on sessions with no bar
    v0: int
    end: int


def companies(listing: dict, inside: set[str]) -> dict[str, int]:
    """Ticker -> CIK outside the 300: the first ticker of each CIK, leaving out every CIK a company of the 300 uses."""
    used = {row["cik_str"] for row in listing.values() if row["ticker"].replace("-", ".") in inside}
    out: dict[str, int] = {}
    seen: set[int] = set()
    for row in listing.values():
        cik = int(row["cik_str"])
        if cik not in used and cik not in seen:
            seen.add(cik)
            out[row["ticker"].replace("-", ".")] = cik
    return out


def cmd_fetch() -> None:
    inside = set(load_config(Path(f.cfg["swing_config"]), SwingConfig).symbols)
    get = sec.http_get(sec.user_agent(Path(".env")))
    names = companies(get(sec.TICKERS_URL), inside)
    out = Path(cfg["sec_dir"])
    out.mkdir(parents=True, exist_ok=True)
    print("companies", len(names), flush=True)
    failed = []
    for i, (sym, cik) in enumerate(names.items()):
        path = out / f"{sym}.parquet"
        if path.exists():
            continue
        try:
            facts = sec.company_facts(get, cik)
            pl.concat([extract(facts), f.extract_extra(facts)]).unique(maintain_order=True).write_parquet(path)
        except (OSError, ValueError, KeyError) as e:
            failed.append((sym, str(e)[:80]))
        if i % 500 == 0:
            print("filings", i, flush=True)
    print("filings stored", len(list(out.glob("*.parquet"))), "failed", len(failed), failed[:10], flush=True)
    bars = Path(cfg["bars_dir"])
    bars.mkdir(parents=True, exist_ok=True)
    fetch = http_fetch(load_credentials(Path(".env")), timeout_s=60, allowed=HISTORY_PATHS)
    wanted = [p.stem for p in sorted(out.glob("*.parquet")) if has_figures(pl.read_parquet(p))]
    print("companies with the figures", len(wanted), flush=True)
    for i, sym in enumerate(wanted):
        path = bars / f"{sym}.parquet"
        if path.exists():
            continue
        try:
            rows = daily_rows(iter_history(fetch, HISTORY_BARS_PATH, sym, f"{cfg['history_start']}T00:00:00Z", f"{cfg['last_fetch']}T23:59:59Z", "sip",
                                           min_interval_s=cfg["min_interval_s"], extra={"timeframe": "1Day", "adjustment": "all"}))
        except (OSError, ValueError) as e:
            print("bars failed", sym, str(e)[:80], flush=True)
            continue
        pl.DataFrame(rows, schema={"date": pl.Date, "open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64}).write_parquet(path)
        if i % 500 == 0:
            print("bars", i, flush=True)
    print("bars stored", len(list(bars.glob("*.parquet"))), flush=True)


def has_figures(rows: pl.DataFrame) -> bool:
    """Filed all three of the rule's figures during the window."""
    live = rows.filter(pl.col("filed").is_between(FIRST - timedelta(days=550), LAST))
    return NEEDED <= set(live["item"].unique().to_list())


def market() -> Market:
    """Every SPY session through last_scored, for each company with bars, less any under exclude."""
    spy = pl.read_parquet(Path(cfg["spy_dir"]) / f"{cfg['spy']}.parquet").filter(pl.col("date") <= LAST).sort("date")
    dates = spy["date"].to_list()
    calendar = pl.DataFrame(dict(date=dates), schema={"date": pl.Date})
    frames = {}
    for path in sorted(Path(cfg["bars_dir"]).glob("*.parquet")):
        if path.stem not in cfg["exclude"]:
            d = pl.read_parquet(path).filter(pl.col("date") <= LAST)
            if d.height:
                frames[path.stem] = d
    symbols = tuple(frames)
    opens, closes, dollars = (np.empty((len(dates), len(symbols))) for _ in range(3))
    for j, s in enumerate(symbols):
        row = calendar.join(frames[s].select("date", "open", "close", "volume"), on="date", how="left")
        filled = row["close"].forward_fill().backward_fill()
        closes[:, j] = filled.to_numpy()
        opens[:, j] = row["open"].fill_null(filled).to_numpy()
        dollars[:, j] = (row["close"] * row["volume"]).fill_null(0.0).to_numpy()
    bil = pl.read_parquet(Path(cfg["cash_dir"]) / f"{cfg['cash_symbol']}.parquet").sort("date")
    px = calendar.join(bil.select("date", "close"), on="date", how="left")["close"].forward_fill().backward_fill().to_numpy()
    grow = lambda x: np.r_[0.0, x[1:] / x[:-1] - 1]
    ok = [i for i, d in enumerate(dates) if FIRST <= d <= LAST]
    p = Panel(tuple(dates), symbols, opens, closes, closes, closes, np.zeros_like(closes))
    return Market(p, grow(px), grow(spy["close"].to_numpy()), dollars, ok[0], ok[-1] + 1)


def liquid(m: Market, t: int, floor: float) -> np.ndarray:
    return np.median(m.dollars[t - cfg["volume_days"]: t], axis=0) >= floor


def scores(m: Market, days: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Score and net income of each company on each rebalance, from filings filed before it, for the companies liquid at the lower floor."""
    p = m.p
    n = len(p.symbols)
    at = [p.dates[t] for t in days]
    empty = np.zeros((len(at), n))
    gpa = fundamental_features(Panel(tuple(at), p.symbols, empty, empty, empty, empty, empty), Path(cfg["sec_dir"]), empty)["fund_gpa"]
    rows = {s: pl.read_parquet(Path(cfg["sec_dir"]) / f"{s}.parquet") for s in p.symbols}
    score, income = np.full((len(at), n), np.nan), np.full((len(at), n), np.nan)
    for k, t in enumerate(days):
        ok = liquid(m, t, cfg["min_dollar_volume"]) & np.isfinite(gpa[k])
        for j in np.nonzero(ok)[0]:
            q = f.filing_scores(rows[p.symbols[j]], at[k], True).get("earnings_quality")
            if q is not None:
                score[k, j] = q
                income[k, j] = f.latest_two(rows[p.symbols[j]], "net_income", at[k])[1]
    return score, income


def sector_book(m: Market, sector_inputs: tuple[np.ndarray, np.ndarray], t: int, score: np.ndarray, seen: np.ndarray) -> list[int]:
    """Top fifth of each sector, among companies with bars on all of group_days."""
    idx = np.nonzero(np.isfinite(score) & seen)[0]
    group = np.full(len(score), -1)
    group[idx] = bottleneck.sectors(m.p.close[:, idx], *sector_inputs, t)
    return f.by_sector(np.where(group >= 0, score, np.nan), group)


def annual(r: np.ndarray, rf: np.ndarray) -> dict:
    return f.stats(r, rf)


def gap_interval(d: np.ndarray) -> list[float]:
    rng = np.random.default_rng(cfg["seed"])
    block, count = cfg["block"], int(np.ceil(len(d) / cfg["block"]))
    starts = rng.integers(0, len(d) - block + 1, (cfg["bootstrap_samples"], count))
    ids = (starts[:, :, None] + np.arange(block)).reshape(len(starts), -1)[:, : len(d)]
    return [round(float(x) * 252 * 100, 1) for x in np.percentile(d[ids].mean(axis=1), [2.5, 97.5])]


def pool_books(m: Market, days: list[int], score: np.ndarray, income: np.ndarray, floor: float) -> dict[str, list[list[int]]]:
    out: dict[str, list[list[int]]] = dict(top=[], bottom=[], all=[], profitable=[])
    for k, t in enumerate(days):
        s = np.where(liquid(m, t, floor), score[k], np.nan)
        pool = np.nonzero(np.isfinite(s))[0].tolist()
        if len(pool) < cfg["min_eligible"]:
            sys.exit(f"refused: only {len(pool)} eligible on {m.p.dates[t]}")
        out["top"].append(f.top(s, cfg["top_share"]))
        out["bottom"].append(f.top(s, cfg["top_share"], highest=False))
        out["all"].append(pool)
        out["profitable"].append(f.top(np.where(income[k] > 0, s, np.nan), cfg["top_share"]))
    return out


def returns(m: Market, days: list[int], books: list[list[int]], cost: float) -> np.ndarray:
    p, n = m.p, len(m.p.symbols)
    x = np.zeros((len(p.dates), n))
    for t, names in zip(days, books):
        x[t: t + EVERY, names] = n / len(names)
    return run(p, x, x, m.v0, m.end, cost, None, m.cash).returns


def evaluate(m: Market, days: list[int], score: np.ndarray, income: np.ndarray, floor: float, sector_inputs: tuple[np.ndarray, np.ndarray]) -> dict:
    rf = m.cash[m.v0: m.end]
    b = pool_books(m, days, score, income, floor)
    r = {k: returns(m, days, v, cfg["cost_bps"]) for k, v in b.items()}
    seen = [m.dollars[t - bottleneck.cfg["group_days"]: t].min(axis=0) > 0 for t in days]
    b["sector"] = [sector_book(m, sector_inputs, t, np.where(liquid(m, t, floor), score[k], np.nan), seen[k]) for k, t in enumerate(days)]
    r["sector"] = returns(m, days, b["sector"], cfg["cost_bps"])
    spy = m.spy[m.v0: m.end]
    s = {k: annual(v, rf) for k, v in r.items()} | dict(spy=annual(spy, rf))
    years = np.array([d.year for d in m.p.dates[m.v0: m.end]])
    half = np.arange(len(rf)) < len(rf) // 2
    y = r["all"]
    yearly = {int(a): dict(rule=round(float((1 + r["top"][years == a]).prod() - 1) * 100, 1), all=round(float((1 + y[years == a]).prod() - 1) * 100, 1),
                           spy=round(float((1 + spy[years == a]).prod() - 1) * 100, 1)) for a in sorted(set(years))}
    best = sorted(yearly, key=lambda a: -(yearly[a]["rule"] - yearly[a]["all"]))[:2]
    keep = ~np.isin(years, best)
    sharper = lambda i: annual(r["top"][i], rf[i])["sharpe"] > annual(y[i], rf[i])["sharpe"]
    tests = dict(test1=f.beats(s["top"], s["all"]), test2=bool(s["top"]["ann_return"] > s["bottom"]["ann_return"]), test3=bool(sharper(keep)), test4=bool(sharper(half) and sharper(~half)),
                 test5=f.beats(s["sector"], s["all"]), test6=f.beats(s["profitable"], s["all"]), test7=f.beats(s["top"], s["spy"]))
    loss = float(np.mean([np.mean(income[k, names] < 0) for k, names in enumerate(b["top"])]))
    return dict(tests=tests, passed=all(tests.values()), figures=s, at_side_cost=annual(returns(m, days, b["top"], cfg["side_cost_bps"]), rf), by_year=yearly, best_years=best,
                eligible=[min(map(len, b["all"])), max(map(len, b["all"]))], held=[min(map(len, b["top"])), max(map(len, b["top"]))], loss_maker_share=round(loss, 3),
                gap_to_yardstick_95=gap_interval(r["top"] - y), gap_to_spy_95=gap_interval(r["top"] - spy),
                monthly_turnover=round(float(np.mean([1 - len(set(a) & set(c)) / max(len(a), len(c)) for a, c in zip(b["top"], b["top"][1:])]) / 3), 3))


def rebalances(m: Market) -> list[int]:
    return list(range(m.v0, m.end, EVERY))


def cmd_check() -> None:
    m = market()
    days = rebalances(m)
    score, _ = scores(m, days)
    counts = [int(np.isfinite(s).sum()) for s in score]
    large = [int(np.isfinite(np.where(liquid(m, t, cfg["min_dollar_volume_large"]), score[k], np.nan)).sum()) for k, t in enumerate(days)]
    print("companies with bars", len(m.p.symbols), "rebalances", len(days), "eligible", min(counts), max(counts), "large", min(large), max(large), flush=True)
    ever = np.isfinite(score).any(axis=0)
    c = m.p.close
    moves = c[1:] / c[:-1] - 1
    i, j = np.nonzero(ever & ((moves > 2.0) | (moves < -0.75)))
    print("eligible at some point", int(ever.sum()), "daily moves beyond +200% or -75% in them", len(i))
    for a, b in sorted(zip(i.tolist(), j.tolist()), key=lambda x: -abs(moves[x]))[:40]:
        print(m.p.symbols[b], m.p.dates[a + 1], round(float(moves[a, b]), 2), "close", round(float(c[a, b]), 2), "->", round(float(c[a + 1, b]), 2))


def against_spy() -> dict:
    """The 300 and Hold against SPY, on v23's window and 2017-2018. A side result."""
    import v19_pairs as pairs
    from atlas.swing.backtest import simulate
    from atlas.swing.families import HOLD, exposures

    held = pairs.market(f.cfg["swing_config"])
    p, v0, end = held.p, held.v0, held.end
    n = p.close.shape[1]
    swing = load_config(Path(f.cfg["swing_config"]), SwingConfig)
    raw = load_panel(swing.raw_dataset_dir, swing.symbols)
    raw_at = {d: i for i, d in enumerate(raw.dates)}
    feats = fundamental_features(p, Path(f.cfg["fundamentals_dir"]), raw.close[[raw_at[d] for d in p.dates]])
    rows = {s: pl.concat([pl.read_parquet(Path(f.cfg[k]) / f"{s}.parquet") for k in ("fundamentals_dir", "extra_dir")]) for s in p.symbols}
    spy = pl.read_parquet(Path(cfg["spy_dir"]) / f"{cfg['spy']}.parquet").sort("date")
    at = dict(zip(spy["date"].to_list(), spy["close"].to_list()))
    close = np.array([at[d] for d in p.dates])
    mkt = np.r_[0.0, close[1:] / close[:-1] - 1]

    def span(a: int, b: int) -> dict:
        x = np.zeros((len(p.dates), n))
        for t in range(a, b, EVERY):
            score = np.full(n, np.nan)
            for j, s in enumerate(p.symbols):
                if np.isfinite(feats["fund_gpa"][t, j]):
                    score[j] = f.filing_scores(rows[s], p.dates[t], True).get("earnings_quality", np.nan)
            x[t: t + EVERY, f.top(score, f.cfg["top_share"])] = n / len(f.top(score, f.cfg["top_share"]))
        rf = held.cash[a:b]
        r = run(p, x, x, a, b, cfg["cost_bps"], None, held.cash).returns
        h = simulate(p, exposures(p, *HOLD), a, b, cfg["cost_bps"], held.cash).returns
        s = mkt[a:b]
        return dict(first=str(p.dates[a]), last=str(p.dates[b - 1]), rule=annual(r, rf), hold=annual(h, rf), spy=annual(s, rf), rule_gap_95=gap_interval(r - s), hold_gap_95=gap_interval(h - s))

    start = next(i for i, d in enumerate(p.dates) if d >= date(2017, 1, 24))
    return dict(window=span(v0, end), before=span(start, v0))


def cmd_run() -> None:
    m = market()
    days = rebalances(m)
    score, income = scores(m, days)
    e = load_panel(Path(bottleneck.cfg["etf_dir"]), bottleneck.cfg["etfs"] + [bottleneck.cfg["market"]])
    at = {d: i for i, d in enumerate(e.dates)}
    idx = [at[d] for d in m.p.dates]
    inputs = (e.close[idx, :-1], e.close[idx, -1])
    res = dict(config="configs/new_companies_v25.yaml", window=f"{m.p.dates[m.v0]}..{m.p.dates[m.end - 1]}", companies=len(m.p.symbols), excluded=cfg["exclude"],
               main=evaluate(m, days, score, income, cfg["min_dollar_volume"], inputs), large_side=evaluate(m, days, score, income, cfg["min_dollar_volume_large"], inputs), the_300=against_spy())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    {"fetch": cmd_fetch, "check": cmd_check, "run": cmd_run}[sys.argv[1]]()
