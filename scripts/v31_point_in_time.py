"""Earnings quality and the rotation with no survivor bias (see configs/point_in_time_v31.yaml, which holds the pre-registered tests). fetch reads
the stored insider zips for who existed, then downloads daily bars (read-only, resumable). build joins them by company and downloads the
filings of companies that ever reach the pool. check prints counts and the largest daily moves, and no rule's return. run writes the report.

    .venv/Scripts/python.exe scripts/v31_point_in_time.py fetch
    .venv/Scripts/python.exe scripts/v31_point_in_time.py build
    .venv/Scripts/python.exe scripts/v31_point_in_time.py check
    .venv/Scripts/python.exe scripts/v31_point_in_time.py run
"""

from __future__ import annotations

import io
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import yaml

import v22_bottleneck as bottleneck
import v23_filings as f
import v27_finrl_rotation as v
import v29_monthly_rotation as slow
from atlas.feeds import sec
from atlas.feeds.alpaca import HISTORY_BARS_PATH, HISTORY_PATHS, http_fetch, iter_history, load_credentials
from atlas.swing.backtest import run, simulate
from atlas.swing.data import Panel, daily_rows, load_panel
from atlas.swing.families import HOLD, exposures
from atlas.swing.fundamentals import extract, fundamental_features

cfg = yaml.safe_load(Path("configs/point_in_time_v31.yaml").read_text(encoding="utf-8"))
START, LAST_FETCH = date.fromisoformat(cfg["history_start"]), date.fromisoformat(cfg["last_fetch"])
FIRST, LAST = date.fromisoformat(cfg["first_scored"]), date.fromisoformat(cfg["last_scored"])
OUT = Path(cfg["out_dir"])
BARS, COMPANIES, FILINGS = OUT / "bars", OUT / "companies", OUT / "filings"
TICKER = re.compile(r"[A-Z]{1,5}(\.[A-Z])?")
SCHEMA = {"date": pl.Date, "open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64}
GONE = 1e-6  # what's left of a failed company's price
TICKERS: dict[str, str] = {}


def clean(symbol: str | None) -> str | None:
    """The first ticker of what was filed, or None if it isn't one."""
    text = (symbol or "").strip().upper()
    s = re.split(r"[,;/\s]+", text)[0].replace("-", ".")
    return s if TICKER.fullmatch(s) and text not in ("NONE", "NA", "N/A", "N.A", "N.A.") else None


def read_filings() -> pl.DataFrame:
    """Each filing's company, ticker and day, through last_fetch."""
    frames = []
    for path in sorted(Path(cfg["zips_dir"]).glob("*_form345.zip")):
        with zipfile.ZipFile(path) as z:
            t = pl.read_csv(io.BytesIO(z.read("SUBMISSION.tsv")), separator="\t", quote_char=None, infer_schema_length=0, truncate_ragged_lines=True)
        frames.append(t.select(cik=pl.col("ISSUERCIK").cast(pl.Int64), symbol=pl.col("ISSUERTRADINGSYMBOL"), day=pl.col("FILING_DATE").str.to_date("%d-%b-%Y", strict=False)))
    return pl.concat(frames).filter(pl.col("day") <= LAST_FETCH).drop_nulls("day")


def segments(rows: pl.DataFrame) -> list[dict]:
    """One row per run of a company's quarters under one ticker: the ticker most used each quarter, so a typo can't count."""
    t = rows.with_columns(symbol=pl.col("symbol").map_elements(clean, return_dtype=pl.String)).drop_nulls("symbol")
    t = t.with_columns(q=pl.col("day").dt.year() * 4 + pl.col("day").dt.quarter())
    counts = t.group_by("cik", "q", "symbol").agg(n=pl.len(), first=pl.col("day").min(), last=pl.col("day").max()).sort("cik", "q", "n", "symbol", descending=[False, False, True, False])
    top = counts.unique(["cik", "q"], keep="first", maintain_order=True)
    days = t.group_by("cik", "q").agg(a=pl.col("day").min(), b=pl.col("day").max())
    top = top.join(days, on=["cik", "q"]).sort("cik", "q")
    out: list[dict] = []
    for r in top.iter_rows(named=True):
        if out and out[-1]["cik"] == r["cik"] and out[-1]["symbol"] == r["symbol"]:
            out[-1]["last"] = r["b"]
        else:
            out.append(dict(cik=r["cik"], symbol=r["symbol"], first=r["a"], last=r["b"]))
    return out


def stitch(pieces: list[tuple[pl.DataFrame, date, date]]) -> pl.DataFrame:
    """A company's bars from its tickers, oldest first, each kept to its days. An earlier piece is scaled so the join makes no return."""
    out = pl.DataFrame(schema=SCHEMA)
    for bars, a, b in pieces:
        part = bars.filter(pl.col("date").is_between(a, b) & (pl.col("volume") > 0))
        if out.height:
            part = part.filter(pl.col("date") > out["date"][-1])
        if out.height and part.height:
            k = part["open"][0] / out["close"][-1]
            out = out.with_columns(*[(pl.col(c) * k).alias(c) for c in ("open", "high", "low", "close")], (pl.col("volume") / k).alias("volume"))
        out = pl.concat([out, part.select(list(SCHEMA))])
    return out


def run_end(bars: pl.DataFrame, day: date) -> date:
    """The end of the unbroken run of real bars reaching past day. Quiet insiders aren't a stop, and a break means the ticker's company ended."""
    end = day
    for d in bars.filter((pl.col("date") > day) & (pl.col("volume") > 0))["date"]:
        if (d - end).days > cfg["max_gap_days"]:
            break
        end = d
    return end


def takeovers(segs: list[dict]) -> dict[str, list[dict]]:
    """Each ticker's segments, to see who takes it over."""
    out: dict[str, list[dict]] = {}
    for s in segs:
        out.setdefault(s["symbol"], []).append(s)
    return out


def taken_on(s: dict, users: list[dict]) -> date | None:
    """The first day another company takes this ticker. One that stopped filing before this company did is a typo, not a new owner."""
    return min((u["first"] for u in users if u["cik"] != s["cik"] and u["first"] > s["first"] and u["last"] > s["last"]), default=None)


def company_pieces(company: list[dict], bars_for, taken: date | None = None) -> list[tuple[pl.DataFrame, date, date]]:
    """Each ticker's bars and days. A ticker with no bars hands its days to the next ticker, and the first days reach back to history_start only if
    the company was filing when the zips begin. The next ticker takes over the day after the last real bar, so a rename leaves no hole. The last
    ticker runs to run_end, or to the day another company takes it."""
    pieces = []
    low = START if company[0]["first"] <= date(2016, 6, 30) else company[0]["first"]
    for i, s in enumerate(company):
        bars = bars_for(s["symbol"])
        if i + 1 < len(company):
            high = company[i + 1]["first"] - timedelta(days=1)
        else:
            high = run_end(bars, s["last"])
            if taken:
                high = min(high, max(s["last"], taken - timedelta(days=1)))
        kept = bars.filter(pl.col("date").is_between(low, high) & (pl.col("volume") > 0)) if bars.height else bars
        if kept.height:
            pieces.append((bars, low, high))
            low = kept["date"][-1] + timedelta(days=1)
    return pieces


def wanted(segs: list[dict]) -> list[str]:
    return sorted({s["symbol"] for s in segs if s["last"] >= START and s["first"] <= LAST_FETCH})


def cmd_fetch() -> None:
    segs = segments(read_filings())
    BARS.mkdir(parents=True, exist_ok=True)
    (OUT / "segments.json").write_text(json.dumps(segs, default=str), encoding="utf-8")
    names = wanted(segs)
    print("companies", len({s["cik"] for s in segs}), "segments", len(segs), "tickers", len(names), flush=True)
    fetch = http_fetch(load_credentials(Path(".env")), timeout_s=60, allowed=HISTORY_PATHS)
    for i, sym in enumerate(names):
        path = BARS / f"{sym}.parquet"
        if path.exists():
            continue
        try:
            rows = daily_rows(iter_history(fetch, HISTORY_BARS_PATH, sym, f"{cfg['history_start']}T00:00:00Z", f"{cfg['last_fetch']}T23:59:59Z", "sip",
                                           min_interval_s=cfg["min_interval_s"], extra={"timeframe": "1Day", "adjustment": "all"}))
        except (OSError, ValueError) as e:
            print("bars failed", sym, str(e)[:80], flush=True)
            continue
        pl.DataFrame(rows, schema=SCHEMA).write_parquet(path)
        if i % 500 == 0:
            print("bars", i, flush=True)
    print("bars stored", len(list(BARS.glob("*.parquet"))), flush=True)


def load_segments() -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for s in json.loads((OUT / "segments.json").read_text(encoding="utf-8")):
        s |= dict(first=date.fromisoformat(s["first"]), last=date.fromisoformat(s["last"]))
        out.setdefault(s["cik"], []).append(s)
    return out


def calendar() -> list[date]:
    """Every SPY session through last_scored. It starts before history_start because the rotation's volatility windows need SPY's longer past."""
    return pl.read_parquet(Path(cfg["spy_dir"]) / "SPY.parquet").filter(pl.col("date") <= LAST).sort("date")["date"].to_list()


def pool_at(dollars: np.ndarray, t: int, size: int) -> np.ndarray:
    """The columns with a bar on the last close and the highest median dollar volume over the sessions before t."""
    median = np.median(dollars[t - cfg["volume_days"]: t], axis=0)
    ok = np.nonzero((dollars[t - 1] > 0) & (median > 0))[0]
    return ok[np.argsort(-median[ok], kind="stable")[:size]]


def list_pool(dollars: np.ndarray, t: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    """On last close t: the columns with every one of group_days + 1 sessions, top size by mean dollar volume over list_days, and those means."""
    seen = (dollars[t - bottleneck.cfg["group_days"]: t + 1] > 0).all(axis=0)
    mean = dollars[t + 1 - v.cfg["list_days"]: t + 1].mean(axis=0)
    ok = np.nonzero(seen)[0]
    return ok[np.argsort(-mean[ok], kind="stable")[:size]], mean


def year_lasts(dates: list[date]) -> dict[int, int]:
    """Each scored year's last close before it starts."""
    return {y: max(i for i, d in enumerate(dates) if d.year < y) for y in range(FIRST.year, LAST.year + 1)}


def load_columns(names: list[str], dates: list[date]) -> tuple[np.ndarray, ...]:
    """Open, close, dollars and traded, one column per file. A column keeps its first price before it starts and its last after it stops."""
    frame = pl.DataFrame(dict(date=dates), schema={"date": pl.Date})
    cols = []
    for n in names:
        row = frame.join(pl.read_parquet(COMPANIES / f"{n}.parquet").select("date", "open", "close", "volume"), on="date", how="left")
        close = row["close"].forward_fill().backward_fill()
        cols.append((row["open"].fill_null(close).to_numpy(), close.to_numpy(), (row["close"] * row["volume"]).fill_null(0.0).to_numpy(), row["close"].is_not_null().to_numpy()))
    return tuple(np.column_stack([c[i] for c in cols]) for i in range(4))


def cmd_build() -> None:
    (OUT / "segments.json").write_text(json.dumps(segments(read_filings()), default=str), encoding="utf-8")
    companies = load_segments()
    COMPANIES.mkdir(parents=True, exist_ok=True)
    FILINGS.mkdir(parents=True, exist_ok=True)

    def bars_for(symbol: str) -> pl.DataFrame:
        path = BARS / f"{symbol}.parquet"
        return pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=SCHEMA)

    built = []
    users = takeovers([s for segs in companies.values() for s in segs])
    for cik, segs in companies.items():
        live = [s for s in segs if s["last"] >= START and s["first"] <= LAST_FETCH]
        rows = stitch(company_pieces(live, bars_for, taken_on(live[-1], users[live[-1]["symbol"]]))) if live else pl.DataFrame(schema=SCHEMA)
        if str(cik) in cfg["cut_at"]:
            rows = rows.filter(pl.col("date") <= date.fromisoformat(cfg["cut_at"][str(cik)][0]))
        if rows.height and str(cik) not in cfg["exclude"]:
            rows.write_parquet(COMPANIES / f"{cik}.parquet")
            built.append(str(cik))
    print("companies with bars", len(built), flush=True)
    dates = calendar()
    _, _, dollars, _ = load_columns(built, dates)
    a = next(i for i, d in enumerate(dates) if d >= FIRST)
    ever: set[int] = set()
    for t in range(a, len(dates), cfg["every"]):
        ever |= set(pool_at(dollars, t, max(cfg["pool_size"], cfg["pool_size_side"])).tolist())
    for t in year_lasts(dates).values():
        ever |= set(list_pool(dollars, t, cfg["pool_size"])[0].tolist())
    names = [built[j] for j in sorted(ever)]
    (OUT / "ever.json").write_text(json.dumps(names), encoding="utf-8")
    print("ever in a pool", len(names), flush=True)
    get = sec.http_get(sec.user_agent(Path(".env")))
    failed = []
    for n in names:
        path = FILINGS / f"{n}.parquet"
        if path.exists():
            continue
        try:
            facts = sec.company_facts(get, int(n))
            pl.concat([extract(facts), f.extract_extra(facts)]).unique(maintain_order=True).write_parquet(path)
        except (OSError, ValueError, KeyError) as e:
            failed.append((n, str(e)[:60]))
    print("filings stored", len(list(FILINGS.glob("*.parquet"))), "failed", len(failed), failed[:8], flush=True)


def stop_points(close: np.ndarray, traded: np.ndarray, last: int) -> list[tuple[int, int, bool]]:
    """Column, last real bar and whether it failed, for each company that stopped."""
    out = []
    for j in range(traded.shape[1]):
        on = np.nonzero(traded[:, j])[0]
        if len(on) and on[-1] < last - cfg["stop_sessions"]:
            e = int(on[-1])
            start = max(int(on[0]), e - cfg["fail_look_sessions"])
            out.append((j, e, bool(close[e, j] <= cfg["fail_share"] * close[start, j])))
    return out


@dataclass(frozen=True)
class Market:
    p: Panel  # the companies, then the rule's ETFs
    n: int  # how many columns are companies
    cash: np.ndarray
    spy: np.ndarray  # SPY's daily return, dividends included
    dollars: np.ndarray
    traded: np.ndarray
    sector_etfs: np.ndarray
    mkt: np.ndarray
    v0: int
    end: int
    stopped: list[tuple[int, int, bool]]


def market(losses: bool) -> Market:
    """Every company that ever reached a pool, plus v27's ETFs. With losses, a failed company falls to nothing the session after its last bar."""
    dates = calendar()
    names = [n for n in json.loads((OUT / "ever.json").read_text(encoding="utf-8")) if n not in cfg["exclude"]]
    opens, closes, dollars, traded = load_columns(names, dates)
    stopped = [(j, e, failed and names[j] not in cfg["not_failed"]) for j, e, failed in stop_points(closes, traded, len(dates) - 1)]
    if losses:
        for j, e, failed in stopped:
            if failed:
                closes[e + 1:, j] = closes[e, j] * GONE
                opens[e + 1:, j] = closes[e, j] * GONE
    etfs = list(dict.fromkeys([*[s for g in v.cfg["free_fixed"].values() for s in g], *v.cfg["fallback"], v.cfg["market"], v.cfg["benchmark"]]))
    e = load_panel(Path(v.cfg["bars_dir"]), etfs)
    at = {d: i for i, d in enumerate(e.dates)}
    idx = [at[d] for d in dates]
    s = load_panel(Path(v.cfg["etf_dir"]), bottleneck.cfg["etfs"] + [bottleneck.cfg["market"]])
    sat = {d: i for i, d in enumerate(s.dates)}
    sidx = [sat[d] for d in dates]
    c = np.hstack([closes, e.close[idx]])
    p = Panel(tuple(dates), (*names, *etfs), np.hstack([opens, e.open[idx]]), c, c, c, np.zeros_like(c))
    ok =[i for i, d in enumerate(dates) if FIRST <= d <= LAST]
    spy = e.close[idx, etfs.index(v.cfg["market"])]
    bil = pl.read_parquet(Path(cfg["cash_dir"]) / f"{cfg['cash_symbol']}.parquet").sort("date")
    px = pl.DataFrame(dict(date=dates), schema={"date": pl.Date}).join(bil.select("date", "close"), on="date", how="left")["close"].forward_fill().backward_fill().to_numpy()
    grow = lambda x: np.r_[0.0, x[1:] / x[:-1] - 1]
    return Market(p, len(names), grow(px), grow(spy), dollars, traded, s.close[sidx, :-1], s.close[sidx, -1], ok[0], ok[-1] + 1, stopped)


def rebalances(m: Market) -> list[int]:
    return list(range(m.v0, m.end, cfg["every"]))


def scores(m: Market, days: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Score and net income of each company on each rebalance, from filings filed before it, for every company in the larger pool."""
    p, n = m.p, m.n
    names = list(p.symbols[:n])
    at = [p.dates[t] for t in days]
    has = [j for j in range(n) if (FILINGS / f"{names[j]}.parquet").exists()]
    empty = np.zeros((len(at), len(has)))
    gpa = fundamental_features(Panel(tuple(at), tuple(names[j] for j in has), empty, empty, empty, empty, empty), FILINGS, empty)["fund_gpa"]
    rows = {j: pl.read_parquet(FILINGS / f"{names[j]}.parquet") for j in has}
    score, income = np.full((len(at), n), np.nan), np.full((len(at), n), np.nan)
    for k, t in enumerate(days):
        pool = set(pool_at(m.dollars, t, max(cfg["pool_size"], cfg["pool_size_side"])).tolist())
        for c, j in enumerate(has):
            if j in pool and np.isfinite(gpa[k, c]):
                q = f.filing_scores(rows[j], at[k], True).get("earnings_quality")
                if q is not None:
                    score[k, j] = q
                    income[k, j] = f.latest_two(rows[j], "net_income", at[k])[1]
    return score, income


def gap_interval(d: np.ndarray) -> list[float]:
    rng = np.random.default_rng(cfg["seed"])
    block, count = cfg["block"], int(np.ceil(len(d) / cfg["block"]))
    starts = rng.integers(0, len(d) - block + 1, (cfg["bootstrap_samples"], count))
    ids = (starts[:, :, None] + np.arange(block)).reshape(len(starts), -1)[:, : len(d)]
    return [round(float(x) * 252 * 100, 1) for x in np.percentile(d[ids].mean(axis=1), [2.5, 97.5])]


def returns(m: Market, days: list[int], books: list[list[int]], cost: float) -> np.ndarray:
    p = m.p
    x = np.zeros_like(p.close)
    for t, names in zip(days, books):
        x[t: t + cfg["every"], names] = p.close.shape[1] / len(names)
    return run(p, x, x, m.v0, m.end, cost, None, m.cash).returns


def quality(m: Market, days: list[int], score: np.ndarray, income: np.ndarray, size: int) -> dict:
    """Tests 1 to 6 at a pool size."""
    b: dict[str, list[list[int]]] = dict(top=[], bottom=[], hold=[], profitable=[])
    eligible = []
    for k, t in enumerate(days):
        pool = pool_at(m.dollars, t, size)
        inside = np.full(m.n, np.nan)
        inside[pool] = score[k, pool]
        count = int(np.isfinite(inside).sum())
        if count < cfg["min_eligible"]:
            sys.exit(f"refused: only {count} eligible on {m.p.dates[t]}")
        eligible.append(count)
        b["top"].append(f.top(inside, cfg["top_share"]))
        b["bottom"].append(f.top(inside, cfg["top_share"], highest=False))
        b["hold"].append(pool.tolist())
        b["profitable"].append(f.top(np.where(income[k] > 0, inside, np.nan), cfg["top_share"]))
    r ={k: returns(m, days, x, cfg["quality_cost_bps"]) for k, x in b.items()}
    rf, spy = m.cash[m.v0: m.end], m.spy[m.v0: m.end]
    s = {k: f.stats(x, rf) for k, x in r.items()} | dict(spy=f.stats(spy, rf))
    years = np.array([d.year for d in m.p.dates[m.v0: m.end]])
    half = np.arange(len(rf)) < len(rf) // 2
    y = r["hold"]
    by_year = {int(a): dict(rule=round(float((1 + r["top"][years == a]).prod() - 1) * 100, 1), hold=round(float((1 + y[years == a]).prod() - 1) * 100, 1), spy=round(float((1 + spy[years == a]).prod() - 1) * 100, 1)) for a in sorted(set(years))}
    best = sorted(by_year, key=lambda a: -(by_year[a]["rule"] - by_year[a]["hold"]))[:2]
    keep = ~np.isin(years, best)
    sharper = lambda i: f.stats(r["top"][i], rf[i])["sharpe"] > f.stats(y[i], rf[i])["sharpe"]
    tests = dict(test1=f.beats(s["top"], s["hold"]), test2=bool(s["top"]["ann_return"] > s["bottom"]["ann_return"]), test3=bool(sharper(keep)), test4=bool(sharper(half) and sharper(~half)),
                 test5=f.beats(s["profitable"], s["hold"]), test6=f.beats(s["top"], s["spy"]))
    loss = float(np.mean([np.mean(income[k, names] < 0) for k, names in enumerate(b["top"])]))
    return dict(tests=tests, passed=all(tests.values()), figures=s, by_year=by_year, best_years=best, eligible=[min(eligible), max(eligible)], loss_maker_share=round(loss, 3),
                gap_to_hold_95=gap_interval(r["top"] - y), gap_to_spy_95=gap_interval(r["top"] - spy))


def lists_at(m: Market) -> dict[int, dict[str, list[int]]]:
    """Each year's groups, as v27 builds them, from the year's own pool instead of the survivors."""
    p = m.p
    etfs = bottleneck.cfg["etfs"]
    at = {s: j for j, s in enumerate(p.symbols)}
    out = {}
    for y, t in year_lasts(list(p.dates)).items():
        pool, mean = list_pool(m.dollars, t, cfg["pool_size"])
        sector = bottleneck.sectors(p.close[:, pool], m.sector_etfs, m.mkt, t + 1)
        groups = {g: [at[s] for s in fixed] for g, fixed in v.cfg["free_fixed"].items()}
        for g, count in v.cfg["list_share"].items():
            wanted = pool[np.isin(sector, [etfs.index(e) for e in v.cfg["list_etfs"][g]])]
            groups[g] = wanted[np.argsort(-mean[wanted], kind="stable")[:count]].tolist() + groups[g]
        out[y] = groups
    return out


def rotation(m: Market) -> dict:
    """Tests 7 and 8."""
    p = m.p
    rot = v.rotation(p)
    lists = lists_at(m)
    live = np.hstack([m.traded, np.ones((len(p.dates), p.close.shape[1] - m.n), dtype=bool)])
    weekly = [v.choose(rot, k, {g: [j for j in js if live[t, j]] for g, js in lists[p.dates[t].year].items()}) for k, t in enumerate(rot.ends) if p.dates[t].year >= FIRST.year]
    books = slow.monthly(weekly, p.dates)
    cost = cfg["rotation_cost_bps"]
    r = v.net(rot, books, m.v0, m.end, cost, m.cash)
    plain = slow.hold_list(p, lists, m.v0, m.end, cost, m.cash)
    rf, spy = m.cash[m.v0: m.end], m.spy[m.v0: m.end]
    s = dict(rotation=f.stats(r, rf), spy=f.stats(spy, rf), hold_the_list=f.stats(plain, rf))
    names = {y: {g: [ticker(p.symbols[j]) for j in js] for g, js in groups.items()} for y, groups in lists.items()}
    return dict(tests=dict(test7=f.beats(s["rotation"], s["spy"]), test8=f.beats(s["rotation"], s["hold_the_list"])), figures=s, gap_to_spy_95=gap_interval(r - spy),
                by_year=v.by_year(list(p.dates[m.v0: m.end]), r, dict(spy=spy, hold_the_list=plain)), lists=names)


def ticker(name: str) -> str:
    if not TICKERS:
        for cik, segs in load_segments().items():
            TICKERS[str(cik)] = segs[-1]["symbol"]
    return TICKERS.get(name, name)


def cmd_check() -> None:
    m = market(True)
    p, n = m.p, m.n
    days = rebalances(m)
    pools = [pool_at(m.dollars, t, cfg["pool_size"]) for t in days]
    print("companies", n, "rebalances", len(days), "pool sizes", min(map(len, pools)), max(map(len, pools)), "stopped", len(m.stopped), "failed", sum(x[2] for x in m.stopped), flush=True)
    ever = set(np.concatenate(pools).tolist())
    print("ever in the 300 pool", len(ever), "of them stopped", sum(j in ever for j, _, _ in m.stopped), "failed", sum(j in ever and x for j, _, x in m.stopped))
    print("failed from the 300 pool:", sorted((ticker(p.symbols[j]), str(p.dates[e])) for j, e, x in m.stopped if x and j in ever))
    print("first pool, top 20:", [ticker(p.symbols[j]) for j in pools[0][:20]])
    print("last pool, top 20:", [ticker(p.symbols[j]) for j in pools[-1][:20]])
    raw = market(False)
    c = raw.p.close[:, :n]
    moves = c[1:] / c[:-1] - 1
    i, j = np.nonzero(np.isin(np.arange(n), sorted(ever | set(np.concatenate([pool_at(m.dollars, t, cfg["pool_size_side"]) for t in days]).tolist()))) & ((moves > 2.0) | (moves < -0.75)))
    print("daily moves beyond +200% or -75%", len(i))
    for a, b in sorted(zip(i.tolist(), j.tolist()), key=lambda x: -abs(moves[x])):
        dv = raw.dollars[:, b]
        print(ticker(p.symbols[b]), p.symbols[b], p.dates[a + 1], round(float(moves[a, b]), 2), "close", round(float(c[a, b]), 2), "->", round(float(c[a + 1, b]), 2), f"$vol {dv[a] / 1e6:.2f}m -> {dv[a + 1] / 1e6:.2f}m", "in300" if b in ever else "")


def cmd_run() -> None:
    m, raw = market(True), market(False)
    days = rebalances(m)
    score, income = scores(m, days)
    main_q = quality(m, days, score, income, cfg["pool_size"])
    full, stocks = v.free_panel()
    a, b = v.window(stocks.dates)
    cash = v.cash_for(stocks.dates)
    held = simulate(stocks, exposures(stocks, *HOLD), a, b, cfg["quality_cost_bps"], cash).returns
    ours = returns(m, days, [pool_at(m.dollars, t, cfg["pool_size"]).tolist() for t in days], cfg["quality_cost_bps"])
    years = np.array([d.year for d in stocks.dates[a:b]])
    grow = lambda x, y, g: round(float((1 + x[y == g]).prod() - 1) * 100, 1)
    mine = np.array([d.year for d in m.p.dates[m.v0: m.end]])
    measure = dict(hold_on_the_300=f.stats(held, cash[a:b]), hold_on_this_pool=f.stats(ours, m.cash[m.v0: m.end]),
                   by_year={int(y): dict(the_300=grow(held, years, y), this_pool=grow(ours, mine, y)) for y in sorted(set(years))})
    res = dict(config="configs/point_in_time_v31.yaml", window=f"{m.p.dates[m.v0]}..{m.p.dates[m.end - 1]}", companies=m.n, excluded=cfg["exclude"], stopped=len(m.stopped), failed=sum(x[2] for x in m.stopped),
               survivor_bias=measure, quality=main_q, rotation=rotation(m),
               side=dict(all_at_the_last_close=dict(quality=quality(raw, days, score, income, cfg["pool_size"]), rotation=rotation(raw)), larger_pool=quality(m, days, score, income, cfg["pool_size_side"])))
    out = Path(cfg["report"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    {"fetch": cmd_fetch, "build": cmd_build, "check": cmd_check, "run": cmd_run}[sys.argv[1]]()
