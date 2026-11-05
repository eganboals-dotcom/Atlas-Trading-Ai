"""Opening-range breakout on stocks in play (see configs/stocks_in_play_v14.yaml, which holds the pre-registered tests).

fetch downloads, day by day and resumably, each company's first 5-minute bar and then 1-minute bars for that day's
picks and random controls only (read-only Alpaca market data, stopping before the holdout). run scores them and
writes artifacts/reports/swing/v14_stocks_in_play.json.

    .venv/Scripts/python.exe scripts/research/trading/v14_stocks_in_play.py fetch
    .venv/Scripts/python.exe scripts/research/trading/v14_stocks_in_play.py run
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import yaml

from atlas.feeds.alpaca import HISTORY_BARS_PATH, HISTORY_PATHS, FeedDisconnected, http_fetch, load_credentials

cfg = yaml.safe_load(Path("configs/stocks_in_play_v14.yaml").read_text(encoding="utf-8"))
SYMBOLS = yaml.safe_load(Path(cfg["universe_config"]).read_text(encoding="utf-8"))["symbols"]
OPEN5 = Path("data/processed/u300_open5")
MINUTE = Path("data/processed/u300_orb_1min")
OUT = Path("artifacts/reports/swing/v14_stocks_in_play.json")
NY = ZoneInfo("America/New_York")
YEAR = 252
BAR = ("t", "o", "h", "l", "c", "v")


def daily(path: str) -> dict[str, np.ndarray]:
    """Date-by-symbol arrays over the universe's common days, before the holdout."""
    frames = {s: pl.read_parquet(Path(path) / f"{s}.parquet").sort("date") for s in SYMBOLS}
    cut = date.fromisoformat(cfg["holdout_start"])
    days = sorted(set().union(*(f["date"].to_list() for f in frames.values())))
    days = [d for d in days if d < cut]
    out: dict = {"days": days}
    for c in ("open", "high", "low", "close", "volume"):
        a = np.full((len(days), len(SYMBOLS)), np.nan)
        idx = {d: i for i, d in enumerate(days)}
        for j, s in enumerate(SYMBOLS):
            f = frames[s].filter(pl.col("date") < cut)
            a[[idx[d] for d in f["date"].to_list()], j] = f[c].to_numpy()
        out[c] = a
    return out


def rolling_prev(a: np.ndarray, n: int) -> np.ndarray:
    """Mean of the previous ``n`` rows (today excluded), NaN until ``n`` are available."""
    out = np.full(a.shape, np.nan)
    for i in range(n, len(a)):
        w = a[i - n:i]
        ok = (~np.isnan(w)).sum(axis=0) == n
        out[i, ok] = w[:, ok].mean(axis=0)
    return out


def features() -> dict:
    """What was known before each day's open: eligibility and ATR as a fraction of price."""
    adj, raw = daily(cfg["daily_dir"]), daily(cfg["raw_daily_dir"])
    prev = np.vstack([np.full(adj["close"].shape[1], np.nan), adj["close"][:-1]])
    tr = np.nanmax(np.stack([adj["high"] - adj["low"], np.abs(adj["high"] - prev), np.abs(adj["low"] - prev)]), axis=0)
    atr = rolling_prev(tr, cfg["atr_days"])
    atr_frac = atr / prev
    raw_prev = np.vstack([np.full(raw["close"].shape[1], np.nan), raw["close"][:-1]])
    avg_vol = rolling_prev(raw["volume"], cfg["atr_days"])
    eligible = (raw_prev > cfg["min_price"]) & (avg_vol >= cfg["min_avg_volume"]) & (atr_frac * raw_prev > cfg["min_atr"])
    return dict(days=adj["days"], eligible=eligible, atr_frac=atr_frac, adj=adj)


def get_bars(fetch, symbols: list[str], start: str, end: str, timeframe: str) -> dict[str, list[dict]]:
    token, out = None, {}
    while True:
        params = {"symbols": ",".join(symbols), "start": start, "end": end, "feed": cfg["feed"], "limit": "10000",
                  "sort": "asc", "timeframe": timeframe, "adjustment": cfg["adjustment"]}
        if token:
            params["page_token"] = token
        for attempt in range(9):
            try:
                data = fetch(HISTORY_BARS_PATH, params)
                break
            except FeedDisconnected:
                if attempt == 8:
                    raise
                time.sleep(min(60.0, 2.0 * (attempt + 1)))
        time.sleep(cfg["min_interval_s"])
        for s, bars in (data.get("bars") or {}).items():
            out.setdefault(s, []).extend(bars)
        token = data.get("next_page_token")
        if not token:
            return out


def et(day: date, hour: int, minute: int) -> str:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=NY).isoformat()


def frame(bars: dict[str, list[dict]]) -> pl.DataFrame:
    rows = [{"symbol": s, **{k: b[k] for k in BAR}} for s, v in bars.items() for b in v]
    schema = {"symbol": pl.String, "t": pl.String, "o": pl.Float64, "h": pl.Float64, "l": pl.Float64, "c": pl.Float64,
              "v": pl.Float64}
    return pl.DataFrame(rows, schema=schema)


def picks(f: dict, first5: dict[date, pl.DataFrame]) -> dict[date, dict]:
    """Each day's in-play names and random controls, from the first bars and the previous days only."""
    rng = np.random.default_rng(cfg["seed"])
    idx = {s: j for j, s in enumerate(SYMBOLS)}
    days = f["days"]
    vol = np.full(f["eligible"].shape, np.nan)
    up = np.full(f["eligible"].shape, np.nan)
    hi, lo = np.full(vol.shape, np.nan), np.full(vol.shape, np.nan)
    pos = {d: i for i, d in enumerate(days)}
    for d, df in first5.items():
        i = pos[d]
        j = [idx[s] for s in df["symbol"].to_list()]
        vol[i, j], hi[i, j], lo[i, j] = df["v"].to_numpy(), df["h"].to_numpy(), df["l"].to_numpy()
        up[i, j] = np.sign(df["c"].to_numpy() - df["o"].to_numpy())
    base = rolling_prev(vol, cfg["rvol_days"])
    with np.errstate(divide="ignore", invalid="ignore"):
        rvol = vol / base
    out = {}
    for i, d in enumerate(days):
        ok = f["eligible"][i] & ~np.isnan(rvol[i]) & (up[i] != 0) & ~np.isnan(up[i])
        names = np.nonzero(ok)[0]
        if not len(names):
            continue
        strong = names[rvol[i, names] >= cfg["min_rvol"]]
        top = strong[np.argsort(-rvol[i, strong])][: cfg["top_n"]]
        rand = rng.choice(names, min(cfg["top_n"], len(names)), replace=False)
        info = {int(j): (float(up[i, j]), float(hi[i, j]), float(lo[i, j]), float(f["atr_frac"][i, j])) for j in set(top) | set(rand)}
        out[d] = dict(top=[int(j) for j in top], rand=[int(j) for j in rand], info=info)
    return out


def load_open5() -> dict[date, pl.DataFrame]:
    return {date.fromisoformat(p.stem): pl.read_parquet(p) for p in sorted(OPEN5.glob("*.parquet"))}


def cmd_fetch() -> None:
    fetch = http_fetch(load_credentials(Path(".env")), timeout_s=60, allowed=HISTORY_PATHS)
    f = features()
    OPEN5.mkdir(parents=True, exist_ok=True)
    MINUTE.mkdir(parents=True, exist_ok=True)

    def first(d: date) -> None:
        p = OPEN5 / f"{d}.parquet"
        if not p.exists():
            frame(get_bars(fetch, SYMBOLS, et(d, 9, 30), et(d, 9, 34), "5Min")).write_parquet(p)

    def minute(item: tuple[date, dict]) -> None:
        d, pick = item
        p = MINUTE / f"{d}.parquet"
        if not p.exists():
            syms = sorted({SYMBOLS[j] for j in pick["top"] + pick["rand"]})
            frame(get_bars(fetch, syms, et(d, 9, 35), et(d, 15, 59), "1Min")).write_parquet(p)

    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        for n, _ in enumerate(pool.map(first, f["days"]), 1):
            if n % 100 == 0:
                print(f"first bars: {n}/{len(f['days'])}", flush=True)
        todo = sorted(picks(f, load_open5()).items())
        for n, _ in enumerate(pool.map(minute, todo), 1):
            if n % 100 == 0:
                print(f"minute bars: {n}/{len(todo)}", flush=True)
    print("done", len(list(OPEN5.glob("*.parquet"))), len(list(MINUTE.glob("*.parquet"))), flush=True)


def trade(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, side: float, hi: float, lo: float, atr: float) -> float | None:
    """The rule's gross return for one name and day; None if it never triggered."""
    trig = hi if side > 0 else lo
    hit = np.nonzero(h >= trig if side > 0 else l <= trig)[0]
    if not len(hit):
        return None
    k = int(hit[0])
    entry = max(trig, o[k]) if side > 0 else min(trig, o[k])
    stop = entry * (1 - side * cfg["stop_atr"] * atr)
    if (side > 0 and l[k] <= stop) or (side < 0 and h[k] >= stop):
        return side * (stop / entry - 1)
    for b in range(k + 1, len(c)):
        if (side > 0 and l[b] <= stop) or (side < 0 and h[b] >= stop):
            fill = min(stop, o[b]) if side > 0 else max(stop, o[b])
            return side * (fill / entry - 1)
    return side * (c[-1] / entry - 1)


def perf(daily_r: np.ndarray) -> dict:
    eq = (1 + daily_r).cumprod()
    sd = daily_r.std()
    return dict(ann_return=float(eq[-1] ** (YEAR / len(daily_r)) - 1), sharpe=float(daily_r.mean() / sd * YEAR**0.5) if sd else 0.0,
                max_drawdown=float((eq / np.maximum.accumulate(eq) - 1).min()), mean_daily=float(daily_r.mean()),
                best_day=float(daily_r.max()), worst_day=float(daily_r.min()), days_up=float((daily_r > 0).mean()))


def book(t: pl.DataFrame, days: list[date], weight: float, cost: float) -> np.ndarray:
    got = t.group_by("day").agg(((pl.col("gross") - 2 * cost) * weight).sum().alias("r"))
    m = dict(zip(got["day"].to_list(), got["r"].to_list()))
    return np.array([m.get(d, 0.0) for d in days])


def cmd_run() -> None:
    f = features()
    pick = picks(f, load_open5())
    rows = []
    for d, p in sorted(pick.items()):
        path = MINUTE / f"{d}.parquet"
        if not path.exists():
            continue
        bars = pl.read_parquet(path)
        by = {s[0]: g.sort("t") for s, g in bars.group_by(["symbol"])}
        for group, names in (("top", p["top"]), ("rand", p["rand"])):
            for j in names:
                g = by.get(SYMBOLS[j])
                if g is None or not g.height:
                    continue
                side, hi, lo, atr = p["info"][j]
                r = trade(g["o"].to_numpy(), g["h"].to_numpy(), g["l"].to_numpy(), g["c"].to_numpy(), side, hi, lo, atr)
                if r is not None:
                    rows.append(dict(day=d, symbol=SYMBOLS[j], group=group, side=side, gross=r))
    t = pl.DataFrame(rows)
    days = sorted(d for d in pick if (MINUTE / f"{d}.parquet").exists())
    top, rand = t.filter(pl.col("group") == "top"), t.filter(pl.col("group") == "rand")
    cost = cfg["cost_bps"] / 1e4
    primary = book(top, days, cfg["weight_4x"], cost)
    adj = f["adj"]
    pos = {d: i for i, d in enumerate(f["days"])}
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = adj["close"][1:] / adj["close"][:-1] - 1
    hold_all = np.nanmean(rets, axis=1)
    hold = np.array([hold_all[pos[d] - 1] for d in days if pos[d] > 0])
    rng = np.random.default_rng(cfg["seed"])
    boot = primary[rng.integers(0, len(primary), (cfg["bootstrap_samples"], len(primary)))].mean(axis=1)
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

    def trades(x: pl.DataFrame) -> dict:
        g = x["gross"].to_numpy() - 2 * cost
        return dict(n=len(g), mean_bps=float(g.mean() * 1e4), win_rate=float((g > 0).mean()), avg_win_bps=float(g[g > 0].mean() * 1e4),
                    avg_loss_bps=float(g[g <= 0].mean() * 1e4))

    res: dict = dict(config="configs/stocks_in_play_v14.yaml", first=str(days[0]), last=str(days[-1]), days=len(days),
                     primary=perf(primary), one_x=perf(book(top, days, cfg["weight_1x"], cost)), hold=perf(hold),
                     mean_daily_ci=ci, top_trades=trades(top), random_trades=trades(rand),
                     random_book=perf(book(rand, days, cfg["weight_4x"], cost)),
                     longs=trades(top.filter(pl.col("side") > 0)), shorts=trades(top.filter(pl.col("side") < 0)),
                     by_cost={str(c): perf(book(top, days, cfg["weight_4x"], c / 1e4)) for c in cfg["side_cost_bps"]})
    sym = top.with_columns(((pl.col("gross") - 2 * cost) * cfg["weight_4x"]).alias("net")).group_by("symbol").agg(pl.col("net").sum().alias("sum"), pl.len().alias("n")).sort("sum", descending=True)
    best3 = sym["symbol"].to_list()[:3]
    res["top_symbols"] = sym.head(5).to_dicts()
    without3 = perf(book(top.filter(~pl.col("symbol").is_in(best3)), days, cfg["weight_4x"], cost))
    years = sorted({d.year for d in days})
    arr = np.array([d.year for d in days])
    hy = np.array([d.year for d in days if pos[d] > 0])
    by_year = {y: dict(book=float((1 + primary[arr == y]).prod() - 1), hold=float((1 + hold[hy == y]).prod() - 1)) for y in years}
    best2 = sorted(years, key=lambda y: -by_year[y]["book"])[:2]
    keep = ~np.isin(arr, best2)
    without2 = perf(primary[keep])
    res.update(by_year=by_year, without_best3=dict(symbols=best3, **without3), without_best2_years=dict(years=best2, **without2))
    res["tests"] = dict(
        test1=res["primary"]["sharpe"] > res["hold"]["sharpe"] and res["primary"]["ann_return"] >= res["hold"]["ann_return"] and ci[0] > 0,
        test2=res["top_trades"]["mean_bps"] > res["random_trades"]["mean_bps"],
        test3=without3["sharpe"] > res["hold"]["sharpe"] and without2["sharpe"] > res["hold"]["sharpe"])
    res["target"] = dict(target_daily=cfg["target_daily"], mean_daily=res["primary"]["mean_daily"],
                         days_at_or_above=float((primary >= cfg["target_daily"]).mean()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    {"fetch": cmd_fetch, "run": cmd_run}[sys.argv[1]]()
