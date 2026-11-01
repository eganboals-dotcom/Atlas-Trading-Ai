"""Intraday momentum beyond the Noise Area (see configs/noise_area_v15.yaml, which holds the pre-registered tests).

fetch downloads 1-minute OHLCV bars for the configured ETFs (read-only Alpaca market data, regular hours, stopping
before the holdout). run scores them and writes artifacts/reports/swing/v15_noise_area.json.

    .venv/Scripts/python.exe scripts/v15_noise_area.py fetch
    .venv/Scripts/python.exe scripts/v15_noise_area.py run
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from atlas.feeds.alpaca import HISTORY_BARS_PATH, HISTORY_PATHS, http_fetch, iter_history, load_credentials

cfg = yaml.safe_load(Path("configs/noise_area_v15.yaml").read_text(encoding="utf-8"))
BARS = Path("data/processed/etf_1min_ohlcv")
OUT = Path("artifacts/reports/swing/v15_noise_area.json")
YEAR = 252
N = 390
GRID = range(30, 361, 30)  # minutes after 09:30 at which positions may change: 10:00 to 15:30


def cmd_fetch() -> None:
    fetch = http_fetch(load_credentials(Path(".env")), timeout_s=60, allowed=HISTORY_PATHS)
    last = (date.fromisoformat(cfg["holdout_start"]) - timedelta(days=1)).isoformat()
    BARS.mkdir(parents=True, exist_ok=True)
    for sym in cfg["symbols"]:
        path = BARS / f"{sym}.parquet"
        if path.exists():
            continue
        rows = [{k: b[k] for k in ("t", "o", "h", "l", "c", "v")}
                for b in iter_history(fetch, HISTORY_BARS_PATH, sym, f"{cfg['first']}T00:00:00Z", f"{last}T23:59:59Z",
                                      cfg["feed"], min_interval_s=cfg["min_interval_s"],
                                      extra={"timeframe": "1Min", "adjustment": "split"})]
        df = (pl.DataFrame(rows).with_columns(pl.col("t").str.to_datetime(time_zone="UTC", time_unit="ns")
                                              .dt.convert_time_zone("America/New_York").alias("et"))
              .with_columns(pl.col("et").dt.date().alias("day"),
                            (pl.col("et").dt.hour().cast(pl.Int32) * 60 + pl.col("et").dt.minute().cast(pl.Int32) - 570).alias("m"))
              .filter((pl.col("m") >= 0) & (pl.col("m") < N)).select("day", "m", "o", "h", "l", "c", "v").sort("day", "m"))
        df.write_parquet(path)
        print(sym, df.height, df["day"].n_unique(), flush=True)


def load_days(sym: str) -> list[dict]:
    """One dict per session: open, closes and vwap on a full 390-minute grid (gaps carried forward), and the last minute."""
    df = pl.read_parquet(BARS / f"{sym}.parquet")
    out = []
    for (d,), g in df.group_by(["day"], maintain_order=True):
        m = g["m"].to_numpy()
        if m[0] > 5:  # no real open (a halted start would need one)
            continue
        c, o = np.full(N, np.nan), np.full(N, np.nan)
        pv, vol = np.zeros(N), np.zeros(N)
        c[m], o[m] = g["c"].to_numpy(), g["o"].to_numpy()
        typical = (g["h"].to_numpy() + g["l"].to_numpy() + g["c"].to_numpy()) / 3
        pv[m], vol[m] = typical * g["v"].to_numpy(), g["v"].to_numpy()
        last = int(m[-1])
        for i in range(1, N):  # carry prices across empty minutes: the last trade stands
            if np.isnan(c[i]):
                c[i] = c[i - 1]
            if np.isnan(o[i]):
                o[i] = c[i - 1]
        cv = np.cumsum(vol)
        with np.errstate(divide="ignore", invalid="ignore"):
            vwap = np.cumsum(pv) / cv
        out.append(dict(day=d, open=float(g["o"][0]), c=c, o=o, vwap=vwap, last=last, close=float(c[last])))
    return out


def plan(day: dict, upper: np.ndarray, lower: np.ndarray, long_only: bool = False, use_vwap: bool = True) -> list[tuple[int, float]]:
    """(minute, target position) changes. A decision at grid minute g reads the bar that ended at g and fills at the
    open of bar g; the minute's sigma is that of the bar read."""
    changes, pos = [], 0.0
    for g in GRID:
        if g > day["last"]:
            break
        k = g - 1
        px, up, lo = day["c"][k], upper[k], lower[k]
        v = day["vwap"][k]
        if pos > 0:
            keep = px >= (max(up, v) if use_vwap else lo)
        elif pos < 0:
            keep = px <= (min(lo, v) if use_vwap else up)
        else:
            keep = False
        if keep:
            want = pos
        elif pos > 0:
            want = -1.0 if px < lo else 0.0
        elif pos < 0:
            want = 1.0 if px > up else 0.0
        else:
            want = 1.0 if px > up else -1.0 if px < lo else 0.0
        if long_only:
            want = max(want, 0.0)
        if want != pos:
            changes.append((g, want))
            pos = want
    return changes


def legs(day: dict, changes: list[tuple[int, float]], delay: int = 0) -> list[tuple[float, float, float]]:
    """(position, entry price, exit price) per held stretch, out at the last bar's close. ``delay`` fills that many
    minutes late."""
    out, pos, entry = [], 0.0, 0.0
    for g, want in changes:
        px = day["o"][g + delay]
        if pos:
            out.append((pos, entry, px))
        pos, entry = want, px
    if pos:
        out.append((pos, entry, day["close"]))
    return out


def day_return(day: dict, lg: list[tuple[float, float, float]], lev: float, cost_frac: float, per_share: float = 0.0) -> float:
    """Return on capital: shares are set at the open from ``lev``, so each leg earns its move over that open."""
    o = day["open"]
    gross = sum(p * (x - e) for p, e, x in lg)
    fees = sum((e + x) * cost_frac + 2 * per_share for _, e, x in lg)
    return lev * (gross - fees) / o


def simulate(sym: str, delay: int = 0) -> dict:
    days = load_days(sym)
    lb = cfg["lookback_days"]
    closes = np.array([d["close"] for d in days])
    daily_ret = np.r_[np.nan, closes[1:] / closes[:-1] - 1]
    moves = np.vstack([np.abs(d["c"] / d["open"] - 1) for d in days])
    rows = []
    for i in range(lb + 1, len(days)):
        d = days[i]
        sigma = np.nanmean(moves[i - lb:i], axis=0)
        prev = days[i - 1]["close"]
        upper, lower = max(d["open"], prev) * (1 + sigma), min(d["open"], prev) * (1 - sigma)
        sd = float(np.std(daily_ret[i - lb:i], ddof=1))
        lev = min(cfg["max_leverage"], cfg["sigma_target"] / sd) if sd > 0 else cfg["max_leverage"]
        rows.append(dict(day=d, lev=lev, base=legs(d, plan(d, upper, lower), delay),
                         long=legs(d, plan(d, upper, lower, long_only=True), delay)))
    return dict(rows=rows, dates=[r["day"]["day"] for r in rows])


def series(sim: dict, cost_bps: float, sized: bool = True, key: str = "base", per_share: float = 0.0) -> np.ndarray:
    c = cost_bps / 1e4
    return np.array([day_return(r["day"], r[key], r["lev"] if sized else 1.0, c, per_share) for r in sim["rows"]])


def perf(r: np.ndarray) -> dict:
    eq = (1 + r).cumprod()
    sd = r.std()
    return dict(ann_return=float(eq[-1] ** (YEAR / len(r)) - 1), sharpe=float(r.mean() / sd * YEAR**0.5) if sd else 0.0,
                ann_vol=float(sd * YEAR**0.5), max_drawdown=float((eq / np.maximum.accumulate(eq) - 1).min()),
                mean_daily=float(r.mean()), best_day=float(r.max()), worst_day=float(r.min()), days_up=float((r > 0).mean()), days=len(r))


def hold_returns(sym: str, dates: list[date]) -> np.ndarray:
    f = pl.read_parquet(Path(cfg["daily_dir"]) / f"{sym}.parquet").sort("date")
    px = dict(zip(f["date"].to_list(), f["close"].to_list()))
    ds = sorted(px)
    pos = {d: i for i, d in enumerate(ds)}
    return np.array([px[d] / px[ds[pos[d] - 1]] - 1 for d in dates])


def main_run() -> None:
    rng = np.random.default_rng(cfg["seed"])
    res: dict = dict(config="configs/noise_area_v15.yaml")
    sims = {s: simulate(s) for s in cfg["symbols"]}
    book, hold = {}, {}
    for s, sim in sims.items():
        book[s] = series(sim, cfg["cost_bps"])
        hold[s] = hold_returns(s, sim["dates"])
        n_trades = np.array([len(r["base"]) for r in sim["rows"]])
        res[s] = dict(first=str(sim["dates"][0]), last=str(sim["dates"][-1]), primary=perf(book[s]), hold=perf(hold[s]),
                      plain_1x=perf(series(sim, cfg["cost_bps"], sized=False)), long_only=perf(series(sim, cfg["cost_bps"], key="long")),
                      paper_cost=perf(series(sim, 0.0, per_share=cfg["paper_cost_per_share"])),
                      gross=perf(series(sim, 0.0)), corr_with_hold=float(np.corrcoef(book[s], hold[s])[0, 1]),
                      avg_leverage=float(np.mean([r["lev"] for r in sim["rows"]])), trades_per_day=float(n_trades.mean()),
                      days_in_market=float((n_trades > 0).mean()),
                      by_cost={str(c): perf(series(sim, c)) for c in cfg["side_cost_bps"]})
    sim = sims["SPY"]
    r, h = book["SPY"], hold["SPY"]
    dates = sim["dates"]
    boot = r[rng.integers(0, len(r), (cfg["bootstrap_samples"], len(r)))].mean(axis=1)
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    years = np.array([d.year for d in dates])
    by_year = {int(y): dict(book=float((1 + r[years == y]).prod() - 1), plain_1x=float((1 + series(sim, cfg["cost_bps"], sized=False)[years == y]).prod() - 1),
                            hold=float((1 + h[years == y]).prod() - 1)) for y in sorted(set(years))}
    best2 = sorted(by_year, key=lambda y: -by_year[y]["book"])[:2]
    keep = ~np.isin(years, best2)
    cut = date.fromisoformat(cfg["paper_end"])
    after = np.array([d >= cut for d in dates])
    res.update(mean_daily_ci=ci, by_year=by_year, after_paper=dict(book=perf(r[after]), hold=perf(h[after])),
               without_best2=dict(years=best2, book=perf(r[keep]), hold=perf(h[keep])))
    c = cfg["cost_bps"] / 1e4
    flips = []
    for _ in range(cfg["control_flips"]):
        total = np.zeros(len(r))
        for i, row in enumerate(sim["rows"]):
            lg = [(p * (1.0 if rng.random() < 0.5 else -1.0), e, x) for p, e, x in row["base"]]
            total[i] = day_return(row["day"], lg, row["lev"], c)
        flips.append(total.mean() / total.std() * YEAR**0.5)
    flips = np.array(flips)
    res["control"] = dict(flip_sharpe_95=float(np.percentile(flips, 95)), flip_sharpe_median=float(np.median(flips)), beat_fraction=float((perf(r)["sharpe"] > flips).mean()))
    common = set(sims["SPY"]["dates"]) & set(sims["QQQ"]["dates"]) & set(sims["IWM"]["dates"])
    idx = {s: [i for i, d in enumerate(sims[s]["dates"]) if d in common] for s in sims}
    res["blend"] = perf(np.mean([book[s][idx[s]] for s in sims], axis=0))
    p, hp = res["SPY"]["primary"], res["SPY"]["hold"]
    res["tests"] = dict(
        test1=p["sharpe"] > hp["sharpe"] and p["ann_return"] >= hp["ann_return"] and ci[0] > 0,
        test2=res["after_paper"]["book"]["ann_return"] > 0 and res["after_paper"]["book"]["sharpe"] > res["after_paper"]["hold"]["sharpe"],
        test3=all(res[s]["primary"]["sharpe"] > res[s]["hold"]["sharpe"] for s in ("QQQ", "IWM")),
        test4=res["without_best2"]["book"]["sharpe"] > res["without_best2"]["hold"]["sharpe"],
        control=res["control"]["beat_fraction"] >= 0.95)
    res["target"] = dict(target_daily=cfg["target_daily"], mean_daily=p["mean_daily"], days_at_or_above=float((r >= cfg["target_daily"]).mean()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    {"fetch": cmd_fetch, "run": main_run}[sys.argv[1]]()
