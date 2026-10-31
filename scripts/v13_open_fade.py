"""The opening fade (see configs/open_fade_v13.yaml, which holds the pre-registered tests), on real 1-minute bars.

Writes artifacts/reports/swing/v13_open_fade.json. Reads local bars only; stops before the holdout.

    .venv/Scripts/python.exe scripts/v13_open_fade.py
"""

from __future__ import annotations

import json
from datetime import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

cfg = yaml.safe_load(Path("configs/open_fade_v13.yaml").read_text(encoding="utf-8"))
OUT = Path("artifacts/reports/swing/v13_open_fade.json")
COST = cfg["cost_bps"] / 1e4
WEEKS = 252


def load(dataset: str, symbols: list[str]) -> dict[str, dict]:
    """{symbol: {day: (minute-of-session array, close array)}}, before the holdout."""
    df = (pl.scan_parquet(f"data/processed/{dataset}/*.parquet")
          .filter((pl.col("event_type") == "trade") & pl.col("symbol").is_in(symbols))
          .select("symbol", "event_ts", "price").collect()
          .with_columns(pl.from_epoch("event_ts", time_unit="ns").dt.replace_time_zone("UTC")
                        .dt.convert_time_zone("America/New_York").alias("t"))
          .with_columns(pl.col("t").dt.date().alias("day"),
                        ((pl.col("t").dt.hour().cast(pl.Int32) * 60 + pl.col("t").dt.minute().cast(pl.Int32)) - 570).alias("m"))
          .filter(pl.col("day") < pl.lit(cfg["holdout_start"]).str.to_date())
          .sort("symbol", "t"))
    out: dict[str, dict] = {s: {} for s in symbols}
    for (s, d), g in df.group_by(["symbol", "day"], maintain_order=True):
        out[s][d] = (g["m"].to_numpy(), g["price"].to_numpy())
    return out


def day_rows(sym: str, days: dict) -> list[dict]:
    rows = []
    for d, (m, px) in sorted(days.items()):
        if len(m) < 380 or m[0] != 0:
            continue
        a, b = np.searchsorted(m, 29, side="right") - 1, np.searchsorted(m, 30, side="right") - 1
        if m[a] != 29 or m[b] != 30:
            continue
        o, p10, entry, last = px[0], px[a], px[b], px[-1]
        move = p10 / o - 1
        after = px[b + 1:]
        side = -np.sign(move)  # the fade's direction
        crossed = side * (after - o) >= 0 if move else np.zeros(len(after), bool)
        hit = bool(crossed.any())
        fade = (side * (o / entry - 1) if hit else side * (last / entry - 1)) - 2 * COST
        ride = side * (last / entry - 1) - 2 * COST
        follow = -side * (last / entry - 1) - 2 * COST
        lh_a = np.searchsorted(m, 359, side="right") - 1  # 15:29 close, start of the last half hour
        last_half = np.sign(px[a] / o - 1) * (last / px[lh_a] - 1) - 2 * COST
        rows.append(dict(symbol=sym, day=d, move=move, fade=fade, ride=ride, follow=follow, last_half=last_half,
                         orb=orb(m, px), touched=hit, nearer=abs(last - o) < abs(p10 - o),
                         gave_back=(1 - (last - o) / (p10 - o)) if p10 != o else np.nan, eod=last, open=o))
    return rows


def orb(m: np.ndarray, px: np.ndarray) -> float:
    """Five-minute opening-range breakout: go the first five minutes' way past their extreme, stop at the other
    side, out at the close. Bars hold closes, so the range is of closes and fills are at the bar that crosses."""
    n = int(np.searchsorted(m, 4, side="right"))
    if n < 2:
        return 0.0
    hi, lo, up = px[:n].max(), px[:n].min(), px[n - 1] > px[0]
    if px[n - 1] == px[0] or hi == lo:
        return 0.0
    rest = px[n:]
    side, trig, stop = (1, hi, lo) if up else (-1, lo, hi)
    go = np.nonzero(side * (rest - trig) > 0)[0]
    if not len(go):
        return 0.0
    entry = rest[go[0]]
    after = rest[go[0] + 1:]
    out = np.nonzero(side * (after - stop) <= 0)[0]
    exit_px = after[out[0]] if len(out) else rest[-1]
    return side * (exit_px / entry - 1) - 2 * COST


def perf(daily: np.ndarray) -> dict:
    ann = float((1 + daily).prod() ** (WEEKS / len(daily)) - 1)
    sd = daily.std()
    eq = (1 + daily).cumprod()
    return dict(ann_return=ann, sharpe=float(daily.mean() / sd * WEEKS**0.5) if sd else 0.0,
                max_drawdown=float((eq / np.maximum.accumulate(eq) - 1).min()), mean_daily=float(daily.mean()),
                best_day=float(daily.max()), worst_day=float(daily.min()), days_up=float((daily > 0).mean()))


def ci(x: np.ndarray, rng: np.random.Generator) -> list[float]:
    means = x[rng.integers(0, len(x), (cfg["bootstrap_samples"], len(x)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def portfolio(df: pl.DataFrame, col: str, syms: list[str], cal: list) -> np.ndarray:
    """Equal capital per symbol, cash when a symbol has no trade: daily return of the book."""
    w = 1 / len(syms)
    got = df.group_by("day").agg(pl.col(col).sum().alias("r"))
    m = dict(zip(got["day"].to_list(), got["r"].to_list()))
    return np.array([m.get(d, 0.0) * w for d in cal])


def trades(df: pl.DataFrame, col: str, rng: np.random.Generator) -> dict:
    x = df[col].to_numpy()
    return dict(n=len(x), mean_bps=float(x.mean() * 1e4), win_rate=float((x > 0).mean()),
                avg_win_bps=float(x[x > 0].mean() * 1e4) if (x > 0).any() else 0.0,
                avg_loss_bps=float(x[x <= 0].mean() * 1e4) if (x <= 0).any() else 0.0, ci_bps=[v * 1e4 for v in ci(x, rng)])


def main() -> None:
    rng = np.random.default_rng(cfg["seed"])
    rows: list[dict] = []
    for ds, syms in cfg["datasets"].items():
        for s, days in load(ds, syms).items():
            rows += day_rows(s, days)
            print(s, len(days), flush=True)
    df = pl.DataFrame(rows).sort("symbol", "day").with_columns(pl.col("move").abs().alias("size"))
    df = df.with_columns(pl.col("size").shift(1).rolling_quantile(cfg["volatile_pct"] / 100, window_size=cfg["lookback_days"],
                                                                  min_samples=cfg["lookback_days"]).over("symbol").alias("cut")
                         ).drop_nulls("cut").with_columns((pl.col("size") > pl.col("cut")).alias("volatile"))
    df = df.with_columns((pl.col("gave_back") >= cfg["corrected_fraction"]).alias("corrected"))
    mkt, names = cfg["market_symbols"], cfg["single_names"]
    res: dict = dict(config="configs/open_fade_v13.yaml", first=str(df["day"].min()), last=str(df["day"].max()))

    def block(syms: list[str]) -> dict:
        d = df.filter(pl.col("symbol").is_in(syms))
        vol, quiet = d.filter("volatile"), d.filter(~pl.col("volatile"))
        cal = sorted(d["day"].unique().to_list())
        rates = {k: dict(volatile=float(vol[k].mean()), quiet=float(quiet[k].mean())) for k in ("corrected", "touched", "nearer")}
        book = {k: perf(portfolio(vol, k, syms, cal)) for k in ("fade", "ride", "follow")}
        book["fade_quiet"] = perf(portfolio(quiet, "fade", syms, cal))
        return dict(symbols=syms, days=len(cal), rates=rates, book=book,
                    fade_trades=trades(vol, "fade", rng), quiet_trades=trades(quiet, "fade", rng),
                    follow_trades=trades(vol, "follow", rng))

    res["market"], res["single_names"] = block(mkt), block(names)
    m = df.filter(pl.col("symbol").is_in(mkt))
    cal = sorted(m["day"].unique().to_list())
    wide = m.pivot(on="symbol", index="day", values="eod").sort("day").drop_nulls()
    ret = np.mean([wide[s].to_numpy()[1:] / wide[s].to_numpy()[:-1] - 1 for s in mkt], axis=0)
    res["hold"] = dict(perf(ret), note="price only from raw bars: dividends of about 1.3% a year are left out, which flatters the fade")
    vol = m.filter("volatile")
    t2 = res["market"]["book"]["fade"]
    res["tests"] = dict(
        test1=res["market"]["rates"]["corrected"]["volatile"] >= cfg["claim"]
        and res["market"]["rates"]["corrected"]["volatile"] > res["market"]["rates"]["corrected"]["quiet"],
        test2=t2["sharpe"] > res["hold"]["sharpe"] and t2["ann_return"] >= res["hold"]["ann_return"]
        and res["market"]["fade_trades"]["ci_bps"][0] > 0,
        control_beats_quiet=res["market"]["fade_trades"]["mean_bps"] > res["market"]["quiet_trades"]["mean_bps"],
        control_beats_follow=t2["ann_return"] > res["market"]["book"]["follow"]["ann_return"])
    daily = portfolio(vol, "fade", mkt, cal)
    by_year = {}
    for y in sorted({d.year for d in cal}):
        k = np.array([d.year == y for d in cal])
        hy = np.array([d.year == y for d in wide["day"].to_list()[1:]])
        by_year[y] = dict(fade=float((1 + daily[k]).prod() - 1), hold=float((1 + ret[hy]).prod() - 1))
    res["by_year"] = by_year
    res["side"] = dict(
        last_half_hour_momentum={s: trades(df.filter(pl.col("symbol") == s), "last_half", rng) for s in mkt},
        orb_5min=dict(market=perf(portfolio(m, "orb", mkt, cal)),
                      single_names=perf(portfolio(df.filter(pl.col("symbol").is_in(names)), "orb", names, sorted(df["day"].unique().to_list())))))
    res["target"] = dict(target_daily=cfg["target_daily"], fade_mean_daily=float(daily.mean()),
                         times_short=float(cfg["target_daily"] / daily.mean()) if daily.mean() > 0 else None,
                         days_at_or_above=float((daily >= cfg["target_daily"]).mean()),
                         yearly_if_compounded=(1 + cfg["target_daily"]) ** WEEKS - 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
