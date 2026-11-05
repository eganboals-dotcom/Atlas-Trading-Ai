"""Late-day flow (see configs/late_day_flow_v16.yaml, which holds the pre-registered tests), on real 1-minute bars.

Writes artifacts/reports/swing/v16_late_day_flow.json. Reads local bars only; stops before the holdout. Unadjusted bars
span stock splits, so a day whose previous close is more than 40% from its first price is skipped.

    .venv/Scripts/python.exe scripts/research/trading/v16_late_day_flow.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import yaml

cfg = yaml.safe_load(Path("configs/late_day_flow_v16.yaml").read_text(encoding="utf-8"))
OUT = Path("artifacts/reports/swing/v16_late_day_flow.json")
SPLIT = 0.4


def load(dataset: str, symbols: list[str]) -> pl.DataFrame:
    return (pl.scan_parquet(f"data/processed/{dataset}/*.parquet")
            .filter((pl.col("event_type") == "trade") & pl.col("symbol").is_in(symbols))
            .select("symbol", "event_ts", "price").collect()
            .with_columns(pl.from_epoch("event_ts", time_unit="ns").dt.replace_time_zone("UTC")
                          .dt.convert_time_zone("America/New_York").alias("t"))
            .with_columns(pl.col("t").dt.date().alias("day"),
                          ((pl.col("t").dt.hour().cast(pl.Int32) * 60 + pl.col("t").dt.minute().cast(pl.Int32)) - 570).alias("m"))
            .filter(pl.col("day") < pl.lit(cfg["holdout_start"]).str.to_date()).sort("symbol", "day", "m"))


def rows_for(df: pl.DataFrame, cost: float) -> list[dict]:
    rows = []
    for (s,), g in df.group_by(["symbol"], maintain_order=True):
        prev = None
        for (d,), day in g.group_by(["day"], maintain_order=True):
            m, px = day["m"].to_numpy(), day["price"].to_numpy()
            a, b = np.searchsorted(m, 359, side="right") - 1, np.searchsorted(m, 360, side="right") - 1
            if prev is not None and m[a] == 359 and m[b] == 360 and abs(px[0] / prev - 1) < SPLIT:
                move = px[a] / prev - 1
                side = np.sign(move)
                gross = side * (px[-1] / px[b] - 1)
                rows.append(dict(symbol=s, day=d, move=move, gross=gross, net=gross - 2 * cost))
            prev = px[-1]
    return rows


def stats(x: np.ndarray, rng: np.random.Generator) -> dict:
    means = x[rng.integers(0, len(x), (cfg["bootstrap_samples"], len(x)))].mean(axis=1)
    return dict(n=len(x), mean_bps=float(x.mean() * 1e4), win_rate=float((x > 0).mean()),
                ci_bps=[float(np.percentile(means, 2.5) * 1e4), float(np.percentile(means, 97.5) * 1e4)])


def main() -> None:
    rng = np.random.default_rng(cfg["seed"])
    rows: list[dict] = []
    for ds, syms in cfg["datasets"].items():
        df = load(ds, syms)
        cost = (cfg["etf_cost_bps"] if syms[0] in cfg["etfs"] else cfg["stock_cost_bps"]) / 1e4
        rows += rows_for(df, cost)
        print(ds, len(rows), flush=True)
    d = pl.DataFrame(rows).sort("symbol", "day").with_columns(pl.col("move").abs().alias("size"))
    d = d.with_columns(pl.col("size").shift(1).rolling_quantile(cfg["big_pct"] / 100, window_size=cfg["lookback_days"],
                                                                min_samples=cfg["lookback_days"]).over("symbol").alias("cut")
                       ).drop_nulls("cut").with_columns((pl.col("size") > pl.col("cut")).alias("big"))
    etf, stock = d.filter(pl.col("symbol").is_in(cfg["etfs"])), d.filter(pl.col("symbol").is_in(cfg["stocks"]))
    start = date.fromisoformat(cfg["single_stock_etf_start"])
    big, quiet = etf.filter("big"), etf.filter(~pl.col("big"))
    sb = stock.filter("big")
    years = sorted({x.year for x in big["day"].to_list()})
    yearly = {y: big.filter(pl.col("day").dt.year() == y)["net"].mean() * 1e4 for y in years}
    best2 = sorted(years, key=lambda y: -yearly[y])[:2]
    rest = big.filter(~pl.col("day").dt.year().is_in(best2))
    res = dict(config="configs/late_day_flow_v16.yaml", first=str(d["day"].min()), last=str(d["day"].max()),
               etf_big=stats(big["net"].to_numpy(), rng), etf_big_gross=stats(big["gross"].to_numpy(), rng),
               etf_quiet=stats(quiet["net"].to_numpy(), rng),
               stocks_since=stats(sb.filter(pl.col("day") >= start)["net"].to_numpy(), rng),
               stocks_before=stats(sb.filter(pl.col("day") < start)["net"].to_numpy(), rng),
               etf_by_symbol={s: stats(big.filter(pl.col("symbol") == s)["net"].to_numpy(), rng) for s in cfg["etfs"]},
               stocks_since_by_symbol={s: stats(sb.filter((pl.col("symbol") == s) & (pl.col("day") >= start))["net"].to_numpy(), rng) for s in cfg["stocks"]},
               by_year_bps=yearly, without_best2=dict(years=best2, **stats(rest["net"].to_numpy(), rng)))
    res["tests"] = dict(test1=res["etf_big"]["ci_bps"][0] > 0, test2=res["etf_big"]["mean_bps"] > res["etf_quiet"]["mean_bps"],
                        test3=res["stocks_since"]["ci_bps"][0] > 0, test4=res["without_best2"]["mean_bps"] > 0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
