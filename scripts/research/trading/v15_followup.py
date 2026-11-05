"""Follow-up to v15, after its pre-registered tests failed. NOT pre-registered: it passes or fails nothing.

v15's 1 bp was far above the paper's cost and the result hangs on it, so this measures real SPY and QQQ spreads from
sampled SIP quotes (data/processed/alpaca_sip_hist) and shows the rule at several costs, by year next to the paper's own
table, after publication, and added on top of Hold. Writes artifacts/reports/swing/v15_followup.json.

    .venv/Scripts/python.exe scripts/research/trading/v15_followup.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

import v15_noise_area as v

OUT = Path("artifacts/reports/swing/v15_followup.json")
PAPER_YEARS = {2016: -12.8, 2017: -6.9, 2018: 61.1, 2019: 6.9, 2020: 26.8, 2021: 34.8, 2022: 24.4, 2023: 37.2, 2024: 32.2}  # its appendix table
COSTS = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)
PUBLISHED = date(2024, 5, 10)  # first version of the paper


def spreads() -> dict:
    df = (pl.scan_parquet("data/processed/alpaca_sip_hist/*.parquet").filter(pl.col("event_type") == "quote")
          .select("symbol", "event_ts", "bid", "ask").collect()
          .filter((pl.col("bid") > 0) & (pl.col("ask") > pl.col("bid")))
          .with_columns(pl.from_epoch("event_ts", time_unit="ns").dt.replace_time_zone("UTC").dt.convert_time_zone("America/New_York").alias("t"))
          .with_columns(pl.col("t").dt.date().alias("day"), ((pl.col("t").dt.hour().cast(pl.Int32) * 60 + pl.col("t").dt.minute().cast(pl.Int32)) - 570).alias("m"),
                        ((pl.col("ask") - pl.col("bid")) / ((pl.col("ask") + pl.col("bid")) / 2) * 1e4).alias("bps"))
          .filter(pl.col("m") >= 15))
    out = {}
    for (s,), g in df.group_by(["symbol"]):
        out[s] = dict(quotes=g.height, days=g["day"].n_unique(), median_spread_bps=float(g["bps"].median()), mean_spread_bps=float(g["bps"].mean()),
                      p95_spread_bps=float(g["bps"].quantile(0.95)), half_spread_bps=float(g["bps"].median() / 2),
                      since_2022_median=float(g.filter(pl.col("day") >= date(2022, 1, 1))["bps"].median() or 0))
    return out


def main() -> None:
    res: dict = dict(note="not pre-registered; follow-up to configs/noise_area_v15.yaml", spreads=spreads())
    for sym in v.cfg["symbols"]:
        sim = v.simulate(sym)
        dates = sim["dates"]
        h = v.hold_returns(sym, dates)
        after = np.array([d >= PUBLISHED for d in dates])
        years = np.array([d.year for d in dates])
        one = {}
        for c in COSTS:
            r = v.series(sim, c)
            combo = h + 0.5 * r  # Hold plus the rule at half size on top: needs margin, and the same days as Hold
            one[str(c)] = dict(all=v.perf(r), after_published=v.perf(r[after]), hold_after=v.perf(h[after]), hold_plus_half=v.perf(combo),
                               hold_plus_half_after=v.perf(combo[after]))
        res[sym] = dict(hold=v.perf(h), by_cost=one, corr_with_hold=float(np.corrcoef(v.series(sim, 0.2), h)[0, 1]))
        if sym == "SPY":
            res["SPY_vs_paper"] = {int(y): dict(gross=round(float((1 + v.series(sim, 0.0)[years == y]).prod() - 1) * 100, 1),
                                                at_paper_cost=round(float((1 + v.series(sim, 0.0, per_share=v.cfg["paper_cost_per_share"])[years == y]).prod() - 1) * 100, 1),
                                                paper=PAPER_YEARS.get(int(y))) for y in sorted(set(years))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    f = lambda d: {k: round(x, 3) for k, x in d.items() if k in ("ann_return", "sharpe", "max_drawdown")}
    print(res["spreads"])
    print(res["SPY_vs_paper"])
    for s in v.cfg["symbols"]:
        print(s, "hold", f(res[s]["hold"]), "corr", round(res[s]["corr_with_hold"], 3))
        for c, x in res[s]["by_cost"].items():
            print(" ", c, "all", f(x["all"]), "after", f(x["after_published"]), "holdAfter", f(x["hold_after"]), "H+.5", f(x["hold_plus_half"]), "H+.5after", f(x["hold_plus_half_after"]))


if __name__ == "__main__":
    main()
