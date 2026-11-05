"""The Noise Area rule on markets not yet looked at (see configs/noise_area_breadth_v15b.yaml, which holds the
pre-registered tests). Reuses scripts/research/trading/v15_noise_area.py unchanged. Writes artifacts/reports/swing/v15b_breadth.json.

    .venv/Scripts/python.exe scripts/research/trading/v15b_noise_area_breadth.py fetch
    .venv/Scripts/python.exe scripts/research/trading/v15b_noise_area_breadth.py run
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import yaml

import v15_noise_area as v

cfg = yaml.safe_load(Path("configs/noise_area_breadth_v15b.yaml").read_text(encoding="utf-8"))
OUT = Path("artifacts/reports/swing/v15b_breadth.json")


def hold_series(sym: str, dates: list[date], sim: dict) -> np.ndarray:
    for d in cfg["daily_dirs"]:
        path = Path(d) / f"{sym}.parquet"
        if path.exists():
            f = pl.read_parquet(path).sort("date")
            px = dict(zip(f["date"].to_list(), f["close"].to_list()))
            ds = sorted(px)
            pos = {x: i for i, x in enumerate(ds)}
            return np.array([px[x] / px[ds[pos[x] - 1]] - 1 for x in dates])
    closes = np.array([r["day"]["close"] for r in sim["rows"]])  # no daily file (AMD, which pays no dividend): the bars' own closes
    return np.r_[0.0, closes[1:] / closes[:-1] - 1]


def book(symbols: list[str], cost: float) -> dict:
    per: dict = {}
    series, holds, gross, dates = {}, {}, {}, {}
    for s in symbols:
        sim = v.simulate(s)
        dates[s] = sim["dates"]
        series[s], gross[s], holds[s] = v.series(sim, cost), v.series(sim, 0.0), hold_series(s, sim["dates"], sim)
        per[s] = dict(rule=v.perf(series[s]), gross=v.perf(gross[s]), hold=v.perf(holds[s]), corr=float(np.corrcoef(series[s], holds[s])[0, 1]))
    common = sorted(set.intersection(*(set(d) for d in dates.values())))
    idx = {s: {d: i for i, d in enumerate(dates[s])} for s in symbols}
    pick = lambda m, s: m[s][[idx[s][d] for d in common]]
    r = np.mean([pick(series, s) for s in symbols], axis=0)
    h = np.mean([pick(holds, s) for s in symbols], axis=0)
    after = np.array([d >= date.fromisoformat(cfg["published"]) for d in common])
    return dict(symbols=per, book=v.perf(r), hold=v.perf(h), after=v.perf(r[after]), hold_after=v.perf(h[after]),
                corr=float(np.corrcoef(r, h)[0, 1]), winners=int(sum(per[s]["rule"]["ann_return"] > 0 for s in symbols)))


def stress(sym: str, cost: float) -> dict:
    out = {}
    hold = None
    for delay in (0, 1, 5):
        sim = v.simulate(sym, delay)
        r = v.series(sim, cost)
        hold = v.hold_returns(sym, sim["dates"]) if hold is None else hold
        out[f"delay_{delay}"] = v.perf(r)
    sim = v.simulate(sym)
    r = v.series(sim, cost)
    years = np.array([d.year for d in sim["dates"]])
    top = np.argsort(-r)[:10]
    drop = r.copy()
    drop[top] = 0.0
    out.update(by_year={int(y): dict(rule=round(float((1 + r[years == y]).prod() - 1) * 100, 1), hold=round(float((1 + hold[years == y]).prod() - 1) * 100, 1)) for y in sorted(set(years))},
               plain_1x=v.perf(v.series(sim, cost, sized=False)), without_10_best_days=v.perf(drop), hold=v.perf(hold),
               long_only=v.perf(v.series(sim, cost, key="long")))
    return out


def cmd_fetch() -> None:
    v.cfg["symbols"] = cfg["etfs"] + cfg["stocks"]
    v.cmd_fetch()


def cmd_run() -> None:
    etf, stock = book(cfg["etfs"], cfg["etf_cost_bps"]), book(cfg["stocks"], cfg["stock_cost_bps"])
    res: dict = dict(config="configs/noise_area_breadth_v15b.yaml", etfs=etf, stocks=stock,
                     stress={s: stress(s, c) for s, c in cfg["measured_cost_bps"].items()})
    res["tests"] = dict(test1=etf["book"]["sharpe"] > etf["hold"]["sharpe"], test2=etf["winners"] >= 6,
                        test3=stock["book"]["sharpe"] > stock["hold"]["sharpe"], test4=etf["after"]["ann_return"] > 0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    f = lambda d: (round(d["ann_return"] * 100, 1), round(d["sharpe"], 2), round(d["max_drawdown"] * 100, 1))
    print(res["tests"])
    for name, g in (("etfs", etf), ("stocks", stock)):
        print(name, "book", f(g["book"]), "hold", f(g["hold"]), "after", f(g["after"]), "hold after", f(g["hold_after"]), "corr", round(g["corr"], 2), "winners", g["winners"])
        for s, x in g["symbols"].items():
            print(" ", s, "rule", f(x["rule"]), "gross", f(x["gross"]), "hold", f(x["hold"]))
    for s, x in res["stress"].items():
        print(s, {k: f(a) for k, a in x.items() if k not in ("by_year",)})
        print(x["by_year"])


if __name__ == "__main__":
    {"fetch": cmd_fetch, "run": cmd_run}[sys.argv[1]]()
