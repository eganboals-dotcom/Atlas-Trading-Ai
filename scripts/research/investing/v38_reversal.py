"""Buying last week's losers on v31's list with the failed companies in it (see configs/reversal_v38.yaml, which holds the pre-registered tests).
run writes artifacts/reports/swing/v38_reversal.json. It needs v31's built data under data/processed/v31_pit.

    .venv/Scripts/python.exe scripts/research/investing/v38_reversal.py run
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v22_bottleneck as bottleneck  # noqa: E402
import v23_filings as f  # noqa: E402
import v31_point_in_time as pit  # noqa: E402
from atlas.swing.backtest import run  # noqa: E402

CONFIG = Path("configs/reversal_v38.yaml")
cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
OUT = Path("artifacts/reports/swing/v38_reversal.json")
assert cfg["v31_config"] == "configs/point_in_time_v31.yaml"  # v31 reads that file itself


def scores(close: np.ndarray, traded: np.ndarray, t: int) -> np.ndarray:
    """Each column's return from the close look + 1 sessions before t to the close before t; NaN unless both are real bars."""
    a, b = t - 1 - cfg["look"], t - 1
    return np.where(traded[a] & traded[b], close[b] / close[a] - 1, np.nan)


def books(m: pit.Market, days: list[int]) -> dict[str, list[list[int]]]:
    out: dict[str, list[list[int]]] = dict(losers=[], winners=[], hold=[], sector=[])
    n = m.n
    for t in days:
        pool = pit.pool_at(m.dollars, t, pit.cfg["pool_size"])
        s = np.full(n, np.nan)
        s[pool] = scores(m.p.close[:, :n], m.traded, t)[pool]
        count = int(np.isfinite(s).sum())
        if count < cfg["min_eligible"]:
            sys.exit(f"refused: only {count} eligible on {m.p.dates[t]}")
        group = np.full(n, -1)
        group[pool] = bottleneck.sectors(m.p.close[:, pool], m.sector_etfs, m.mkt, t)
        out["losers"].append(f.top(-s, cfg["top_share"]))  # the top of minus the score is the lowest score
        out["winners"].append(f.top(s, cfg["top_share"]))
        out["hold"].append(pool.tolist())
        out["sector"].append(f.by_sector(-s, group))
    return out


def returns(m: pit.Market, days: list[int], book: list[list[int]], cost: float) -> np.ndarray:
    x = np.zeros_like(m.p.close)
    for t, names in zip(days, book):
        x[t: t + cfg["every"], names] = x.shape[1] / len(names)
    return run(m.p, x, x, m.v0, m.end, cost, None, m.cash).returns


def turnover(book: list[list[int]]) -> float:
    """The mean share of the holding replaced at each rebalance."""
    return float(np.mean([1 - len(set(a) & set(b)) / len(b) for a, b in zip(book, book[1:])]))


def cmd_run() -> None:
    assert f.cfg["top_share"] == cfg["top_share"] and f.cfg["min_group"] == cfg["min_group"]  # by_sector reads v23's settings
    m, raw = pit.market(True), pit.market(False)
    days = list(range(m.v0, m.end, cfg["every"]))
    b = books(m, days)  # pools and scores use bars before each rebalance, the same in both markets
    rf, spy = m.cash[m.v0: m.end], m.spy[m.v0: m.end]
    r = {k: returns(m, days, x, cfg["cost_bps"]) for k, x in b.items()}
    s = {k: f.stats(x, rf) for k, x in r.items()} | dict(spy=f.stats(spy, rf), losers_at_side_cost=f.stats(returns(m, days, b["losers"], cfg["side_cost_bps"]), rf),
                                                     losers_failed_at_last_close=f.stats(returns(raw, days, b["losers"], cfg["cost_bps"]), rf))
    years = np.array([d.year for d in m.p.dates[m.v0: m.end]])
    half = np.arange(len(rf)) < len(rf) // 2
    x, h = r["losers"], r["hold"]
    grow = lambda a, y: round(float((1 + a[years == y]).prod() - 1) * 100, 1)  # noqa: E731
    by_year = {int(y): dict(rule=grow(x, y), hold=grow(h, y), spy=grow(spy, y)) for y in sorted(set(years))}
    best = sorted(by_year, key=lambda y: -(by_year[y]["rule"] - by_year[y]["hold"]))[:2]
    keep = ~np.isin(years, best)
    sharper = lambda i: f.stats(x[i], rf[i])["sharpe"] > f.stats(h[i], rf[i])["sharpe"]  # noqa: E731
    tests = dict(test1=f.beats(s["losers"], s["hold"]), test2=bool(s["losers"]["ann_return"] > s["winners"]["ann_return"]), test3=bool(sharper(keep)),
                 test4=bool(sharper(half) and sharper(~half)), test5=f.beats(s["sector"], s["hold"]), test6=f.beats(s["losers"], s["spy"]),
                 test7=f.beats(s["losers_at_side_cost"], s["hold"]))
    fails = {j: e for j, e, failed in m.stopped if failed}
    held = sum(1 for t, names in zip(days, b["losers"]) for j in names if j in fails and t <= fails[j] < t + cfg["every"])
    res = dict(config=str(CONFIG), config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(), window=f"{m.p.dates[m.v0]}..{m.p.dates[m.end - 1]}",
               rebalances=len(days), tests=tests, passed=all(tests.values()), figures=s, by_year=by_year, best_years=best,
               weekly_turnover=round(turnover(b["losers"]), 3), failed_companies_held_into_their_failure=held)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    {"run": cmd_run}[sys.argv[1]]()
