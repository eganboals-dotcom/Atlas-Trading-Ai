"""v27's rotation traded monthly (see configs/monthly_rotation_v29.yaml, which holds the pre-registered tests). Writes
artifacts/reports/swing/v29_monthly_rotation.json.

    .venv/Scripts/python.exe scripts/v29_monthly_rotation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

import v22_bottleneck as bottleneck
import v23_filings as f
import v26_deflated_sharpe as deflate
import v27_finrl_rotation as v
from atlas.swing.backtest import run, simulate
from atlas.swing.data import Panel, load_panel
from atlas.swing.families import HOLD, exposures

cfg = yaml.safe_load(Path("configs/monthly_rotation_v29.yaml").read_text(encoding="utf-8"))
OUT = Path(cfg["report"])


def monthly(books: list[v.Book], dates: tuple) -> list[v.Book]:
    """The last week-end book of each month."""
    return [b for b, c in zip(books, [*books[1:], None]) if c is None or dates[b.t].month != dates[c.t].month]


def hold_list(p: Panel, lists: dict[int, dict[str, list[int]]], a: int, b: int, cost: float, cash: np.ndarray) -> np.ndarray:
    n = p.close.shape[1]
    x = np.zeros_like(p.close)
    for t in range(a - 1, b):
        names = [j for g in lists[p.dates[t].year].values() for j in g]
        x[t, names] = n / len(names)
    return run(p, x, x, a, b, cost, None, cash).returns


def main() -> None:
    full, stocks = v.free_panel()
    e = load_panel(Path(v.cfg["etf_dir"]), bottleneck.cfg["etfs"] + [bottleneck.cfg["market"]])
    at = {d: i for i, d in enumerate(e.dates)}
    idx = [at[d] for d in full.dates]
    etf, mkt = e.close[idx, :-1], e.close[idx, -1]
    n = stocks.close.shape[1]
    lists = v.yearly_lists(full, n, etf, mkt)
    m = v.rotation(full)
    weekly = v.free_books(m, lists)
    books = monthly(weekly, full.dates)
    a, b = v.window(full.dates)
    cash = v.cash_for(full.dates)
    rf = cash[a:b]
    dates = list(full.dates[a:b])
    cost = v.cfg["cost_bps"]
    r = v.net(m, books, a, b, cost, cash)
    spy = v.daily(full, m.spy)[a:b]
    hold = simulate(stocks, exposures(stocks, *HOLD), a, b, cost, cash).returns
    plain = hold_list(full, lists, a, b, cost, cash)
    s = dict(monthly=f.stats(r, rf), spy=f.stats(spy, rf), hold=f.stats(hold, rf), hold_the_list=f.stats(plain, rf))
    luck = v.luck_tests(r, spy, rf, dates)
    rng = v.draws(m, books, a, b, cash, rf)
    tests = dict(t1=f.beats(s["monthly"], s["spy"]), t2=f.beats(s["monthly"], s["hold"]), t3=f.beats(s["monthly"], s["hold_the_list"]), t4=luck["not_luck"], t5=luck["both_halves"],
                 t6=bool(s["monthly"]["ann_return"] > np.percentile(rng, 95)))

    # 2018, a side result. Each year's list comes from the prior year, as above.
    ok = [i for i, d in enumerate(full.dates) if cfg["before_first"] <= d <= cfg["before_last"]]
    pa, pb = ok[0], ok[-1] + 1
    early = v.yearly_lists(full, n, etf, mkt, cfg["before_list_years"])
    past = monthly([v.choose(m, k, early[full.dates[t].year]) for k, t in enumerate(m.ends) if full.dates[t].year in early], full.dates)
    r_past, spy_past = v.net(m, past, pa, pb, cost, cash), v.daily(full, m.spy)[pa:pb]

    d = r - spy
    sharpes = json.loads(Path("artifacts/reports/swing/v26_deflated_sharpe.json").read_text(encoding="utf-8"))["trial_gap_sharpes"]
    var = float(np.var(sharpes, ddof=1)) / 252
    res = dict(config="configs/monthly_rotation_v29.yaml", window=f"{dates[0]}..{dates[-1]}", tests=tests, passed=all(tests.values()), figures=s, luck=luck, books=len(books),
               random=dict(p50=round(float(np.median(rng)), 4), p95=round(float(np.percentile(rng, 95)), 4)), side_costs={str(c): f.stats(v.net(m, books, a, b, c, cash), rf) for c in cfg["side_cost_bps"]},
               shape=v.shape_of(m, books, a), by_year=v.by_year(dates, r, dict(spy=spy, hold=hold, hold_the_list=plain)), gap_to_spy_95=v.gap_interval(d),
               before=dict(window=f"{full.dates[pa]}..{full.dates[pb - 1]}", monthly=f.stats(r_past, cash[pa:pb]), spy=f.stats(spy_past, cash[pa:pb])),
               deflated_gap_to_spy={n_: round(deflate.deflated(d, n_, var), 3) for n_ in (25, 100)})
    OUT.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
