"""Trying to break v29's monthly rotation. NOT pre-registered, and it passes or fails nothing: v29's tests stand as they ran. Run after seeing its
result, because that result looked too good. Writes artifacts/reports/swing/v29_followup.json.

    .venv/Scripts/python.exe scripts/v29_followup.py
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import numpy as np

import v22_bottleneck as bottleneck
import v23_filings as f
import v27_finrl_rotation as v
import v29_monthly_rotation as slow
from atlas.swing.data import load_panel

OUT = Path("artifacts/reports/swing/v29_followup.json")


def phase(books: list[v.Book], dates: tuple, back: int) -> list[v.Book]:
    """The week-end book that many places before the month's last. Months with fewer week-ends use their first."""
    months: dict[tuple[int, int], list[v.Book]] = {}
    for b in books:
        months.setdefault((dates[b.t].year, dates[b.t].month), []).append(b)
    return [m[max(0, len(m) - 1 - back)] for m in months.values()]


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
    a, b = v.window(full.dates)
    cash = v.cash_for(full.dates)
    rf = cash[a:b]
    cost = v.cfg["cost_bps"]
    score = lambda books: f.stats(v.net(m, books, a, b, cost, cash), rf)

    res: dict = dict(note="not pre-registered", spy=f.stats(v.daily(full, m.spy)[a:b], rf), weekly=score(weekly))
    res["week_of_month"] = {f"{k}_before_last": score(phase(weekly, full.dates, k)) for k in range(4)}
    res["one_session_later"] = score([dataclasses.replace(k, t=k.t + 1) for k in slow.monthly(weekly, full.dates)])

    wide = copy.deepcopy(v.cfg["list_share"])
    v.cfg["list_share"] = {"growth": 9, "real": 5}
    spare = v.yearly_lists(full, n, etf, mkt)
    v.cfg["list_share"] = wide
    drop = {full.symbols.index(s) for s in ("NVDA", "TSLA")}
    without = {y: g | dict(growth=[j for j in g["growth"] if j not in drop][: wide["growth"]]) for y, g in spare.items()}
    res["without_nvda_tsla"] = score(slow.monthly(v.free_books(m, without), full.dates))

    v.cfg["list_share"] = {"growth": 30, "real": 20}
    broad = v.yearly_lists(full, n, etf, mkt)
    v.cfg["list_share"] = wide
    res["broad_lists_30_and_20"] = score(slow.monthly(v.free_books(m, broad), full.dates))
    res["broad_list_sizes"] = {y: {g: len(x) for g, x in groups.items()} for y, groups in list(broad.items())[:2]}

    res = json.loads(json.dumps(res, default=str))
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: (dict(ret=v_["ann_return"], sh=v_["sharpe"], dd=v_["max_drawdown"]) if isinstance(v_, dict) and "ann_return" in v_ else v_) for k, v_ in res.items()}, indent=1))


if __name__ == "__main__":
    main()
