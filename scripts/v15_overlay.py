"""Hold plus the Noise Area rule on top, on SPY and QQQ. NOT pre-registered: it passes or fails nothing.

SPY and QQQ were picked after seeing their results, and the rule failed on the other 17 markets tried, so read this as
a candidate to freeze and test on new days, not as evidence. Writes artifacts/reports/swing/v15_overlay.json.

    .venv/Scripts/python.exe scripts/v15_overlay.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

import v15_noise_area as v

OUT = Path("artifacts/reports/swing/v15_overlay.json")
COST = {"SPY": 0.2, "QQQ": 0.3}
PUBLISHED = date(2024, 5, 10)
SIZE = 0.5


def main() -> None:
    res: dict = dict(note="not pre-registered; SPY and QQQ chosen after seeing results")
    sims = {(s, d): v.simulate(s, d) for s in COST for d in (0, 1)}
    dates = sorted(set(sims[("SPY", 0)]["dates"]) & set(sims[("QQQ", 0)]["dates"]))
    pick = lambda sim, x: np.array(x)[[sim["dates"].index(d) for d in dates]]
    idx = {s: {d: i for i, d in enumerate(sims[(s, 0)]["dates"])} for s in COST}

    def aligned(s: str, x: np.ndarray) -> np.ndarray:
        return x[[idx[s][d] for d in dates]]

    hold = np.mean([aligned(s, v.hold_returns(s, sims[(s, 0)]["dates"])) for s in COST], axis=0)
    after = np.array([d >= PUBLISHED for d in dates])
    years = np.array([d.year for d in dates])
    res["hold"], res["hold_after"] = v.perf(hold), v.perf(hold[after])

    def rule(delay: int, extra_bps: float = 0.0, sized: bool = True) -> np.ndarray:
        return np.mean([aligned(s, v.series(sims[(s, delay)], c + extra_bps, sized=sized)) for s, c in COST.items()], axis=0)

    cases = {"measured_costs": rule(0), "costs_plus_0.2_bps": rule(0, 0.2), "one_minute_late": rule(1), "late_and_plus_0.2": rule(1, 0.2), "plain_1x": rule(0, sized=False)}
    for name, r in cases.items():
        combo = hold + SIZE * r
        res[name] = dict(rule=v.perf(r), rule_after=v.perf(r[after]), combo=v.perf(combo), combo_after=v.perf(combo[after]), corr=float(np.corrcoef(r, hold)[0, 1]))
    base = hold + SIZE * cases["measured_costs"]
    res["by_year"] = {int(y): dict(hold=round(float((1 + hold[years == y]).prod() - 1) * 100, 1), rule=round(float((1 + cases["measured_costs"][years == y]).prod() - 1) * 100, 1),
                                   combo=round(float((1 + base[years == y]).prod() - 1) * 100, 1)) for y in sorted(set(years))}
    res["avg_rule_leverage"] = float(np.mean([r["lev"] for s in COST for r in sims[(s, 0)]["rows"]]))
    res["peak_gross"] = float(1 + SIZE * max(r["lev"] for s in COST for r in sims[(s, 0)]["rows"]))
    res["target"] = dict(combo_mean_daily=float(base.mean()), days_at_or_above_1pct=float((base >= 0.01).mean()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    f = lambda d: (round(d["ann_return"] * 100, 1), round(d["sharpe"], 2), round(d["max_drawdown"] * 100, 1))
    print("hold", f(res["hold"]), "after", f(res["hold_after"]))
    for k in cases:
        x = res[k]
        print(k, "rule", f(x["rule"]), "after", f(x["rule_after"]), "combo", f(x["combo"]), "after", f(x["combo_after"]), "corr", round(x["corr"], 2))
    print(res["by_year"])
    print(res["avg_rule_leverage"], res["peak_gross"], res["target"])


if __name__ == "__main__":
    main()
