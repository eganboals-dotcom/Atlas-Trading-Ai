"""v34: the ETF-only rotation (v32) on history, an estimate asked for on 2026-11-05, after the tracker was frozen.

Not pre-registered and not a test. v32's config says a backtest can't prove the rule, and these are the years its idea came from (v29, v33),
so the figure is flattered. It changes nothing about v32: the same frozen code picks every book, and its verdict still comes only from days
after 2026-11-02. Writes artifacts/reports/swing/v34_etf_rotation_history.json. Research only: no order is placed anywhere.

    .venv/Scripts/python.exe scripts/research/investing/v34_etf_rotation_history.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "trackers"))
import v32_etf_rotation_track as t  # noqa: E402
from atlas.swing.backtest import summary  # noqa: E402

START, END = date(2019, 1, 24), date(2026, 1, 29)  # the window the other investing backtests used
OUT = Path("artifacts/reports/swing/v34_etf_rotation_history.json")
NAMES = dict(rotation="ETF rotation", SPY="SPY", plain="All its ETFs held equally", QQQ="QQQ")


def main() -> dict:
    cfg = yaml.safe_load(t.CONFIG.read_text(encoding="utf-8"))
    m, cash = t.market(cfg, END)
    p = m.p
    first = max(k for k, e in enumerate(m.ends) if p.dates[e] < START)  # the last week-end before the window; its book trades first
    rule = {**cfg["rule"], "first_signal": str(p.dates[m.ends[first]]), "first_day": str(p.dates[m.ends[first] + 1])}
    groups = t.groups_of(p, rule)
    books = [dict(signal=str(p.dates[m.ends[k]]), weights={p.symbols[j]: x for j, x in t.pick(m, k, groups)[0].items()}) for k in range(first, len(m.ends))]
    r = t.score({**cfg, "rule": rule}, m, cash, books)
    a = p.dates.index(date.fromisoformat(rule["first_day"]))
    days = [d for d in p.dates[a:] if d <= END]
    r = {k: x[: len(days)] for k, x in r.items()}
    rows = {NAMES[k]: summary(r[k], r["rf"]) for k in NAMES}
    years = {y: {NAMES[k]: float(np.prod(1 + r[k][[d.year == y for d in days]]) - 1) for k in ("rotation", "SPY")} for y in sorted({d.year for d in days})}
    out = dict(window=[str(days[0]), str(days[-1])], rows=rows, years=years)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    res = main()
    print(f"{res['window'][0]} to {res['window'][1]}, after 2 bps each way, cash earning BIL\n")
    print(f"{'':28}{'a year':>9}{'Sharpe':>8}{'worst fall':>12}")
    for name, s in res["rows"].items():
        print(f"{name:28}{s['ann_return']:>+9.1%}{s['sharpe']:>8.2f}{s['max_drawdown']:>12.1%}")
    print("\n" + "  ".join(f"{y}: {v['ETF rotation']:+.1%} vs {v['SPY']:+.1%}" for y, v in res["years"].items()))
