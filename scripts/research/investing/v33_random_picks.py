"""The rotation against chance on v31's list (see configs/random_picks_v33.yaml, which holds the pre-registered tests). Writes
artifacts/reports/swing/v33_random_picks.json.

    .venv/Scripts/python.exe scripts/research/investing/v33_random_picks.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

import v23_filings as f
import v27_finrl_rotation as v
import v31_point_in_time as pit
from atlas.swing.backtest import run

cfg = yaml.safe_load(Path("configs/random_picks_v33.yaml").read_text(encoding="utf-8"))


def group_draw(k: v.Book, groups: dict[str, list[int]], rng: np.random.Generator) -> dict[int, float]:
    """A book with the same budget and weights, its groups drawn at random, and its assets drawn inside them. A fallback book stays."""
    if not k.slots:
        return dict(k.weights)
    names = sorted(g for g, js in groups.items() if len(js) >= max(len(w) for _, w in k.slots))
    weights: dict[int, float] = {}
    for (_, slot), c in zip(k.slots, rng.choice(len(names), len(k.slots), replace=False).tolist()):
        for j, x in zip(rng.choice(groups[names[c]], len(slot), replace=False).tolist(), slot):
            weights[j] = x
    return weights


def group_run(rot: v.Rotation, books: list[v.Book], lists: dict, live: np.ndarray, a: int, b: int, cash: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    dates = rot.p.dates
    drawn = [group_draw(k, pit.live_groups(lists, live, k.t, dates[k.t].year), rng) for k in books]
    x = v.exposure(drawn, [k.t for k in books], rot.p.close.shape)
    return run(rot.p, x, x, a, b, v.cfg["cost_bps"], None, cash).returns


def spread(s: list[dict], ann: float, spy: float) -> dict:
    a = np.array([x["ann_return"] for x in s])
    return dict(p5=round(float(np.percentile(a, 5)), 4), median=round(float(np.median(a)), 4), p95=round(float(np.percentile(a, 95)), 4), median_sharpe=round(float(np.median([x["sharpe"] for x in s])), 3),
                rule_beats_share=round(float((a < ann).mean()), 3), share_beating_spy=round(float((a > spy).mean()), 3))


def main() -> None:
    m = pit.market(True)
    rot, lists, live, books = pit.rotation_books(m)
    a, b = m.v0, m.end
    rf, spy_r = m.cash[a:b], m.spy[a:b]
    rule = f.stats(v.net(rot, books, a, b, pit.cfg["rotation_cost_bps"], m.cash), rf)
    seen = json.loads(Path(cfg["v31_report"]).read_text(encoding="utf-8"))["rotation"]["figures"]["rotation"]["ann_return"]
    if rule["ann_return"] != seen:
        raise SystemExit(f"refused: annual return {rule['ann_return']} isn't v31's {seen}")
    spy = f.stats(spy_r, rf)
    rng = np.random.default_rng(cfg["seed"])
    picks = [f.stats(v.random_run(rot, books, a, b, m.cash, rng), rf) for _ in range(cfg["draws"])]
    groups = [f.stats(group_run(rot, books, lists, live, a, b, m.cash, rng), rf) for _ in range(cfg["draws"])]
    p, g = spread(picks, rule["ann_return"], spy["ann_return"]), spread(groups, rule["ann_return"], spy["ann_return"])
    tests = dict(test1=bool(rule["ann_return"] > p["p95"]), test2=f.beats(dict(ann_return=p["median"], sharpe=p["median_sharpe"]), spy), test3=bool(rule["ann_return"] > g["p95"]))
    res = dict(config="configs/random_picks_v33.yaml", books=len(books), tests=tests, rule=rule, spy=spy, random_assets=p, random_groups_and_assets=g)
    out = Path(cfg["report"])
    out.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
