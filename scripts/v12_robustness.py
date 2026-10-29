"""Robustness of v12 test 1 (headline earnings surprise, top 10, weekly) on the unseen days: is it a few stocks,
a few years, or just "companies that had earnings news"? Not part of the pre-registered test; written after its
result was seen, to check it. Writes artifacts/reports/swing/v12_robustness_u300.json.

    .venv/Scripts/python.exe scripts/v12_robustness.py
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from atlas.config import load_config
from atlas.swing.backtest import simulate, summary
from atlas.swing.data import load_panel
from atlas.swing.families import HOLD, _rank_pick, exposures, fund_rank, prev_close
from atlas.swing.news import load_news, surprise_features
from atlas.swing.walkforward import SwingConfig, folds

cfg = load_config(Path("configs/swing_v12_news_eps_u300.yaml"), SwingConfig)
full = load_panel(cfg.dataset_dir, cfg.symbols + [cfg.cash_symbol]).before(cfg.holdout_start)
p = full.select(cfg.symbols)
raw = load_panel(cfg.raw_dataset_dir, cfg.symbols)
idx = {d: i for i, d in enumerate(raw.dates)}
rc = np.array([raw.close[idx[d]] if d in idx else np.full(len(cfg.symbols), np.nan) for d in p.dates])
feats = surprise_features(p, load_news(cfg.news_dir), rc)
p = dataclasses.replace(p, features=feats)
c = full.select([cfg.cash_symbol]).close[:, 0]
cash = np.zeros_like(c)
cash[1:] = c[1:] / c[:-1] - 1
plan = folds(len(p.dates), cfg)
v0, end = plan[0][2], plan[-1][3]
listed = ~np.isnan(prev_close(p))
G = {"factor": 6.0, "top_k": 10, "rebalance": 5}


def run(x, q=p):
    x = x if isinstance(x, tuple) else (x, x)
    return simulate(q, x, v0, end, 2.0, cash)


def stats(r):
    s = summary(r.returns, cash[v0:end])
    return {"ann_return": round(s["ann_return"] * 100, 2), "sharpe": round(s["sharpe"], 3),
            "max_drawdown": round(s["max_drawdown"] * 100, 1)}


test = fund_rank(p, G)
hold_x = exposures(p, *HOLD)
r_test, r_hold = run(test), run(hold_x)
out = {"window": f"{p.dates[v0]}..{p.dates[end - 1]}", "test_1": stats(r_test), "hold": stats(r_hold)}

# 1. Year by year
years = np.array([d.year for d in p.dates[v0:end]])
out["by_year"] = {}
for y in sorted(set(years)):
    m = years == y
    a, h = np.prod(1 + r_test.returns[m]) - 1, np.prod(1 + r_hold.returns[m]) - 1
    out["by_year"][int(y)] = {"test_1": round(a * 100, 1), "hold": round(h * 100, 1), "diff": round((a - h) * 100, 1)}

# 2. Which stocks drove it: dollar-weighted contribution of each symbol (weight x next-day return, summed)
n = len(p.symbols)
w = test[v0:end] / n
day_ret = np.nan_to_num(p.close[v0:end] / prev_close(p)[v0:end] - 1)
contrib = (w * day_ret).sum(axis=0)
top = np.argsort(-contrib)[:5]
out["top_contributors"] = {p.symbols[j]: round(float(contrib[j]) * 100, 1) for j in top}
out["days_held_share"] = {p.symbols[j]: round(float((w[:, j] > 0).mean()) * 100, 1) for j in top}

# 3. Without the biggest contributors: drop them from the panel entirely (they can't be picked)
for k in (1, 3, 5):
    drop = {p.symbols[j] for j in top[:k]}
    keep = [s for s in p.symbols if s not in drop]
    q = p.select(keep)
    q = dataclasses.replace(q, features={"news_surprise": feats["news_surprise"][:, [p.symbols.index(s) for s in keep]]})
    out[f"without_top_{k}"] = {"dropped": sorted(drop), "test_1": stats(run(fund_rank(q, G), q)),
                               "hold": stats(run(exposures(q, *HOLD), q))}

# 4. Random 10 from the same pool (stocks with a fresh headline surprise), re-picked weekly: is it the surprise
#    size or just "had earnings news"?
pool = ~np.isnan(np.where(listed, feats["news_surprise"], np.nan))
rng = np.random.default_rng(12)
draws = []
for _ in range(200):
    score = np.where(pool, rng.random(pool.shape), np.nan)
    draws.append(summary(run(_rank_pick(score, 10, 5, highest=True)).returns, cash[v0:end])["ann_return"] * 100)
draws = np.array(draws)
out["random_10_from_pool"] = {"median": round(float(np.median(draws)), 2), "p95": round(float(np.percentile(draws, 95)), 2),
                              "test_1_beats_share": round(float((draws < out["test_1"]["ann_return"]).mean()), 3)}
# and the smallest surprises, as a mirror image
out["bottom_10_weekly"] = stats(run(_rank_pick(np.where(pool, feats["news_surprise"], np.nan), 10, 5, highest=False)))

Path("artifacts/reports/swing/v12_robustness_u300.json").write_text(json.dumps(out, indent=1))
for key, v in out.items():
    print(key, v)
