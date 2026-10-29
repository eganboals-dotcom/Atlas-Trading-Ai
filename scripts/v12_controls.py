"""Controls and side numbers for the v12 tests (see configs/swing_v12_news_eps_u300.yaml), on the unseen days.

Writes artifacts/reports/swing/v12_controls_u300.json: the pre-registered control (top vs bottom fifth by headline
surprise), the speed comparison (filing-based surprise re-picked weekly), the combined tilt, costs from 0 to 5 bps,
and for each rule the yearly edge over Hold, its tracking error and the years of data a 2-standard-error result
would take. Needs the u300 data and headlines (see the config).

    .venv/Scripts/python.exe scripts/v12_controls.py
"""

from __future__ import annotations

import dataclasses
import json
from datetime import date
from pathlib import Path

import numpy as np

from atlas.config import load_config
from atlas.swing import shorts
from atlas.swing.backtest import simulate, summary
from atlas.swing.data import load_panel
from atlas.swing.families import HOLD, _rank_pick, exposures, fund_rank, prev_close, tilt, tilt_features
from atlas.swing.fundamentals import fundamental_features
from atlas.swing.news import load_news, surprise_features
from atlas.swing.walkforward import SwingConfig, folds

cfg = load_config(Path("configs/swing_v12_news_eps_u300.yaml"), SwingConfig)
full = load_panel(cfg.dataset_dir, cfg.symbols + [cfg.cash_symbol]).before(cfg.holdout_start)
p = full.select(cfg.symbols)
raw = load_panel(cfg.raw_dataset_dir, cfg.symbols)
idx = {d: i for i, d in enumerate(raw.dates)}
rc = np.array([raw.close[idx[d]] if d in idx else np.full(len(cfg.symbols), np.nan) for d in p.dates])
feats = fundamental_features(p, cfg.fundamentals_dir, rc)
feats |= shorts.short_features(p, shorts.load(cfg.short_interest_path))
feats |= surprise_features(p, load_news(cfg.news_dir), rc)
p = dataclasses.replace(p, features=feats)
p = dataclasses.replace(p, features={**feats, **tilt_features(p)})
c = full.select([cfg.cash_symbol]).close[:, 0]
cash = np.zeros_like(c)
cash[1:] = c[1:] / c[:-1] - 1
plan = folds(len(p.dates), cfg)
v0, end = plan[0][2], plan[-1][3]
s21 = next(i for i, d in enumerate(p.dates) if d >= date(2021, 7, 1))
listed = ~np.isnan(prev_close(p))
hold_x = exposures(p, *HOLD)


def returns(x, start=v0, cost=2.0):
    x = x if isinstance(x, tuple) else (x, x)
    return simulate(p, x, start, end, cost, cash).returns


def stats(x, start=v0, cost=2.0):
    s = summary(returns(x, start, cost), cash[start:end])
    return [round(s["ann_return"] * 100, 2), round(s["sharpe"], 3)]


def edge(x, start=v0):
    """Yearly return over Hold, tracking error (yearly std of the daily difference) and the years of data for the
    edge to be two standard errors from zero: (2 x TE / edge)^2."""
    d = returns(x, start) - returns(hold_x, start)
    e, te = float(d.mean() * 252), float(d.std() * np.sqrt(252))
    return {"edge_pct": round(e * 100, 2), "tracking_error_pct": round(te * 100, 2),
            "years_for_2se": round((2 * te / e) ** 2, 1) if e > 0 else None, "years_tested": round(len(d) / 252, 1)}


def fifths(score, every):
    k = int(np.median((~np.isnan(score[v0:end])).sum(axis=1)) // 5)
    top = _rank_pick(score, k, every, highest=True)
    bot = _rank_pick(np.where(np.isnan(score), np.nan, -score), k, every, highest=True)
    return top, bot, k


news = np.where(listed, feats["news_surprise"], np.nan)
top, bot, k = fifths(news, 5)
coverage = float(np.mean((~np.isnan(news[v0:end])).sum(axis=1) / listed[v0:end].sum(axis=1)))
t1 = fund_rank(p, {"factor": 6.0, "top_k": 10, "rebalance": 5})
t2 = tilt(p, {"factor": 7.0, "tilt": 0.5, "rebalance": 5})
combo = tilt(p, {"factor": 8.0, "tilt": 0.5, "rebalance": 21})
quality = fund_rank(p, {"factor": 0.0, "top_k": 10, "rebalance": 63})
sue = fund_rank(p, {"factor": 5.0, "top_k": 10, "rebalance": 21})
short_tilt = tilt(p, {"factor": 6.0, "tilt": 0.5, "rebalance": 21})
out = {
    "window": f"{p.dates[v0]}..{p.dates[end - 1]}", "hold": stats(hold_x),
    "headline_coverage": round(coverage, 3),
    "news_surprise": {"top 10, weekly (test 1)": stats(t1), "tilt 0.5, weekly (test 2)": stats(t2),
                      f"top fifth ({k}), weekly": stats(top), f"bottom fifth ({k}), weekly": stats(bot)},
    "speed": {"filing surprise (SUE) top 10, weekly": stats(fund_rank(p, {"factor": 5.0, "top_k": 10, "rebalance": 5})),
              "filing surprise (SUE) top 10, monthly (v11 test)": stats(sue),
              "headline surprise top 10, monthly": stats(fund_rank(p, {"factor": 6.0, "top_k": 10, "rebalance": 21}))},
    "combined": {"tilt 0.5 monthly": stats(combo), "from 2021-07": stats(combo, s21),
                 "hold from 2021-07": stats(hold_x, s21)},
    "costs": {f"{b} bps": {"test 1": stats(t1, cost=b), "test 2": stats(t2, cost=b), "hold": stats(hold_x, cost=b)}
              for b in (0.0, 1.0, 2.0, 5.0)},
    "edges": {"quality top 10, quarterly": edge(quality), "filing surprise top 10, monthly": edge(sue),
              "low short interest tilt, from 2021-07": edge(short_tilt, s21),
              "headline surprise top 10, weekly": edge(t1), "headline surprise tilt, weekly": edge(t2),
              "combined tilt, monthly": edge(combo)},
}
Path("artifacts/reports/swing/v12_controls_u300.json").write_text(json.dumps(out, indent=1))
for key, v in out.items():
    print(key, v)
