"""Attempts to break the earnings-quality result of v23. NOT pre-registered: it passes or fails nothing, and v23's verdict stands.

Asks who the rule holds, whether a few names or rebounds from write-offs drive it, whether it beats random picks from the same
companies, and how it did on 2017-2018, before the window it was tested on. Days from 2026-04-28 on are never read.
Writes artifacts/reports/swing/v23_followup.json.

    .venv/Scripts/python.exe scripts/research/investing/v23_followup.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "trading"))
import v19_pairs as pairs
import v22_bottleneck as bottleneck
import v23_filings as f
from atlas.config import load_config
from atlas.swing.backtest import run, simulate
from atlas.swing.data import load_panel
from atlas.swing.families import HOLD, exposures
from atlas.swing.fundamentals import SCHEMA, fundamental_features
from atlas.swing.walkforward import SwingConfig

OUT = Path("artifacts/reports/swing/v23_followup.json")
cfg = f.cfg
EVERY = cfg["every"]
EARLY = date(2017, 1, 24)
SEED, RANDOM, BLOCK = 2026, 1000, 21


def stats(r: np.ndarray, rf: np.ndarray) -> dict:
    s = f.stats(r, rf)
    return dict(ann_return=s["ann_return"], sharpe=s["sharpe"], max_drawdown=s["max_drawdown"])


def main() -> None:
    swing = load_config(Path(cfg["swing_config"]), SwingConfig)
    m = pairs.market(cfg["swing_config"])
    p, v0, end = m.p, m.v0, m.end
    n = p.close.shape[1]
    raw = load_panel(swing.raw_dataset_dir, swing.symbols)
    raw_at = {d: i for i, d in enumerate(raw.dates)}
    feats = fundamental_features(p, Path(cfg["fundamentals_dir"]), raw.close[[raw_at[d] for d in p.dates]])
    e = load_panel(Path(bottleneck.cfg["etf_dir"]), bottleneck.cfg["etfs"] + [bottleneck.cfg["market"]])
    at = {d: i for i, d in enumerate(e.dates)}
    idx = [at[d] for d in p.dates]
    etf, mkt = e.close[idx, :-1], e.close[idx, -1]
    rows = {}
    for s in p.symbols:
        files = [Path(cfg[k]) / f"{s}.parquet" for k in ("fundamentals_dir", "extra_dir")]
        rows[s] = pl.concat([pl.read_parquet(x) for x in files if x.exists()]) if any(x.exists() for x in files) else pl.DataFrame(schema=SCHEMA)

    def score_at(t: int) -> dict[str, np.ndarray]:
        d = p.dates[t]
        out = {k: np.full(n, np.nan) for k in ("quality", "income", "cash")}
        for j, s in enumerate(p.symbols):
            if not np.isfinite(feats["fund_gpa"][t, j]):
                continue
            q = f.filing_scores(rows[s], d, True).get("earnings_quality")
            assets, income, cash = (f.latest_two(rows[s], k, d) for k in ("assets", "net_income", "cfo"))
            if q is None or not (assets and income and cash):
                continue
            out["quality"][j] = q
            out["income"][j], out["cash"][j] = income[1] / assets[1], cash[1] / assets[1]
        return out

    def book(start: int, stop: int, pick) -> tuple[np.ndarray, list]:
        """Top-fifth-style book: pick(t, scores) gives the names held from t."""
        x, held = np.zeros((len(p.dates), n)), []
        for t in range(start, stop, EVERY):
            names = pick(t, score_at(t))
            x[t: t + EVERY, names] = n / len(names)
            held.append((t, names))
        return x, held

    top = lambda t, s: f.top(s["quality"], cfg["top_share"])
    ret = lambda x, a=v0, b=end: run(p, x, x, a, b, cfg["cost_bps"], None, m.cash).returns
    rf = m.cash[v0:end]
    h = simulate(p, exposures(p, *HOLD), v0, end, cfg["cost_bps"], m.cash).returns
    x_top, held = book(v0, end, top)
    r = ret(x_top)
    res: dict = dict(note="not pre-registered", hold=stats(h, rf), rule=stats(r, rf))

    x_all, _ = book(v0, end, lambda t, s: np.nonzero(np.isfinite(s["quality"]))[0].tolist())
    res["all_eligible"] = stats(ret(x_all), rf)

    # Names the rule leans on: each stock's share of the book's return, with the five largest removed.
    daily = np.vstack([np.zeros((1, n)), p.close[1:] / p.close[:-1] - 1])
    gain = Counter()
    for t, names in held:
        for j in names:
            gain[j] += float(np.prod(1 + daily[t: min(t + EVERY, end), j]) - 1) / len(names)
    best = [j for j, _ in gain.most_common(5)]
    res["top_contributors"] = [(p.symbols[j], round(v * 100, 1)) for j, v in gain.most_common(8)]
    res["without_top_5_names"] = stats(ret(book(v0, end, lambda t, s: [j for j in top(t, s) if j not in best])[0]), rf)
    res["most_held"] = [(p.symbols[j], c) for j, c in Counter(j for _, names in held for j in names).most_common(12)]
    sec_names = bottleneck.cfg["etfs"]
    mix = Counter()
    for t, names in held:
        g = bottleneck.sectors(p.close, etf, mkt, t)
        mix.update(sec_names[int(g[j])] for j in names)
    total = sum(mix.values())
    res["sector_mix"] = {k: round(v / total, 3) for k, v in mix.most_common()}

    # Write-off rebounds: the same rule among profitable companies only, and how often it holds a loss-maker.
    res["profitable_only"] = stats(ret(book(v0, end, lambda t, s: f.top(np.where(s["income"] > 0, s["quality"], np.nan), cfg["top_share"]))[0]), rf)
    res["loss_maker_share"] = round(float(np.mean([np.mean(score_at(t)["income"][names] < 0) for t, names in held])), 3)
    res["top_cash_profit"] = stats(ret(book(v0, end, lambda t, s: f.top(s["cash"], cfg["top_share"]))[0]), rf)
    res["top_net_income"] = stats(ret(book(v0, end, lambda t, s: f.top(s["income"], cfg["top_share"]))[0]), rf)

    # Random picks of the same size from the same companies each quarter, on simple daily means, without costs.
    rng = np.random.default_rng(SEED)
    sims = np.zeros(RANDOM)
    actual = []
    length = 0
    for t, names in held:
        stop = min(t + EVERY, end)
        pool = np.nonzero(np.isfinite(score_at(t)["quality"]))[0]
        q = daily[t: stop][:, pool].mean(axis=0)  # each name's mean daily return this quarter
        k = len(names)
        draws = np.argsort(rng.random((RANDOM, len(pool))), axis=1)[:, :k]
        sims += q[draws].mean(axis=1) * (stop - t)
        actual.append(daily[t: stop][:, names].mean() * (stop - t))
        length += stop - t
    res["random_beaten"] = round(float((sum(actual) > sims).mean()), 4)
    res["random_ann_pct"] = dict(rule=round(sum(actual) / length * 252 * 100, 1), median=round(float(np.median(sims)) / length * 252 * 100, 1), p95=round(float(np.percentile(sims, 95)) / length * 252 * 100, 1))

    # Block bootstrap of the yearly gap to Hold.
    d = r - h
    k = int(np.ceil(len(d) / BLOCK))
    starts = rng.integers(0, len(d) - BLOCK + 1, (5000, k))
    ids = (starts[:, :, None] + np.arange(BLOCK)).reshape(len(starts), -1)[:, : len(d)]
    res["gap_to_hold_pct_95"] = [round(float(x) * 252 * 100, 1) for x in np.percentile(d[ids].mean(axis=1), [2.5, 97.5])]

    # Before the tested window: 2017-01-24 to the day before it. These days were outside every test of this rule.
    a = next(i for i, x in enumerate(p.dates) if x >= EARLY)
    xe, _ = book(a, v0, top)
    he = simulate(p, exposures(p, *HOLD), a, v0, cfg["cost_bps"], m.cash).returns
    xb, _ = book(a, v0, lambda t, s: f.top(s["quality"], cfg["top_share"], highest=False))
    rfe = m.cash[a:v0]
    res["before_window"] = dict(first=str(p.dates[a]), last=str(p.dates[v0 - 1]), rule=stats(ret(xe, a, v0), rfe), hold=stats(he, rfe), bottom_fifth=stats(ret(xb, a, v0), rfe))
    OUT.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
