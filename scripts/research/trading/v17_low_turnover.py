"""Residual momentum and the 52-week high with a buy/hold band (see configs/low_turnover_v17.yaml, which holds the
pre-registered tests), on the unseen days. Writes artifacts/reports/swing/v17_low_turnover.json.

    .venv/Scripts/python.exe scripts/research/trading/v17_low_turnover.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from atlas.config import load_config
from atlas.swing.backtest import simulate, summary
from atlas.swing.data import load_panel
from atlas.swing.families import HOLD, _hold_between_rebalances, _rank_pick, exposures, prev_close
from atlas.swing.walkforward import SwingConfig, folds

cfg = yaml.safe_load(Path("configs/low_turnover_v17.yaml").read_text(encoding="utf-8"))
swing = load_config(Path(cfg["swing_config"]), SwingConfig)
OUT = Path("artifacts/reports/swing/v17_low_turnover.json")
full = load_panel(swing.dataset_dir, swing.symbols + [swing.cash_symbol]).before(swing.holdout_start)
p = full.select(swing.symbols)
c = full.select([swing.cash_symbol]).close[:, 0]
cash = np.zeros_like(c)
cash[1:] = c[1:] / c[:-1] - 1
plan = folds(len(p.dates), swing)
v0, end = plan[0][2], plan[-1][3]
pc = prev_close(p)
M = cfg["every"]


def residual_score() -> np.ndarray:
    n_days, n = pc.shape
    fit = cfg["fit_months"]
    score = np.full(pc.shape, np.nan)
    for t in range(fit * M, n_days):
        px = pc[t - np.arange(fit + 1) * M]  # newest first
        with np.errstate(invalid="ignore", divide="ignore"):
            r = px[:-1] / px[1:] - 1
        m = np.nanmean(r, axis=1)
        ok = ~np.isnan(r).any(axis=0)
        if ok.sum() < 2:
            continue
        x = np.c_[np.ones(fit), m]
        beta = np.linalg.lstsq(x, r[:, ok], rcond=None)[0]
        e = r[:, ok] - x @ beta
        w = e[1:12]  # months 2 to 12 back
        with np.errstate(invalid="ignore", divide="ignore"):
            score[t, ok] = w.sum(axis=0) / w.std(axis=0, ddof=1)
    return score


def high_score() -> np.ndarray:
    n = cfg["high_days"]
    out = np.full(pc.shape, np.nan)
    for t in range(n, len(pc)):
        w = pc[t - n + 1: t + 1]
        ok = (~np.isnan(w)).sum(axis=0) >= n // 2
        with np.errstate(invalid="ignore", divide="ignore"):
            out[t, ok] = pc[t, ok] / np.nanmax(w[:, ok], axis=0)
    return out


def banded(score: np.ndarray, every: int) -> np.ndarray:
    """Top fifth kept until it leaves the top 30%: a buy/hold band that cuts trading."""
    n = score.shape[1]
    held: list[int] = []

    def target(t: int) -> np.ndarray:
        nonlocal held
        sc = score[t]
        ok = np.nonzero(~np.isnan(sc))[0]
        w = np.zeros(n)
        if len(ok) < 10:
            held = []
            return w
        order = ok[np.argsort(-sc[ok], kind="stable")]
        want = max(1, int(len(ok) * cfg["top_share"]))
        band = set(order[: int(len(ok) * cfg["keep_share"])].tolist())
        held = [j for j in held if j in band]
        for j in order:
            if len(held) >= want:
                break
            if int(j) not in held:
                held.append(int(j))
        w[held] = n / len(held)
        return w

    return _hold_between_rebalances(target, len(score), n, every)


def returns(x: np.ndarray, cost: float = cfg["cost_bps"]) -> np.ndarray:
    return simulate(p, (x, x), v0, end, cost, cash).returns


def stats(r: np.ndarray) -> dict:
    s = summary(r, cash[v0:end])
    return {k: round(v, 4) for k, v in s.items()}


def yearly_edge(r: np.ndarray, h: np.ndarray, years: np.ndarray) -> dict:
    return {int(y): dict(rule=round(float((1 + r[years == y]).prod() - 1), 4), hold=round(float((1 + h[years == y]).prod() - 1), 4)) for y in sorted(set(years))}


def main() -> None:
    hold_x = exposures(p, *HOLD)
    h = simulate(p, hold_x, v0, end, cfg["cost_bps"], cash).returns
    years = np.array([d.year for d in p.dates[v0:end]])
    half = len(h) // 2
    res: dict = dict(config="configs/low_turnover_v17.yaml", window=f"{p.dates[v0]}..{p.dates[end - 1]}", hold=stats(h))
    tests = {}
    for name, sc in (("residual", residual_score()), ("high", high_score())):
        top_x, bot_x = banded(sc, M), banded(np.where(np.isnan(sc), np.nan, -sc), M)
        r, b = returns(top_x), returns(bot_x)
        yearly = yearly_edge(r, h, years)
        best = sorted(yearly, key=lambda y: -(yearly[y]["rule"] - yearly[y]["hold"]))[:2]
        keep = ~np.isin(years, best)
        s, sh = stats(r), stats(h)
        beat = bool(s["sharpe"] > sh["sharpe"] and s["ann_return"] >= sh["ann_return"])
        ex = summary(r[keep], cash[v0:end][keep])
        ex_h = summary(h[keep], cash[v0:end][keep])
        a, ah = summary(r[:half], cash[v0:end][:half]), summary(h[:half], cash[v0:end][:half])
        z, zh = summary(r[half:], cash[v0:end][half:]), summary(h[half:], cash[v0:end][half:])
        d = r - h
        edge, te = float(d.mean() * 252), float(d.std() * 252**0.5)
        turn = float(np.abs(np.diff(top_x[v0:end], axis=0)).sum(axis=1).mean() / top_x.shape[1] / 2 * M)
        k = int(sc[v0:end].shape[1])
        top10 = returns(_rank_pick(sc, 10, M, highest=True))
        res[name] = dict(top_fifth=s, bottom_fifth=stats(b), top_10=stats(top10), at_5bps=stats(returns(top_x, cfg["side_cost_bps"])),
                         monthly_turnover=round(turn, 3), by_year=yearly, best_years=best,
                         without_best=dict(rule_sharpe=round(ex["sharpe"], 3), hold_sharpe=round(ex_h["sharpe"], 3)),
                         halves=dict(first=[round(a["sharpe"], 3), round(ah["sharpe"], 3)], second=[round(z["sharpe"], 3), round(zh["sharpe"], 3)]),
                         edge_pct=round(edge * 100, 2), tracking_error_pct=round(te * 100, 2),
                         years_to_prove=round((2 * te / edge) ** 2, 1) if edge > 0 else None, stocks=k)
        tests[name] = dict(beats_hold=beat, control=bool(s["ann_return"] > stats(b)["ann_return"]),
                           not_luck=bool(ex["sharpe"] > ex_h["sharpe"]),
                           both_halves=bool(a["sharpe"] > ah["sharpe"] and z["sharpe"] > zh["sharpe"]))
    res["tests"] = tests
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
