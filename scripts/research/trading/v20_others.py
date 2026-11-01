"""The pre-FOMC drift, same-month seasonality and long-short trend following (see configs/others_v20.yaml, which holds
the pre-registered tests). Writes artifacts/reports/swing/v20_others.json.

    .venv/Scripts/python.exe scripts/research/trading/v20_others.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import yaml

import v15_noise_area as v
import v19_pairs as pairs
from atlas.swing.backtest import run, simulate, summary
from atlas.swing.families import HOLD, exposures, prev_close

cfg = yaml.safe_load(Path("configs/others_v20.yaml").read_text(encoding="utf-8"))
OUT = Path("artifacts/reports/swing/v20_others.json")
BORROW = cfg["borrow_pct"] / 100 / 252


def stats(r: np.ndarray, rf: np.ndarray) -> dict:
    return {k: round(x, 4) for k, x in summary(r, rf).items()}


def beats(a: dict, b: dict) -> bool:
    return bool(a["sharpe"] > b["sharpe"] and a["ann_return"] >= b["ann_return"])


def fomc_windows(days: list[dict]) -> tuple[np.ndarray, np.ndarray, list[date]]:
    """Returns of the 14:00-to-14:00 windows on decision days and on every other day, after costs."""
    decisions = {d for year in cfg["fomc"] for d in year}
    m, c = cfg["decision_time_minute"], cfg["cost_bps_spy"] / 1e4
    ev, other, when = [], [], []
    for a, b in zip(days[:-1], days[1:]):
        x, y = a["o"][m], b["o"][m]
        r = (y * (1 - c)) / (x * (1 + c)) - 1
        if b["day"] in decisions:
            ev.append(r)
            when.append(b["day"])
        elif a["day"] not in decisions:  # a window starting at a decision would hold its reaction
            other.append(r)
    return np.array(ev), np.array(other), when


def rule_d() -> dict:
    days = v.load_days("SPY")
    ev, other, when = fomc_windows(days)
    rng = np.random.default_rng(cfg["seed"])
    boot = ev[rng.integers(0, len(ev), (cfg["bootstrap_samples"], len(ev)))].mean(axis=1)
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    half = len(ev) // 2
    wanted = {d for year in cfg["fomc"] for d in year}
    trimmed = np.sort(ev)[:-3]
    dates = [d["day"] for d in days]
    hold = v.hold_returns("SPY", dates[1:])
    extra = np.zeros(len(hold))
    pos = {d: i for i, d in enumerate(dates[1:])}
    for d, r in zip(when, ev):
        extra[pos[d]] = r
    perf = lambda x: {k: round(float(z), 4) for k, z in v.perf(x).items()}
    years = np.array([d.year for d in when])
    tests = dict(d1=bool(ev.mean() > 0 and ci[0] > 0), d2=bool(ev.mean() > other.mean()), d3=bool(ev[:half].mean() > 0 and ev[half:].mean() > 0), d4=bool(trimmed.mean() > 0))
    return dict(tests=tests, passed=all(tests.values()), events=len(ev), events_listed=len(wanted), mean_bps=round(float(ev.mean()) * 1e4, 2), ci_bps=[round(x * 1e4, 2) for x in ci],
                other_mean_bps=round(float(other.mean()) * 1e4, 2), first_half_bps=round(float(ev[:half].mean()) * 1e4, 2), second_half_bps=round(float(ev[half:].mean()) * 1e4, 2),
                without_best3_bps=round(float(trimmed.mean()) * 1e4, 2), up_share=round(float((ev > 0).mean()), 3), hold=perf(hold), hold_plus=perf(hold + extra),
                by_year_bps={int(y): round(float(ev[years == y].mean()) * 1e4, 1) for y in sorted(set(years))})


def season_score(p, pc: np.ndarray, starts: list[int]) -> np.ndarray:
    """Score at each month's first day: the mean return of this calendar month in earlier years."""
    month_end_close = pc[starts]  # the close before each month's first open
    r = month_end_close[1:] / month_end_close[:-1] - 1  # r[i] is the month that began at starts[i]
    months = np.array([p.dates[t].month for t in starts[:-1]])
    score = np.full((len(starts), pc.shape[1]), np.nan)
    for i in range(len(starts)):
        same = np.nonzero(months[:i] == p.dates[starts[i]].month)[0]
        if len(same) >= cfg["season_min_years"]:
            score[i] = r[same].mean(axis=0)
    return score


def monthly_book(n_days: int, n: int, starts: list[int], picks: np.ndarray) -> np.ndarray:
    """Equal-weight slots in picks[i] from month i's first day to the next month's."""
    x = np.zeros((n_days, n))
    for i, t in enumerate(starts):
        idx = np.nonzero(picks[i])[0]
        if len(idx):
            stop = starts[i + 1] if i + 1 < len(starts) else n_days
            x[t: stop, idx] = n / len(idx)
    return x


def rule_e() -> dict:
    m = pairs.market(cfg["swing_config"])
    p, v0, end = m.p, m.v0, m.end
    pc = prev_close(p)
    rf = m.cash[v0:end]
    starts = [t for t in range(1, len(p.dates)) if p.dates[t].month != p.dates[t - 1].month]
    score = season_score(p, pc, starts)
    k = int(p.close.shape[1] * cfg["top_share"])
    top, bot = np.zeros(score.shape, bool), np.zeros(score.shape, bool)
    for i in range(len(starts)):
        if np.isfinite(score[i]).all():
            order = np.argsort(-score[i], kind="stable")
            top[i, order[:k]], bot[i, order[-k:]] = True, True
    n = p.close.shape[1]
    x_top, x_bot = monthly_book(len(pc), n, starts, top), monthly_book(len(pc), n, starts, bot)
    r = run(p, x_top, x_top, v0, end, cfg["cost_bps"], None, m.cash).returns
    b = run(p, x_bot, x_bot, v0, end, cfg["cost_bps"], None, m.cash).returns
    h = simulate(p, exposures(p, *HOLD), v0, end, cfg["cost_bps"], m.cash).returns
    years = np.array([d.year for d in p.dates[v0:end]])
    gaps = {int(y): round(float((1 + r[years == y]).prod() - (1 + h[years == y]).prod()) * 100, 1) for y in sorted(set(years))}
    best = sorted(gaps, key=lambda y: -gaps[y])[:2]
    keep = ~np.isin(years, best)
    half = len(r) // 2
    s, sh, sb = stats(r, rf), stats(h, rf), stats(b, rf)
    sharpe = lambda i: stats(r[i], rf[i])["sharpe"] > stats(h[i], rf[i])["sharpe"]
    turn = float(np.abs(np.diff(x_top[v0:end], axis=0)).sum(axis=1).mean() / n / 2 * 21)
    tests = dict(e1=beats(s, sh), e2=bool(s["ann_return"] > sb["ann_return"]), e3=bool(sharpe(keep)), e4=bool(sharpe(np.arange(len(r)) < half) and sharpe(np.arange(len(r)) >= half)))
    return dict(tests=tests, passed=all(tests.values()), top_fifth=s, bottom_fifth=sb, hold=sh, gap_by_year=gaps, best_years=best, monthly_turnover=round(turn, 3))


def rule_f() -> dict:
    m = pairs.market(cfg["etf_config"])
    p, v0, end = m.p, m.v0, m.end
    pc = prev_close(p)
    rf = m.cash[v0:end]
    n = pc.shape[1]
    daily = np.full(pc.shape, np.nan)
    daily[1:] = pc[1:] / pc[:-1] - 1
    x = np.zeros(pc.shape)
    stop = len(pc)
    for t in range(v0, stop, cfg["every"]):
        trend = pc[t] / pc[t - cfg["trend_days"]] - 1
        vol = np.nanstd(daily[t - cfg["trend_vol_days"] + 1: t + 1], axis=0, ddof=1) * 252**0.5
        x[t: t + cfg["every"]] = np.sign(trend) * cfg["trend_vol"] / vol  # slots: weight x n, and the rule's weight is this / n
    gross = run(p, x, x, v0, end, cfg["cost_bps"]).returns  # no cash: profit on capital
    short = np.where(x < 0, -x, 0.0)[v0:end].mean(axis=1)
    pnl = gross - short * BORROW
    book = pnl + rf
    h = simulate(p, exposures(p, *HOLD), v0, end, cfg["cost_bps"], m.cash).returns
    combo = h + cfg["half_size"] * pnl
    years = np.array([d.year for d in p.dates[v0:end]])
    yearly = {int(y): round(float((1 + pnl[years == y]).prod() - 1) * 100, 1) for y in sorted(set(years))}
    best = sorted(yearly, key=lambda y: -yearly[y])[:2]
    keep = ~np.isin(years, best)
    half = len(pnl) // 2
    s, sc, sh = stats(book, rf), stats(combo, rf), stats(h, rf)
    tests = dict(f1=beats(sc, sh), f2=bool(pnl.mean() > 0), f3=bool(pnl[keep].mean() > 0), f4=bool(pnl[:half].mean() > 0 and pnl[half:].mean() > 0))
    return dict(tests=tests, passed=all(tests.values()), book=s, hold=sh, hold_plus_half=sc, corr_with_hold=round(float(np.corrcoef(pnl, h)[0, 1]), 3),
                by_year=yearly, best_years=best, avg_gross=round(float(np.abs(x[v0:end]).mean()), 2), avg_net=round(float(x[v0:end].mean()), 2))


def main() -> None:
    res = dict(config="configs/others_v20.yaml", rule_d=rule_d(), rule_e=rule_e(), rule_f=rule_f())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
