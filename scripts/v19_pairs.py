"""Pairs trading on the 300 companies and the 16 swing ETFs (see configs/pairs_v19.yaml, which holds the pre-registered
tests), on the unseen days. Writes artifacts/reports/swing/v19_pairs.json.

    .venv/Scripts/python.exe scripts/v19_pairs.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import yaml

from atlas.config import load_config
from atlas.swing.backtest import simulate, summary
from atlas.swing.data import Panel, load_panel
from atlas.swing.families import HOLD, exposures
from atlas.swing.walkforward import SwingConfig

cfg = yaml.safe_load(Path("configs/pairs_v19.yaml").read_text(encoding="utf-8"))
OUT = Path("artifacts/reports/swing/v19_pairs.json")
FORM, TRADE, EVERY = cfg["formation_days"], cfg["trading_days"], cfg["book_every"]
BOOKS = TRADE // EVERY
FIRST, LAST = date.fromisoformat(cfg["first_scored"]), date.fromisoformat(cfg["last_scored"])


@dataclass(frozen=True)
class Market:
    p: Panel
    cash: np.ndarray  # daily return of the cash ETF
    v0: int  # first scored day
    end: int  # one past the last scored day
    on: np.ndarray  # close to open
    intra: np.ndarray  # open to close


def market(config: str) -> Market:
    s = load_config(Path(config), SwingConfig)
    p = load_panel(s.dataset_dir, s.symbols).before(s.holdout_start)
    bil = load_panel(Path(cfg["cash_dir"]), [cfg["cash_symbol"]])  # the ETF data set has no cash ETF of its own
    c = dict(zip(bil.dates, bil.close[:, 0]))
    cl = np.array([c[d] for d in p.dates])
    cash = np.zeros_like(cl)
    cash[1:] = cl[1:] / cl[:-1] - 1
    ok = [i for i, d in enumerate(p.dates) if FIRST <= d <= LAST]
    return Market(p, cash, ok[0], ok[-1] + 1, p.overnight, p.intraday)


def formation(close: np.ndarray, t0: int) -> tuple[np.ndarray, np.ndarray]:
    """Prices of the formation days scaled to start at 1, and the scale. t0 is the first trading day."""
    base = close[t0 - FORM]
    return close[t0 - FORM: t0] / base, base


def ranked_pairs(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All pairs, smallest distance first."""
    a, b = np.triu_indices(f.shape[1], 1)
    gram = f.T @ f
    d2 = np.diag(gram)[a] + np.diag(gram)[b] - 2 * gram[a, b]
    order = np.argsort(d2, kind="stable")
    return a[order], b[order]


def eg_tstat(x: np.ndarray, y: np.ndarray) -> float:
    """Engle-Granger: x on y, then a one-lag Dickey-Fuller regression on the residual, no intercept."""
    z = np.c_[np.ones(len(y)), y]
    e = x - z @ np.linalg.lstsq(z, x, rcond=None)[0]
    lag, d = e[:-1], np.diff(e)
    rho = float(lag @ d / (lag @ lag))
    res = d - rho * lag
    se = (float(res @ res) / (len(d) - 1) / float(lag @ lag)) ** 0.5
    return rho / se


def trade_book(m: Market, t0: int, a: np.ndarray, b: np.ndarray, cost_bps: float, wait: int = 0) -> tuple[np.ndarray, dict]:
    """Daily profit per pair, shape (trading days, pairs), on $1 long and $1 short per pair. Signals use closes; fills
    come at the next open, or ``wait`` days later. Days past the panel's end are left out."""
    close = m.p.close
    stop = min(t0 + TRADE, len(close))
    f, base = formation(close, t0)
    gap_f = f[:, a] - f[:, b]
    sd = gap_f.std(axis=0, ddof=1)
    gap = close[t0 - 1: stop, a] / base[a] - close[t0 - 1: stop, b] / base[b]  # gap[k] is the close before trading day k
    days, n = stop - t0, len(a)
    pnl = np.zeros((days, n))
    pos = np.zeros(n)
    va, vb = np.zeros(n), np.zeros(n)  # signed dollars held in a and in b
    c, borrow = cost_bps / 1e4, cfg["borrow_pct"] / 100 / 252
    want_hist = np.zeros((days, n))
    want = np.zeros(n)
    opened, converged = 0, 0
    for k in range(days):
        s = gap[k]
        was = want.copy()
        want = np.where(was == 0, np.where(s > cfg["open_sd"] * sd, -1.0, np.where(s < -cfg["open_sd"] * sd, 1.0, 0.0)),
                        np.where(was * s > 0, 0.0, was))  # a position ends when the gap changes sign
        want_hist[k] = want
        target = want_hist[k - wait] if k >= wait else np.zeros(n)
        t = t0 + k
        on_a, on_b, in_a, in_b = m.on[t, a], m.on[t, b], m.intra[t, a], m.intra[t, b]
        night = va * on_a + vb * on_b
        va, vb = va * (1 + on_a), vb * (1 + on_b)
        day_pnl = night - borrow * (np.where(va < 0, -va, 0.0) + np.where(vb < 0, -vb, 0.0))
        move = target != pos
        fee = c * (np.abs(va) + np.abs(vb)) * move  # out of the old position
        new = move & (target != 0)
        fee = fee + 2 * c * new
        opened += int(new.sum())
        converged += int((move & (pos != 0) & (target == 0)).sum())
        va, vb = np.where(move, target, va), np.where(move, -target, vb)
        pos = target.copy()
        pnl[k] = day_pnl - fee + va * in_a + vb * in_b
        va, vb = va * (1 + in_a), vb * (1 + in_b)
    pnl[-1] -= c * (np.abs(va) + np.abs(vb))  # closed at the last close
    return pnl, dict(opened=opened, converged=converged)


def book_starts(m: Market) -> list[int]:
    """Every book that trades on a scored day, with six running from the first scored day."""
    first = m.v0 - (BOOKS - 1) * EVERY
    return [t for t in range(first, m.end, EVERY) if t - FORM >= 0]


def run_rule(m: Market, picks: dict[int, tuple[np.ndarray, np.ndarray]], slots: int, cost_bps: float, wait: int = 0) -> tuple[np.ndarray, dict]:
    """Daily pair profit on total capital, for the scored days."""
    total = np.zeros(len(m.p.dates))
    seen = dict(opened=0, converged=0)
    for t0, (a, b) in picks.items():
        if not len(a):
            continue
        pnl, st = trade_book(m, t0, a, b, cost_bps, wait)
        total[t0: t0 + len(pnl)] += pnl.sum(axis=1) / slots / BOOKS
        seen = {k: seen[k] + st[k] for k in seen}
    return total[m.v0: m.end], seen


def random_control(m: Market, slots: int, cost_bps: float) -> np.ndarray:
    """Sharpe above cash of random_books copies of the engine on random pairs. Books run in one batch for speed."""
    rng = np.random.default_rng(cfg["seed"])
    draws = cfg["random_books"]
    n = m.p.close.shape[1]
    total = np.zeros((len(m.p.dates), draws))
    for t0 in book_starts(m):
        a = rng.integers(0, n, draws * slots)
        b = (a + rng.integers(1, n, draws * slots)) % n  # never the same stock twice in a pair
        pnl, _ = trade_book(m, t0, a, b, cost_bps)
        total[t0: t0 + len(pnl)] += pnl.reshape(len(pnl), draws, slots).sum(axis=2) / slots / BOOKS
    r = total[m.v0: m.end]
    return r.mean(axis=0) / r.std(axis=0) * 252**0.5


def picks_top(m: Market, k: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for t0 in book_starts(m):
        a, b = ranked_pairs(formation(m.p.close, t0)[0])
        out[t0] = (a[:k], b[:k])
    return out


def picks_coint(m: Market, k: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for t0 in book_starts(m):
        f = formation(m.p.close, t0)[0]
        a, b = ranked_pairs(f)
        keep = []
        for i in range(min(cfg["eg_candidates"], len(a))):
            if eg_tstat(f[:, a[i]], f[:, b[i]]) < cfg["eg_t"]:
                keep.append(i)
                if len(keep) == k:
                    break
        out[t0] = (a[keep], b[keep])
    return out


def stats(r: np.ndarray, rf: np.ndarray) -> dict:
    return {k: round(v, 4) for k, v in summary(r, rf).items()}


def by_year(r: np.ndarray, years: np.ndarray) -> dict:
    return {int(y): round(float((1 + r[years == y]).prod() - 1) * 100, 1) for y in sorted(set(years))}


def interval(r: np.ndarray) -> list[float]:
    """95% moving-block bootstrap interval of the mean."""
    rng = np.random.default_rng(cfg["seed"])
    block, n = cfg["bootstrap_block"], len(r)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, (cfg["bootstrap_samples"], k))
    idx = (starts[:, :, None] + np.arange(block)).reshape(len(starts), -1)[:, :n]
    return [float(x) for x in np.percentile(r[idx].mean(axis=1), [2.5, 97.5])]


def verdict(m: Market, pnl: np.ndarray, seen: dict, flips: np.ndarray, side: dict) -> dict:
    p, v0, end = m.p, m.v0, m.end
    rf = m.cash[v0:end]
    hold = simulate(p, exposures(p, *HOLD), v0, end, cfg["cost_bps"], m.cash).returns
    book = pnl + rf
    years = np.array([d.year for d in p.dates[v0:end]])
    ci = interval(pnl)
    combo = hold + cfg["half_size"] * pnl
    s, sc, sh = stats(book, rf), stats(combo, rf), stats(hold, rf)
    yearly = by_year(pnl, years)
    best = sorted(yearly, key=lambda y: -yearly[y])[:2]
    keep = ~np.isin(years, best)
    half = len(pnl) // 2
    sharpe = float(pnl.mean() / pnl.std() * 252**0.5) if pnl.std() else 0.0
    tests = dict(test1=bool(pnl.mean() > 0 and ci[0] > 0), test2=bool(sc["sharpe"] > sh["sharpe"] and sc["ann_return"] >= sh["ann_return"]),
                 test3=bool(pnl[keep].mean() > 0), test4=bool(pnl[:half].mean() > 0 and pnl[half:].mean() > 0),
                 control=bool((sharpe > flips).mean() >= 0.95))
    return dict(tests=tests, passed=all(tests.values()), book=s, excess_mean_daily_ci=ci, pair_profit_ann_pct=round(float(pnl.mean() * 252) * 100, 2),
                excess_sharpe=round(sharpe, 3), beat_random=round(float((sharpe > flips).mean()), 3), random_sharpe_median=round(float(np.median(flips)), 3),
                random_sharpe_95=round(float(np.percentile(flips, 95)), 3), hold=sh, hold_plus_half=sc, corr_with_hold=round(float(np.corrcoef(pnl, hold)[0, 1]), 3),
                by_year_pair_profit=yearly, best_years=best, trades_opened=seen["opened"], converged_share=round(seen["converged"] / max(1, seen["opened"]), 3), side=side)


def rule(m: Market, picks: dict, slots: int, flips: np.ndarray) -> dict:
    pnl, seen = run_rule(m, picks, slots, cfg["cost_bps"])
    rf = m.cash[m.v0: m.end]
    side = dict(at_side_cost=round(float(run_rule(m, picks, slots, cfg["side_cost_bps"])[0].mean() * 252) * 100, 2),
                wait_one_day=round(float(run_rule(m, picks, slots, cfg["cost_bps"], wait=1)[0].mean() * 252) * 100, 2),
                no_costs=round(float(run_rule(m, picks, slots, 0.0)[0].mean() * 252) * 100, 2), max_drawdown=stats(pnl, rf)["max_drawdown"])
    return verdict(m, pnl, seen, flips, side)


def main() -> None:
    stocks, etfs = market(cfg["swing_config"]), market(cfg["etf_config"])
    k, e = cfg["pairs_top"], cfg["etf_pairs_top"]
    flips = random_control(stocks, k, cfg["cost_bps"])
    res = dict(config="configs/pairs_v19.yaml", window=f"{stocks.p.dates[stocks.v0]}..{stocks.p.dates[stocks.end - 1]}",
               rule_a=rule(stocks, picks_top(stocks, k), k, flips), rule_b=rule(stocks, picks_coint(stocks, k), k, flips),
               rule_c=rule(etfs, picks_top(etfs, e), e, random_control(etfs, e, cfg["cost_bps"])))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
