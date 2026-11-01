"""A copy of FinRL-Trading's Adaptive Rotation (see configs/finrl_rotation_v27.yaml, which holds the rule and the pre-registered tests).
fetch stores daily bars (read-only). fidelity compares the copy with FinRL's published figures under FinRL's own accounting, which is not a
test. run writes artifacts/reports/swing/v27_finrl_rotation.json.

    .venv/Scripts/python.exe scripts/research/investing/v27_finrl_rotation.py fetch
    .venv/Scripts/python.exe scripts/research/investing/v27_finrl_rotation.py fidelity
    .venv/Scripts/python.exe scripts/research/investing/v27_finrl_rotation.py run
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import yaml

import v22_bottleneck as bottleneck
import v23_filings as f
from atlas.config import load_config
from atlas.feeds.alpaca import HISTORY_BARS_PATH, HISTORY_PATHS, http_fetch, iter_history, load_credentials
from atlas.swing.backtest import run, simulate
from atlas.swing.data import Panel, daily_rows, load_panel
from atlas.swing.families import HOLD, exposures
from atlas.swing.walkforward import SwingConfig

cfg = yaml.safe_load(Path("configs/finrl_rotation_v27.yaml").read_text(encoding="utf-8"))
rule = cfg["rule"]
OUT = Path("artifacts/reports/swing/v27_finrl_rotation.json")
FIRST, LAST = cfg["first_scored"], cfg["last_scored"]


@dataclass(frozen=True)
class Rotation:
    p: Panel
    ends: list[int]  # index of the last session of each week
    spy: int
    qqq: int
    vol: np.ndarray  # stands in for VIX
    fallback: list[int]


@dataclass(frozen=True)
class Book:
    t: int  # the signal's session; the book trades at the next open
    weights: dict[int, float]
    slots: tuple[tuple[tuple[int, ...], tuple[float, ...]], ...]  # each kept group's members and the weights its picks got
    budget: float
    kept: tuple[str, ...]


def week_ends(dates: tuple[date, ...]) -> list[int]:
    return [i for i in range(len(dates) - 1) if dates[i].isocalendar()[:2] != dates[i + 1].isocalendar()[:2]]


def realized_vol(close: np.ndarray) -> np.ndarray:
    """Annualized volatility of the last vol_days daily returns, in percent, at each session."""
    r = close[1:] / close[:-1] - 1
    out = np.full(len(close), np.nan)
    n = rule["vol_days"]
    for t in range(n, len(close)):
        out[t] = r[t - n: t].std(ddof=1) * 252**0.5 * 100
    return out


def rotation(p: Panel) -> Rotation:
    s = list(p.symbols)
    return Rotation(p, week_ends(p.dates), s.index(cfg["market"]), s.index(cfg["benchmark"]), realized_vol(p.close[:, s.index(cfg["market"])]), [s.index(x) for x in cfg["fallback"]])


def robust_z(v: np.ndarray) -> float:
    """How far the last value sits from the median, in median absolute deviations."""
    center, last = float(np.median(v)), float(v[-1])
    mad = float(np.median(np.abs(v - center)))
    if mad < rule["min_mad"]:
        return 0.0 if abs(last - center) < rule["min_mad"] else float(10.0 * np.sign(last - center))
    return (last - center) / mad


def strength(weekly: np.ndarray, members: list[int], bench: int) -> tuple[float, float]:
    """A group's information ratio against the benchmark, and its cumulative return above it."""
    p = weekly[-rule["group_weeks"]:]
    r = p[1:] / p[:-1] - 1
    group, b = r[:, members].mean(axis=1), r[:, bench]
    e = group - b
    mad = float(np.median(np.abs(e - np.median(e))))
    spread = mad if mad > 0 else float(e.std(ddof=1))
    return (float(e.mean()) / spread if spread > 0 else 0.0), float(np.prod(1 + group) - np.prod(1 + b))


def budget(m: Rotation, k: int, weekly: np.ndarray) -> float:
    """Share of capital to invest, from the risk score and that session's shock check."""
    t = m.ends[k]
    spx = weekly[:, m.spy]
    score = 0
    if len(spx) >= rule["trend_weeks"] and spx[-1] < spx[-rule["trend_weeks"]:].mean():
        score += 1
    if len(spx) >= rule["drawdown_weeks"] and spx[-1] / spx[-rule["drawdown_weeks"]:].max() - 1 <= -rule["drawdown"]:
        score += 1
    v = m.vol[m.ends[: k + 1]]
    v = v[np.isfinite(v)]
    if len(v) >= rule["vol_min_weeks"] and robust_z(v[-rule["vol_weeks"]:]) >= rule["vol_z"]:
        score += 1
    c = m.p.close
    n = rule["shock_days"]
    shock = t >= n and min(c[t, m.spy] / c[t - n, m.spy], c[t, m.qqq] / c[t - n, m.qqq]) - 1 <= rule["shock_return"]
    d = m.vol[: t + 1]
    d = d[np.isfinite(d)]
    if not shock and len(d) >= rule["shock_vol_min_sessions"]:
        w = rule["shock_vol_sessions"]
        shock = robust_z(d[-w:]) >= rule["vol_z"] and robust_z(np.diff(d)[-w:]) >= rule["shock_delta_z"]
    return rule["budgets"][-1] if shock else rule["budgets"][min(score, len(rule["budgets"]) - 1)]


def scores(r: np.ndarray) -> list[float]:
    """z of each column's last residual to the group, on the rank_weeks of returns in r."""
    residual = r - r.mean(axis=1, keepdims=True)
    return [float(np.clip(robust_z(residual[:, i]), -rule["max_z"], rule["max_z"])) for i in range(r.shape[1])]


def choose(m: Rotation, k: int, groups: dict[str, list[int]]) -> Book:
    t = m.ends[k]
    weekly = m.p.close[m.ends[: k + 1]]
    b = budget(m, k, weekly)
    power = {g: strength(weekly, js, m.qqq) for g, js in groups.items()}
    kept = [g for g in sorted(groups, key=lambda g: -power[g][0]) if power[g][1] > 0][: rule["max_groups"]]
    if not kept:
        return Book(t, {j: b / len(m.fallback) for j in m.fallback}, (), b, ("fallback",))
    last = weekly[-rule["rank_weeks"] - 1:]
    r = last[1:] / last[:-1] - 1
    n = rule["exception_weeks"]
    back = weekly[-1] / weekly[-n] - 1 if len(weekly) >= n else np.zeros(weekly.shape[1])
    weights: dict[int, float] = {}
    picks: list[tuple[list[int], list[int]]] = []
    for g in kept:
        js = groups[g]
        z = scores(r[:, js])
        order = sorted(range(len(js)), key=lambda i: -z[i])[: rule["top_assets"]]
        picks.append((js, [js[i] for i in order]))
        for i in order:
            strong = z[i] >= rule["exception_z"] and back[js[i]] > 0 and back[js[i]] > rule["exception_return_multiple"] * back[m.qqq]
            weights[js[i]] = b / len(kept) / len(order) * (rule["exception_weight"] if strong else 1.0)
    total = sum(weights.values())
    if total > 1:
        weights = {j: w / total for j, w in weights.items()}
    return Book(t, weights, tuple((tuple(js), tuple(weights[j] for j in chosen)) for js, chosen in picks), b, tuple(kept))


def exposure(books: list[dict[int, float]], ts: list[int], shape: tuple[int, int]) -> np.ndarray:
    """Each book is held from the open after its signal until the open after the next signal."""
    x = np.zeros(shape)
    for i, weights in enumerate(books):
        stop = ts[i + 1] + 1 if i + 1 < len(ts) else shape[0]
        for j, w in weights.items():
            x[ts[i] + 1: stop, j] = shape[1] * w
    return x


def net(m: Rotation, books: list[Book], a: int, b: int, cost: float, cash: np.ndarray | None) -> np.ndarray:
    x = exposure([k.weights for k in books], [k.t for k in books], m.p.close.shape)
    return run(m.p, x, x, a, b, cost, None, cash).returns


def random_run(m: Rotation, books: list[Book], a: int, b: int, cash: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The same groups, budget and weights each week, with the assets picked at random from the same members."""
    drawn = []
    for k in books:
        w = dict(k.weights) if not k.slots else {}
        for members, weights in k.slots:
            for j, v in zip(rng.choice(members, len(weights), replace=False).tolist(), weights):
                w[j] = v
        drawn.append(w)
    x = exposure(drawn, [k.t for k in books], m.p.close.shape)
    return run(m.p, x, x, a, b, cfg["cost_bps"], None, cash).returns


def fidelity(m: Rotation, books: list[Book]) -> dict:
    """FinRL's report: Friday close to Friday close, no costs, cash earns nothing, Sharpe is annual return over annual volatility."""
    run_books = [k for k in books if cfg["fidelity_first"] <= m.p.dates[k.t] <= cfg["fidelity_last"]]
    c = m.p.close
    r = np.array([sum(w * (c[y.t, j] / c[x.t, j] - 1) for j, w in x.weights.items()) for x, y in zip(run_books, run_books[1:])])
    eq = np.r_[1.0, np.cumprod(1 + r)]
    years = (m.p.dates[run_books[-1].t] - m.p.dates[run_books[0].t]).days / 365.25
    ann = eq[-1] ** (1 / years) - 1
    vol = r.std(ddof=1) * 52**0.5
    return dict(first=str(m.p.dates[run_books[0].t]), last=str(m.p.dates[run_books[-1].t]), weeks=len(r), final_multiple=round(float(eq[-1]), 2), ann_return=round(float(ann) * 100, 2),
                sharpe=round(float(ann / vol), 2), max_drawdown=round(float((eq / np.maximum.accumulate(eq) - 1).min()) * 100, 2))


def closes_market(symbols: list[str]) -> Panel:
    p = load_panel(Path(cfg["bars_dir"]), [*symbols, cfg["market"], cfg["benchmark"]]).before(LAST + timedelta(days=1))
    return p


def fixed_list(groups: dict[str, list[str]]) -> tuple[Rotation, dict[str, list[int]]]:
    names = list(dict.fromkeys(s for g in groups.values() for s in g))
    names += [s for s in cfg["fallback"] if s not in names and s not in (cfg["market"], cfg["benchmark"])]
    names = [s for s in names if s not in (cfg["market"], cfg["benchmark"])]
    m = rotation(closes_market(names))
    at = {s: j for j, s in enumerate(m.p.symbols)}
    return m, {g: [at[s] for s in v] for g, v in groups.items()}


def fixed_books(m: Rotation, groups: dict[str, list[int]]) -> list[Book]:
    return [choose(m, k, groups) for k, t in enumerate(m.ends) if m.p.dates[t] >= cfg["fidelity_first"]]


def free_panel() -> tuple[Panel, Panel]:
    """The 300 with the ETFs the rule needs added as columns, on the 300's calendar, and the 300 alone."""
    swing = load_config(Path(cfg["swing_config"]), SwingConfig)
    stocks = load_panel(swing.dataset_dir, swing.symbols).before(LAST + timedelta(days=1))
    extra = list(dict.fromkeys([*[s for v in cfg["free_fixed"].values() for s in v], *cfg["fallback"], cfg["market"], cfg["benchmark"]]))
    etf = load_panel(Path(cfg["bars_dir"]), extra)
    at = {d: i for i, d in enumerate(etf.dates)}
    idx = [at[d] for d in stocks.dates]
    join = lambda a, b: np.hstack([a, b[idx]])
    return Panel(stocks.dates, (*stocks.symbols, *extra), join(stocks.open, etf.open), join(stocks.high, etf.high), join(stocks.low, etf.low), join(stocks.close, etf.close), join(stocks.volume, etf.volume)), stocks


def yearly_lists(p: Panel, n: int, etf_close: np.ndarray, mkt: np.ndarray, only: list[int] | None = None) -> dict[int, dict[str, list[int]]]:
    """Each year's groups, from data through the prior year's last session. The first n columns are the 300."""
    years = only or sorted({d.year for d in p.dates if d >= FIRST} | {FIRST.year})
    etfs = bottleneck.cfg["etfs"]
    at = {s: j for j, s in enumerate(p.symbols)}
    out = {}
    for y in years:
        t = max(i for i, d in enumerate(p.dates) if d.year < y)
        sector = bottleneck.sectors(p.close[:, :n], etf_close, mkt, t + 1)
        dollars = (p.close[t + 1 - cfg["list_days"]: t + 1, :n] * p.volume[t + 1 - cfg["list_days"]: t + 1, :n]).mean(axis=0)
        groups = {g: [at[s] for s in fixed] for g, fixed in cfg["free_fixed"].items()}
        for g, count in cfg["list_share"].items():
            wanted = np.nonzero(np.isin(sector, [etfs.index(e) for e in cfg["list_etfs"][g]]))[0]
            groups[g] = wanted[np.argsort(-dollars[wanted], kind="stable")[:count]].tolist() + groups[g]
        out[y] = groups
    return out


def free_books(m: Rotation, lists: dict[int, dict[str, list[int]]]) -> list[Book]:
    return [choose(m, k, lists[m.p.dates[t].year]) for k, t in enumerate(m.ends) if m.p.dates[t].year >= FIRST.year]


def daily(p: Panel, j: int) -> np.ndarray:
    return np.r_[0.0, p.close[1:, j] / p.close[:-1, j] - 1]


def window(dates: tuple[date, ...]) -> tuple[int, int]:
    ok = [i for i, d in enumerate(dates) if FIRST <= d <= LAST]
    return ok[0], ok[-1] + 1


def cash_for(dates: tuple[date, ...]) -> np.ndarray:
    bil = pl.read_parquet(Path(cfg["cash_dir"]) / f"{cfg['cash_symbol']}.parquet").sort("date")
    px = pl.DataFrame(dict(date=list(dates)), schema={"date": pl.Date}).join(bil.select("date", "close"), on="date", how="left")["close"].forward_fill().backward_fill().to_numpy()
    return np.r_[0.0, px[1:] / px[:-1] - 1]


def gap_interval(d: np.ndarray) -> list[float]:
    rng = np.random.default_rng(cfg["seed"])
    block, count = cfg["block"], int(np.ceil(len(d) / cfg["block"]))
    starts = rng.integers(0, len(d) - block + 1, (cfg["bootstrap_samples"], count))
    ids = (starts[:, :, None] + np.arange(block)).reshape(len(starts), -1)[:, : len(d)]
    return [round(float(x) * 252 * 100, 1) for x in np.percentile(d[ids].mean(axis=1), [2.5, 97.5])]


def shape_of(m: Rotation, books: list[Book], a: int) -> dict:
    live = [k for k in books if k.t >= a - 6]
    w = [k.weights for k in live]
    turn = [sum(abs(x.get(j, 0.0) - y.get(j, 0.0)) for j in set(x) | set(y)) / 2 for x, y in zip(w, w[1:])]
    kept: dict[str, int] = {}
    for k in live:
        for g in k.kept:
            kept[g] = kept.get(g, 0) + 1
    return dict(weeks=len(live), weekly_turnover=round(float(np.mean(turn)), 3), mean_budget=round(float(np.mean([k.budget for k in live])), 3), weeks_in_group=kept)


def by_year(dates: list[date], r: np.ndarray, others: dict[str, np.ndarray]) -> dict:
    y = np.array([d.year for d in dates])
    grow = lambda x, a: round(float((1 + x[y == a]).prod() - 1) * 100, 1)
    return {int(a): dict(rule=grow(r, a), **{k: grow(v, a) for k, v in others.items()}) for a in sorted(set(y))}


def luck_tests(r: np.ndarray, spy: np.ndarray, rf: np.ndarray, dates: list[date]) -> dict:
    """Without the two best years against SPY, and each half."""
    y = np.array([d.year for d in dates])
    gap = {int(a): float((1 + r[y == a]).prod() - (1 + spy[y == a]).prod()) for a in sorted(set(y))}
    best = sorted(gap, key=lambda a: -gap[a])[:2]
    keep = ~np.isin(y, best)
    half = np.arange(len(r)) < len(r) // 2
    sharper = lambda i: f.stats(r[i], rf[i])["sharpe"] > f.stats(spy[i], rf[i])["sharpe"]
    return dict(best_years=best, not_luck=bool(sharper(keep)), both_halves=bool(sharper(half) and sharper(~half)))


def draws(m: Rotation, books: list[Book], a: int, b: int, cash: np.ndarray, rf: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(cfg["seed"])
    return np.array([f.stats(random_run(m, books, a, b, cash, rng), rf)["ann_return"] for _ in range(cfg["random_draws"])])


def cmd_fetch() -> None:
    symbols = sorted({s for k in ("groups_v121", "groups_v122", "free_fixed") for v in cfg[k].values() for s in v} | set(cfg["fallback"]) | {cfg["market"], cfg["benchmark"]})
    out = Path(cfg["bars_dir"])
    out.mkdir(parents=True, exist_ok=True)
    fetch = http_fetch(load_credentials(Path(".env")), timeout_s=60, allowed=HISTORY_PATHS)
    for s in symbols:
        if (out / f"{s}.parquet").exists():
            continue
        rows = daily_rows(iter_history(fetch, HISTORY_BARS_PATH, s, f"{cfg['history_start']}T00:00:00Z", f"{cfg['last_fetch']}T23:59:59Z", "sip",
                                       min_interval_s=cfg["min_interval_s"], extra={"timeframe": "1Day", "adjustment": "all"}))
        pl.DataFrame(rows, schema={"date": pl.Date, "open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64}).write_parquet(out / f"{s}.parquet")
        print(s, len(rows), rows[0]["date"] if rows else None, flush=True)
    print("bars stored", len(list(out.glob("*.parquet"))), flush=True)


def cmd_fidelity() -> None:
    res = dict(target=cfg["fidelity_target"])
    for name in ("groups_v121", "groups_v122"):
        m, groups = fixed_list(cfg[name])
        res[name] = fidelity(m, fixed_books(m, groups))
    print(json.dumps(res, indent=1))


def cmd_run() -> None:
    m, groups = fixed_list(cfg["groups_v121"])
    books = fixed_books(m, groups)
    a, b = window(m.p.dates)
    cash = cash_for(m.p.dates)
    rf = cash[a:b]
    dates = list(m.p.dates[a:b])
    spy, qqq = daily(m.p, m.spy)[a:b], daily(m.p, m.qqq)[a:b]
    r_list = net(m, books, a, b, cfg["cost_bps"], cash)

    x = exposure([k.weights for k in books], [k.t for k in books], m.p.close.shape)
    same_close = np.vstack([x[1:], np.zeros((1, x.shape[1]))])  # holds the new book from the signal's close, a trade nobody could make
    costs = dict(friday_close_no_costs_no_cash=f.stats(run(m.p, x, same_close, a, b, 0.0, None, None).returns, rf), next_open=f.stats(run(m.p, x, x, a, b, 0.0, None, None).returns, rf),
                 plus_costs=f.stats(run(m.p, x, x, a, b, cfg["cost_bps"], None, None).returns, rf), plus_cash=f.stats(r_list, rf))

    m2, groups2 = fixed_list(cfg["groups_v122"])
    books2 = fixed_books(m2, groups2)
    a2, b2 = window(m2.p.dates)
    r_v122 = net(m2, books2, a2, b2, cfg["cost_bps"], cash_for(m2.p.dates))

    full, stocks = free_panel()
    e = load_panel(Path(cfg["etf_dir"]), bottleneck.cfg["etfs"] + [bottleneck.cfg["market"]])
    at = {d: i for i, d in enumerate(e.dates)}
    idx = [at[d] for d in full.dates]
    lists = yearly_lists(full, stocks.close.shape[1], e.close[idx, :-1], e.close[idx, -1])
    free = rotation(full)
    free_b = free_books(free, lists)
    fa, fb = window(full.dates)
    fcash = cash_for(full.dates)
    frf = fcash[fa:fb]
    fdates = list(full.dates[fa:fb])
    r_free = net(free, free_b, fa, fb, cfg["cost_bps"], fcash)
    fspy, fqqq = daily(full, free.spy)[fa:fb], daily(full, free.qqq)[fa:fb]
    hold = simulate(stocks, exposures(stocks, *HOLD), fa, fb, cfg["cost_bps"], fcash).returns

    s = dict(finrl_list=f.stats(r_list, rf), spy=f.stats(spy, rf), qqq=f.stats(qqq, rf), free_list=f.stats(r_free, frf), hold=f.stats(hold, frf), free_spy=f.stats(fspy, frf), free_qqq=f.stats(fqqq, frf))
    luck = luck_tests(r_free, fspy, frf, fdates)
    random_free = draws(free, free_b, fa, fb, fcash, frf)
    random_list = draws(m, books, a, b, cash, rf)
    tests = dict(t1=f.beats(s["finrl_list"], s["spy"]), t2=f.beats(s["finrl_list"], s["qqq"]), t3=bool(f.beats(s["free_list"], s["free_spy"]) and f.beats(s["free_list"], s["hold"])), t4=luck["not_luck"],
                 t5=luck["both_halves"], t6=bool(s["free_list"]["ann_return"] > np.percentile(random_free, 95)))
    names = {y: {g: [full.symbols[j] for j in v] for g, v in groups_y.items()} for y, groups_y in lists.items()}
    res = dict(config="configs/finrl_rotation_v27.yaml", window=f"{dates[0]}..{dates[-1]}", tests=tests, passed=all(tests.values()), figures=s, what_each_correction_costs=costs,
               free_list_luck=luck, finrl_list_side=luck_tests(r_list, spy, rf, dates) | dict(ranking_beats_95=bool(s["finrl_list"]["ann_return"] > np.percentile(random_list, 95))),
               random_free=dict(p50=round(float(np.median(random_free)), 4), p95=round(float(np.percentile(random_free, 95)), 4)),
               random_finrl=dict(p50=round(float(np.median(random_list)), 4), p95=round(float(np.percentile(random_list, 95)), 4)),
               at_side_cost=dict(finrl_list=f.stats(net(m, books, a, b, cfg["side_cost_bps"], cash), rf), free_list=f.stats(net(free, free_b, fa, fb, cfg["side_cost_bps"], fcash), frf)),
               v122=f.stats(r_v122, cash_for(m2.p.dates)[a2:b2]), free_beats_qqq=f.beats(s["free_list"], s["free_qqq"]),
               by_year=dict(finrl_list=by_year(dates, r_list, dict(spy=spy, qqq=qqq)), free_list=by_year(fdates, r_free, dict(spy=fspy, qqq=fqqq, hold=hold))),
               gap_95=dict(finrl_vs_spy=gap_interval(r_list - spy), free_vs_spy=gap_interval(r_free - fspy), free_vs_hold=gap_interval(r_free - hold)),
               shape=dict(finrl_list=shape_of(m, books, a), free_list=shape_of(free, free_b, fa)), free_lists=names)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "free_lists"}, indent=1, default=str))


if __name__ == "__main__":
    {"fetch": cmd_fetch, "fidelity": cmd_fidelity, "run": cmd_run}[sys.argv[1]]()
