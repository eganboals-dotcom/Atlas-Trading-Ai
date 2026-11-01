"""Bottlenecks in growing sectors (see configs/bottleneck_v22.yaml, which holds the pre-registered tests), on the unseen
days. Writes artifacts/reports/swing/v22_bottleneck.json.

    .venv/Scripts/python.exe scripts/research/investing/v22_bottleneck.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "trading"))
import v19_pairs as pairs
from atlas.swing.backtest import run, simulate, summary
from atlas.swing.data import load_panel
from atlas.swing.families import HOLD, exposures, prev_close
from atlas.swing.fundamentals import NO_GROSS_PROFIT

cfg = yaml.safe_load(Path("configs/bottleneck_v22.yaml").read_text(encoding="utf-8"))
OUT = Path("artifacts/reports/swing/v22_bottleneck.json")
ITEMS = ("revenue", "gross_profit", "cost_of_revenue", "assets")


def latest_two(rows: pl.DataFrame, d: date) -> dict[str, tuple[float, float]]:
    """For each item, (latest fiscal year, the year before), as last filed before day d."""
    out = {}
    lo, hi = cfg["prior_year_days"]
    known = rows.filter(pl.col("filed") < d).sort("filed")
    for item in ITEMS:
        vals = dict(known.filter(pl.col("item") == item).select("end", "val").rows())  # a later filing replaces an earlier one
        if not vals:
            continue
        end = max(vals)
        if (d - end).days > cfg["max_age_days"]:
            continue
        prev = [e for e in vals if lo <= (end - e).days <= hi]
        if prev:
            out[item] = (vals[end], vals[min(prev, key=lambda e: abs((end - e).days - 365))])
    return out


def company(rows: pl.DataFrame, d: date, sym: str) -> dict[str, float] | None:
    """Revenue now and before, gross-margin change and asset growth; None if a company can't be measured."""
    x = latest_two(rows, d)
    if sym in NO_GROSS_PROFIT or "revenue" not in x or "assets" not in x:
        return None
    rev, rev_prev = x["revenue"]
    if "gross_profit" in x:
        gp, gp_prev = x["gross_profit"]
    elif "cost_of_revenue" in x:
        gp, gp_prev = rev - x["cost_of_revenue"][0], rev_prev - x["cost_of_revenue"][1]
    else:
        return None
    if rev <= 0 or rev_prev <= 0 or x["assets"][1] <= 0:
        return None
    return dict(rev=rev, rev_prev=rev_prev, growth=rev / rev_prev - 1, margin_change=gp / rev - gp_prev / rev_prev, asset_growth=x["assets"][0] / x["assets"][1] - 1)


def sectors(close: np.ndarray, etf: np.ndarray, mkt: np.ndarray, t: int) -> np.ndarray:
    """Index of the ETF each stock moved with most, above the market, on the group_days up to the day before t."""
    n = cfg["group_days"]
    ret = lambda x: x[t - n: t] / x[t - n - 1: t - 1] - 1
    m = ret(mkt)[:, None]
    a, b = ret(close) - m, ret(etf) - m
    a, b = (a - a.mean(axis=0)) / a.std(axis=0), (b - b.mean(axis=0)) / b.std(axis=0)
    return (a.T @ b).argmax(axis=1)


def rebalance(group: np.ndarray, firms: dict[int, dict], n: int, skip: set[int]) -> tuple[list[int], list[int], list[int]]:
    """Rule G's stocks, rule H's, and the rest of the growing sectors. Sectors in skip can't count as growing."""
    members: dict[int, list[int]] = {}
    for j, f in firms.items():
        members.setdefault(int(group[j]), []).append(j)
    eligible = {g: js for g, js in members.items() if len(js) >= cfg["min_group"] and g not in skip}
    if not eligible:
        return [], [], []
    speed = {g: sum(firms[j]["rev"] for j in js) / sum(firms[j]["rev_prev"] for j in js) - 1 for g, js in eligible.items()}
    top = sorted(speed, key=lambda g: -speed[g])[: int(np.ceil(len(speed) / 3))]
    g_rule, h_rule, rest = [], [], []
    for g in top:
        for j in eligible[g]:
            f = firms[j]
            if f["margin_change"] > 0 and f["growth"] > speed[g]:
                g_rule.append(j)
                if f["asset_growth"] < f["growth"]:
                    h_rule.append(j)
            else:
                rest.append(j)
    return g_rule, h_rule, rest


def weights(n: int, names: list[int], everyone: list[int]) -> np.ndarray:
    w = np.zeros(n)
    use = names if len(names) >= cfg["min_names"] else everyone
    w[use] = n / len(use)
    return w


def stats(r: np.ndarray, rf: np.ndarray) -> dict:
    return {k: round(v, 4) for k, v in summary(r, rf).items()}


def beats(a: dict, b: dict) -> bool:
    return bool(a["sharpe"] > b["sharpe"] and a["ann_return"] >= b["ann_return"])


def main() -> None:
    m = pairs.market(cfg["swing_config"])
    p, v0, end = m.p, m.v0, m.end
    rf = m.cash[v0:end]
    n = p.close.shape[1]
    e = load_panel(Path(cfg["etf_dir"]), cfg["etfs"] + [cfg["market"]])
    pos = {d: i for i, d in enumerate(e.dates)}
    at = [pos[d] for d in p.dates]
    etf, mkt = e.close[at, :-1], e.close[at, -1]
    rows = {s: pl.read_parquet(Path(cfg["fundamentals_dir"]) / f"{s}.parquet") for s in p.symbols if (Path(cfg["fundamentals_dir"]) / f"{s}.parquet").exists()}
    tech = {cfg["etfs"].index(g) for g in cfg["tech_groups"]}
    every = list(range(n))
    days = len(p.dates)
    x = {k: np.zeros((days, n)) for k in ("g", "h", "rest", "g_ex", "h_ex", "hold_ex")}
    counts, tech_share = [], []
    for t in range(v0, end, cfg["every"]):
        d = p.dates[t]
        group = sectors(p.close, etf, mkt, t)
        firms = {j: f for j, s in enumerate(p.symbols) if s in rows and (f := company(rows[s], d, s)) is not None}
        g, h, rest = rebalance(group, firms, n, set())
        g_ex, h_ex, _ = rebalance(group, firms, n, tech)
        non_tech = [j for j in every if int(group[j]) not in tech]
        stop = t + cfg["every"]
        x["g"][t: stop], x["h"][t: stop], x["rest"][t: stop] = weights(n, g, every), weights(n, h, every), weights(n, rest, every)
        x["g_ex"][t: stop], x["h_ex"][t: stop] = weights(n, g_ex, non_tech), weights(n, h_ex, non_tech)
        x["hold_ex"][t: stop] = weights(n, non_tech, non_tech)
        counts.append(dict(day=str(d), firms=len(firms), g=len(g), h=len(h), g_ex=len(g_ex), h_ex=len(h_ex)))
        tech_share.append(sum(int(group[j]) in tech for j in g) / max(1, len(g)))
    ret = lambda k, c=cfg["cost_bps"]: run(p, x[k], x[k], v0, end, c, None, m.cash).returns
    h = simulate(p, exposures(p, *HOLD), v0, end, cfg["cost_bps"], m.cash).returns
    years = np.array([d.year for d in p.dates[v0:end]])
    half = np.arange(len(h)) < len(h) // 2
    sh = stats(h, rf)
    res: dict = dict(config="configs/bottleneck_v22.yaml", window=f"{p.dates[v0]}..{p.dates[end - 1]}", hold=sh, hold_ex_tech=stats(ret("hold_ex"), rf), the_rest=stats(ret("rest"), rf),
                     avg_names={k: round(float(np.mean([c[k] for c in counts])), 1) for k in ("firms", "g", "h", "g_ex", "h_ex")}, fallback_quarters=dict(g=sum(c["g"] < cfg["min_names"] for c in counts), h=sum(c["h"] < cfg["min_names"] for c in counts)),
                     tech_share_of_g=round(float(np.mean(tech_share)), 3), quarters=counts)
    tests = {}
    for name in ("g", "h"):
        r = ret(name)
        s = stats(r, rf)
        yearly = {int(y): dict(rule=round(float((1 + r[years == y]).prod() - 1) * 100, 1), hold=round(float((1 + h[years == y]).prod() - 1) * 100, 1)) for y in sorted(set(years))}
        best = sorted(yearly, key=lambda y: -(yearly[y]["rule"] - yearly[y]["hold"]))[:2]
        keep = ~np.isin(years, best)
        sharper = lambda i: stats(r[i], rf[i])["sharpe"] > stats(h[i], rf[i])["sharpe"]
        ex = stats(ret(f"{name}_ex"), rf)
        d = r - h
        edge, te = float(d.mean() * 252), float(d.std() * 252**0.5)
        turn = float(np.abs(np.diff(x[name][v0:end], axis=0)).sum(axis=1).mean() / n / 2 * 21)
        tests[name] = dict(test1=beats(s, sh), test2=bool(s["ann_return"] > res["the_rest"]["ann_return"]), test3=bool(sharper(keep)), test4=bool(sharper(half) and sharper(~half)), test5=beats(ex, res["hold_ex_tech"]))
        res[name] = dict(rule=s, without_tech=ex, at_5bps=stats(ret(name, cfg["side_cost_bps"]), rf), by_year=yearly, best_years=best, edge_pct=round(edge * 100, 2), monthly_turnover=round(turn, 3),
                         years_to_prove=round((2 * te / edge) ** 2, 1) if edge > 0 else None, passed=all(tests[name].values()))
    res["tests"] = tests
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "quarters"}, indent=1, default=str))


if __name__ == "__main__":
    main()
