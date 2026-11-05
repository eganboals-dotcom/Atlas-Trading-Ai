"""Forward tracker for the ETF-only rotation (see configs/etf_rotation_track_v32.yaml, which holds the frozen rule and its verdict).

Reads the ETF bars the rotation tracker downloads first, records each week's book, scores them against SPY and its ETFs held equally, and
writes artifacts/reports/etf_rotation_track/tracker.md. Research only: no order is placed anywhere.

    .venv/Scripts/python.exe scripts/trackers/v32_etf_rotation_track.py
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "investing"))
import v27_finrl_rotation as v
import v30_rotation_track as base
from atlas.swing.backtest import run as run_book, summary
from atlas.swing.data import Panel, load_panel

CONFIG = Path("configs/etf_rotation_track_v32.yaml")
LONG = 60  # annual figures and Sharpe are shown only after this many days, as in the other trackers


def names(rule: dict) -> list[str]:
    return list(dict.fromkeys([*[s for g in rule["groups"].values() for s in g], *v.cfg["fallback"], v.cfg["market"], v.cfg["benchmark"]]))


def pick(m: v.Rotation, k: int, groups: dict[str, list[int]]) -> tuple[dict[int, float], float, list[str]]:
    """Weights, budget and kept groups for one week-end: v27's group choice and budget, with no ranking inside a group."""
    weekly = m.p.close[m.ends[: k + 1]]
    b = v.budget(m, k, weekly)
    power = {g: v.strength(weekly, js, m.qqq) for g, js in groups.items()}
    kept = [g for g in sorted(groups, key=lambda g: -power[g][0]) if power[g][1] > 0][: v.rule["max_groups"]]
    if not kept:
        return {j: b / len(m.fallback) for j in m.fallback}, b, ["fallback"]
    return {j: b / len(kept) / len(groups[g]) for g in kept for j in groups[g]}, b, kept


def fingerprint(rule: dict) -> str:
    code = [inspect.getsource(x) for x in (v.week_ends, v.realized_vol, v.rotation, v.robust_z, v.strength, v.budget, v.exposure, base.closed_weeks, base.sleeve_mix, pick)]
    settings = {k: v.cfg[k] for k in ("rule", "fallback", "market", "benchmark")}
    return hashlib.sha256(json.dumps(dict(rule=rule, rotation=settings, code=code), sort_keys=True).encode("utf-8")).hexdigest()


def check_state(cfg: dict, state_path: Path) -> dict:
    """Refuses a changed rule."""
    fp = fingerprint(cfg["rule"])
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else dict(fingerprint=fp, frozen=str(date.today()), books=[])
    if state["fingerprint"] != fp:
        sys.exit("refused: the frozen rule in configs/etf_rotation_track_v32.yaml, v27's settings or its code has changed")
    return state


def market(cfg: dict, today: date) -> tuple[v.Rotation, np.ndarray]:
    p = load_panel(Path(cfg["etf_dir"]), names(cfg["rule"]))
    bil = pl.read_parquet(Path(cfg["cash_dir"]) / f"{cfg['rule']['cash_symbol']}.parquet").sort("date")
    px = pl.DataFrame(dict(date=list(p.dates)), schema={"date": pl.Date}).join(bil.select("date", "close"), on="date", how="left")["close"].forward_fill().backward_fill().to_numpy()
    return dataclasses.replace(v.rotation(p), ends=base.closed_weeks(p.dates, today)), np.r_[0.0, px[1:] / px[:-1] - 1]


def groups_of(p: Panel, rule: dict) -> dict[str, list[int]]:
    at = {s: j for j, s in enumerate(p.symbols)}
    return {g: [at[s] for s in members] for g, members in rule["groups"].items()}


def score(cfg: dict, m: v.Rotation, cash: np.ndarray, books: list[dict]) -> dict[str, np.ndarray]:
    """Daily returns from first_day, through the session before a book that hasn't been chosen would trade. Each is sold at its last close."""
    rule, p = cfg["rule"], m.p
    day = date.fromisoformat(rule["first_day"])
    weeks = [k for k, t in enumerate(m.ends) if p.dates[t] >= date.fromisoformat(rule["first_signal"])]
    a = p.dates.index(day) if day in p.dates else len(p.dates)
    end = m.ends[weeks[len(books)]] + 1 if len(weeks) > len(books) else len(p.dates)
    if not books or end <= a:
        return {}
    at = {s: j for j, s in enumerate(p.symbols)}
    mix = [{at[s]: w for s, w in x.items()} for x in base.sleeve_mix([b["weights"] for b in books], rule["sleeves"])]
    x = v.exposure(mix, [p.dates.index(date.fromisoformat(b["signal"])) for b in books], p.close.shape)
    members = sorted({j for g in groups_of(p, rule).values() for j in g})
    plain = np.zeros_like(p.close)
    plain[a:end, members] = p.close.shape[1] / len(members)
    go = lambda y: run_book(p, y, y, a, end, rule["cost_bps"], None, cash).returns
    return dict(rotation=go(x), SPY=v.daily(p, m.spy)[a:end], plain=go(plain), QQQ=v.daily(p, m.qqq)[a:end], rf=cash[a:end])


def report(cfg: dict, state: dict, p: Panel, r: dict[str, np.ndarray]) -> str:
    rule = cfg["rule"]
    lines = ["# ETF rotation: forward tracker", "", f"Frozen {state['frozen']}, first day {rule['first_day']}. Research only: no order is placed anywhere, and holdings are not advice.", "",
             f"**{base.verdict(rule, r)}**", ""]
    if r:
        days = len(r["rotation"])
        first = p.dates.index(date.fromisoformat(rule["first_day"]))
        lines += [f"Days {p.dates[first]} to {p.dates[first + days - 1]}. Annual figures and Sharpe are shown after {LONG} days.", "", "| Book | Return | vs SPY | Return/yr | Sharpe | Max DD |", "|---|---|---|---|---|---|"]
        total = {k: float(np.prod(1 + r[k]) - 1) for k in ("rotation", "SPY", "plain", "QQQ")}
        for k, name in dict(rotation="ETF rotation", SPY="SPY", plain="All its ETFs held equally", QQQ="QQQ (side)").items():
            s = summary(r[k], r["rf"])
            lines.append(f"| {name} | {total[k]:+.2%} | {'' if k == 'SPY' else f'{total[k] - total['SPY']:+.2%}'} | {f'{s['ann_return']:+.1%}' if days >= LONG else '-'} | {f'{s['sharpe']:.2f}' if days >= LONG else '-'} | {s['max_drawdown']:.1%} |")
        lines.append("")
    if state["books"]:
        b = state["books"][-1]
        mix = base.sleeve_mix([x["weights"] for x in state["books"]], rule["sleeves"])[-1]
        lines += [f"Book {len(state['books'])}, from the close of {b['signal']}: groups {', '.join(b['kept'])}, invested {b['budget']:.0%}.", "",
                  "Combined weights: " + ", ".join(f"{s} {w:.1%}" for s, w in sorted(mix.items(), key=lambda x: -x[1])), ""]
    return "\n".join(lines)


def run(cfg: dict, now: datetime) -> str:
    rule = cfg["rule"]
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    state = check_state(cfg, state_path)
    m, cash = market(cfg, now.astimezone(base.ET).date())
    p = m.p
    first = date.fromisoformat(rule["first_signal"])
    groups = groups_of(p, rule)
    for k in [k for k, t in enumerate(m.ends) if p.dates[t] >= first][len(state["books"]):]:
        w, budget, kept = pick(m, k, groups)
        state["books"].append(dict(signal=str(p.dates[m.ends[k]]), budget=budget, kept=kept, weights={p.symbols[j]: x for j, x in w.items()}))  # a book never changes once recorded
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    text = report(cfg, state, p, score(cfg, m, cash, state["books"]))
    (out / "tracker.md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    print(run(yaml.safe_load(CONFIG.read_text(encoding="utf-8")), datetime.now(UTC)))
