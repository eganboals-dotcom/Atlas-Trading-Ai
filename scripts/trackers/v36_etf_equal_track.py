"""Forward tracker for the ETF rotation's 13 ETFs held equally (see configs/etf_equal_track_v36.yaml, which holds the frozen rule and its verdict).

Reads the ETF bars the rotation tracker downloads first, scores the equal holding against SPY, and writes
artifacts/reports/etf_equal_track/tracker.md. Research only: no order is placed anywhere.

    .venv/Scripts/python.exe scripts/trackers/v36_etf_equal_track.py
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from atlas.swing.backtest import summary
from atlas.swing.data import Panel, load_panel

CONFIG = Path("configs/etf_equal_track_v36.yaml")
LONG = 60  # annual figures and Sharpe are shown only after this many days, as in the other trackers
NAME = "ETFs held equally"


def returns(rule: dict, p: Panel, cash: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Daily returns from first_day of the holding, SPY and cash, and the holding's weights after the last day.
    Equal at the close before first_day and again before the first session of each quarter, drifting in between."""
    at = {s: j for j, s in enumerate(p.symbols)}
    px, spy, n = p.close[:, [at[s] for s in rule["etfs"]]], p.close[:, at["SPY"]], len(rule["etfs"])
    a = max(1, next((t for t, d in enumerate(p.dates) if d >= date.fromisoformat(rule["first_day"])), len(p.dates)))
    w, out = np.zeros(n), []
    for t in range(a, len(p.dates)):
        cost = 0.0
        if t == a or (p.dates[t].month - 1) // 3 != (p.dates[t - 1].month - 1) // 3:
            cost = float(np.abs(1 / n - w).sum()) * rule["cost_bps"] / 1e4
            w = np.full(n, 1 / n)
        g = px[t] / px[t - 1] - 1
        day = float(w @ g)
        out.append((1 - cost) * (1 + day) - 1)
        w = w * (1 + g) / (1 + day)
    if not out:
        return {}, np.full(n, 1 / n)
    return dict(equal=np.array(out), SPY=spy[a:] / spy[a - 1: -1] - 1, rf=cash[a:]), w


def verdict(rule: dict, r: dict[str, np.ndarray]) -> str:
    n = rule["review_sessions"]
    days = len(r["equal"]) if r else 0
    if days < n:
        return f"Not reviewed yet: {days} of {n} sessions. Nothing before then is a verdict."
    e, s = (summary(r[k][:n], r["rf"][:n]) for k in ("equal", "SPY"))
    return f"Review after {n} sessions: {'PASS' if e['sharpe'] > s['sharpe'] and e['max_drawdown'] < s['max_drawdown'] else 'FAIL'}."


def fingerprint(rule: dict) -> str:
    code = [inspect.getsource(x) for x in (returns, verdict)]
    return hashlib.sha256(json.dumps(dict(rule=rule, code=code), sort_keys=True).encode("utf-8")).hexdigest()


def check_state(rule: dict, state_path: Path) -> dict:
    """Refuses a changed rule."""
    fp = fingerprint(rule)
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else dict(fingerprint=fp, frozen=str(date.today()))
    if state["fingerprint"] != fp:
        sys.exit("refused: the frozen rule in configs/etf_equal_track_v36.yaml or its code has changed")
    return state


def market(cfg: dict) -> tuple[Panel, np.ndarray]:
    rule = cfg["rule"]
    p = load_panel(Path(cfg["etf_dir"]), [*rule["etfs"], "SPY"])
    bil = pl.read_parquet(Path(cfg["cash_dir"]) / f"{rule['cash_symbol']}.parquet").sort("date")
    px = pl.DataFrame(dict(date=list(p.dates)), schema={"date": pl.Date}).join(bil.select("date", "close"), on="date", how="left")["close"].forward_fill().backward_fill().to_numpy()
    return p, np.r_[0.0, px[1:] / px[:-1] - 1]


def report(rule: dict, state: dict, p: Panel, r: dict[str, np.ndarray], w: np.ndarray) -> str:
    lines = [f"# {NAME}: forward tracker", "", f"Frozen {state['frozen']}, first day {rule['first_day']}. Research only: no order is placed anywhere, and holdings are not advice.", "",
             f"**{verdict(rule, r)}**", ""]
    if r:
        days = len(r["equal"])
        first = len(p.dates) - days
        lines += [f"Days {p.dates[first]} to {p.dates[-1]}. Annual figures and Sharpe are shown after {LONG} days.", "", "| Book | Return | vs SPY | Return/yr | Sharpe | Max DD |", "|---|---|---|---|---|---|"]
        total = {k: float(np.prod(1 + r[k]) - 1) for k in ("equal", "SPY")}
        for k, name in dict(equal=NAME, SPY="SPY").items():
            s = summary(r[k], r["rf"])
            lines.append(f"| {name} | {total[k]:+.2%} | {'' if k == 'SPY' else f'{total[k] - total['SPY']:+.2%}'} | {f'{s['ann_return']:+.1%}' if days >= LONG else '-'} | {f'{s['sharpe']:.2f}' if days >= LONG else '-'} | {s['max_drawdown']:.1%} |")
        lines.append("")
    lines += ["Combined weights: " + ", ".join(f"{s} {x:.1%}" for s, x in sorted(zip(rule["etfs"], w), key=lambda y: -y[1])), ""]
    return "\n".join(lines)


def run(cfg: dict) -> str:
    rule, out = cfg["rule"], Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    state = check_state(rule, state_path)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    p, cash = market(cfg)
    r, w = returns(rule, p, cash)
    text = report(rule, state, p, r, w)
    (out / "tracker.md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    print(run(yaml.safe_load(CONFIG.read_text(encoding="utf-8"))))
