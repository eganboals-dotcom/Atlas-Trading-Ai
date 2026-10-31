"""Attempts to break the pre-FOMC result of v20. NOT pre-registered: it passes or fails nothing, and v20's verdict stands.

Recomputes the windows from the raw bars, then tries placebo dates, random dates, other markets, earlier exits and each
event's weight in the mean. Days from 2026-04-28 on are never read. Writes artifacts/reports/swing/v20_followup.json.

    .venv/Scripts/python.exe scripts/v20_followup.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import yaml

cfg = yaml.safe_load(Path("configs/others_v20.yaml").read_text(encoding="utf-8"))
OUT = Path("artifacts/reports/swing/v20_followup.json")
COST = {"SPY": 0.2, "QQQ": 0.3, "IWM": 0.5, "DIA": 0.5}
DECISIONS = {d for year in cfg["fomc"] for d in year}
MINUTE = cfg["decision_time_minute"]


def prices(sym: str, minute: int) -> tuple[list, np.ndarray]:
    """The open of one minute on every session, read directly from the bars."""
    f = pl.read_parquet(f"data/processed/etf_1min_ohlcv/{sym}.parquet").filter(pl.col("m") == minute).sort("day")
    return f["day"].to_list(), f["o"].to_numpy()


def windows(sym: str, shift: int = 0, buy_minute: int = MINUTE, sell_minute: int = MINUTE) -> tuple[np.ndarray, np.ndarray]:
    """Returns of buying before each session and selling in it, shift sessions off the decision, and of all other days."""
    days, buy = prices(sym, buy_minute)
    sell_days, sell = prices(sym, sell_minute)
    px = dict(zip(sell_days, sell))
    c = COST[sym] / 1e4
    ev, other = [], []
    marks = {i + shift for i, d in enumerate(days) if d in DECISIONS}
    for i in range(1, len(days)):
        if days[i] not in px:
            continue
        r = px[days[i]] * (1 - c) / (buy[i - 1] * (1 + c)) - 1
        if i in marks:
            ev.append(r)
        elif days[i - 1] not in DECISIONS and days[i] not in DECISIONS:
            other.append(r)
    return np.array(ev), np.array(other)


def bps(x: np.ndarray) -> float:
    return round(float(x.mean()) * 1e4, 1)


def main() -> None:
    rng = np.random.default_rng(cfg["seed"])
    ev, other = windows("SPY")
    res: dict = dict(note="not pre-registered", events=len(ev), mean_bps=bps(ev), median_bps=round(float(np.median(ev)) * 1e4, 1), other_mean_bps=bps(other))
    se = (ev.var(ddof=1) / len(ev) + other.var(ddof=1) / len(other)) ** 0.5
    res["t_against_zero"] = round(float(ev.mean() / (ev.std(ddof=1) / len(ev) ** 0.5)), 2)
    res["t_against_other_days"] = round(float((ev.mean() - other.mean()) / se), 2)
    sims = rng.choice(other, (20000, len(ev))).mean(axis=1)
    res["random_days_beaten"] = round(float((ev.mean() > sims).mean()), 4)
    res["placebo_shift_bps"] = {s: bps(windows("SPY", s)[0]) for s in (-3, -2, -1, 1, 2, 3)}
    res["other_markets"] = {s: dict(mean_bps=bps(windows(s)[0]), other_bps=bps(windows(s)[1])) for s in ("QQQ", "IWM", "DIA")}
    res["sell_earlier_bps"] = {m: bps(windows("SPY", sell_minute=m)[0]) for m in (240, 255, 265, 269)}
    res["buy_at_open_sell_14_same_day_bps"] = bps(_same_day())
    order = np.sort(ev)[::-1]
    res["without_best_n_bps"] = {n: bps(order[n:]) for n in (1, 3, 5, 10)}
    res["years_to_show_vs_zero"] = round((2 * ev.std(ddof=1) / ev.mean()) ** 2 / 8, 1)
    res["years_to_show_vs_other_days"] = round((2 * ev.std(ddof=1) / (ev.mean() - other.mean())) ** 2 / 8, 1)
    OUT.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))


def _same_day() -> np.ndarray:
    """How much of the drift comes on decision day itself, from the open to 14:00."""
    opens, sell = (dict(zip(*prices("SPY", m))) for m in (0, MINUTE))  # half days have no 14:00 bar, so match on the date
    c = COST["SPY"] / 1e4
    return np.array([sell[d] * (1 - c) / (opens[d] * (1 + c)) - 1 for d in sorted(DECISIONS) if d in opens and d in sell])


if __name__ == "__main__":
    main()
