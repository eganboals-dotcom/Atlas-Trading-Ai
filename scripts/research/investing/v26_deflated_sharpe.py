"""Deflated Sharpe Ratio of earnings quality (see configs/deflated_sharpe_v26.yaml). NOT pre-registered: it passes or fails nothing.
Replays the v24 tracker's code over v23's window, which gives the same figures as v23, and writes
artifacts/reports/swing/v26_deflated_sharpe.json.

    .venv/Scripts/python.exe scripts/research/investing/v26_deflated_sharpe.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
from datetime import date
from pathlib import Path
from statistics import NormalDist

import numpy as np
import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "trackers"))
import v23_filings as f
import v24_quality_track as tracker

cfg = yaml.safe_load(Path("configs/deflated_sharpe_v26.yaml").read_text(encoding="utf-8"))
OUT = Path(cfg["reports_dir"]) / "v26_deflated_sharpe.json"
PHI = NormalDist()
EULER = 0.5772156649015329


def expected_best(n: int, variance: float) -> float:
    """The Sharpe the best of n useless rules would be expected to reach, in the units of variance."""
    if n < 2:
        return 0.0
    return float(variance**0.5 * ((1 - EULER) * PHI.inv_cdf(1 - 1 / n) + EULER * PHI.inv_cdf(1 - 1 / (n * np.e))))


def deflated(r: np.ndarray, n: int, variance: float) -> float:
    """P(true daily Sharpe above the best of n useless rules). variance is of the rules' daily Sharpes."""
    sr = float(r.mean() / r.std(ddof=1))
    z = (r - r.mean()) / r.std(ddof=1)
    skew, kurt = float((z**3).mean()), float((z**4).mean())
    return PHI.cdf((sr - expected_best(n, variance)) * (len(r) - 1) ** 0.5 / (1 - skew * sr + (kurt - 1) / 4 * sr**2) ** 0.5)


def trial_sharpes() -> tuple[list[float], list[float]]:
    """Annual Sharpes of the rules run on this window, and their annual Sharpes against Hold (2 / sqrt(years to prove))."""
    d = Path(cfg["reports_dir"])
    v23 = json.loads((d / "v23_filings.json").read_text(encoding="utf-8"))
    v22 = json.loads((d / "v22_bottleneck.json").read_text(encoding="utf-8"))
    v20 = json.loads((d / "v20_others.json").read_text(encoding="utf-8"))
    rules = [v23[k] for k in ("buybacks", "asset_growth", "earnings_quality", "quality_value")] + [v22[k] for k in ("g", "h")]
    sharpe = [r["rule"]["sharpe"] for r in rules] + [v20["rule_e"]["top_fifth"]["sharpe"], v20["rule_f"]["book"]["sharpe"]]
    gap = [2 / r["years_to_prove"] ** 0.5 if r["years_to_prove"] else 0.0 for r in rules]
    return sharpe, gap


def replay() -> dict[str, np.ndarray]:
    """Daily returns of the rule and Hold over v23's window, from the code the frozen tracker runs."""
    config = yaml.safe_load(tracker.CONFIG.read_text(encoding="utf-8"))
    config["rule"] = config["rule"] | dict(first_day=f.cfg["first_scored"])
    config["spy_dir"] = "data/processed/alpaca_daily"
    symbols = tracker.universe(config["rule"])
    with tempfile.TemporaryDirectory() as tmp:
        for s in symbols:
            files = [Path(f.cfg[k]) / f"{s}.parquet" for k in ("fundamentals_dir", "extra_dir")]
            pl.concat([pl.read_parquet(x) for x in files if x.exists()]).write_parquet(Path(tmp) / f"{s}.parquet")
        config["filings_dir"] = tmp
        m = tracker.market(config, symbols)
        n = sum(d <= date.fromisoformat(f.cfg["last_scored"]) for d in m.p.dates)
        p = dataclasses.replace(m.p, dates=m.p.dates[:n], **{c: getattr(m.p, c)[:n] for c in ("open", "high", "low", "close", "volume")})
        m = tracker.Market(p, m.cash[:n], m.traded[:n], m.spy[:n])
        books = [tracker.choose(config, m, c) for c in tracker.due([], p.dates, config["rule"]["every"])]
    r = tracker.score(config, m, books)
    return dict(rule=r["top"], hold=r["Hold"], cash=m.cash[1:n])


def main() -> None:
    sharpe, gap = trial_sharpes()
    r = replay()
    above_cash, above_hold = r["rule"] - r["cash"], r["rule"] - r["hold"]
    var_cash, var_gap = float(np.var(sharpe, ddof=1)) / 252, float(np.var(gap, ddof=1)) / 252
    res: dict = dict(note="not pre-registered", main_trials=cfg["main_trials"], bar=cfg["bar"], annual_sharpe_above_cash=round(float(above_cash.mean() / above_cash.std() * 252**0.5), 3),
                     annual_sharpe_gap_to_hold=round(float(above_hold.mean() / above_hold.std() * 252**0.5), 3), trial_sharpes=sharpe, trial_gap_sharpes=[round(x, 3) for x in gap],
                     by_trials={})
    for n in cfg["trials"]:
        res["by_trials"][n] = dict(best_of_n_useless_gap=round(expected_best(n, var_gap) * 252**0.5, 3), dsr_gap_to_hold=round(deflated(above_hold, n, var_gap), 3),
                                   best_of_n_useless_cash=round(expected_best(n, var_cash) * 252**0.5, 3), dsr_above_cash=round(deflated(above_cash, n, var_cash), 3))
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
