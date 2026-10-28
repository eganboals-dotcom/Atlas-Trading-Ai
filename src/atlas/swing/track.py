"""Forward tracking of frozen swing rules against Hold, on days that came after the rules were fixed.

Backtests can be overfit, however carefully they are run; days that haven't happened yet can't be. Each run
scores every rule in the config on its universe from ``start_date`` to the latest completed day, next to
equal-weight Hold on the same symbols. Signals use the full history (lookbacks reach back before the start),
but returns count only from the open of ``start_date``. The rules' fingerprint is recorded on the first run,
and a later run with different rules is refused, so the record can't be edited after the fact.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

import numpy as np

from atlas.agents.genome import Genome
from atlas.config import StrictModel
from atlas.swing.backtest import simulate, summary
from atlas.swing.data import Panel, load_panel
from atlas.swing.families import FAMILIES, HOLD, exposures
from atlas.timeutils import MARKET_TZ

# A daily bar is complete once the closing auction has printed; before this time today's bar is left out.
BAR_FINAL = time(16, 20)


class TrackedRule(StrictModel):
    name: str
    universe: str
    family: str
    genome: Genome


class TrackConfig(StrictModel):
    start_date: date
    history_start: date
    out_dir: Path
    dataset_dir: Path
    cash_symbol: str
    cost_bps: float = 2.0
    universes: dict[str, list[str]]
    rules: list[TrackedRule]

    def symbols(self) -> list[str]:
        return sorted({s for syms in self.universes.values() for s in syms} | {self.cash_symbol})


def fingerprint(cfg: TrackConfig) -> str:
    """Hash of everything that decides the results: start, history, costs, universes and rules."""
    frozen = cfg.model_dump(mode="json", exclude={"out_dir", "dataset_dir"})
    return hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()


def check_frozen(cfg: TrackConfig) -> dict[str, Any]:
    """Record the fingerprint on the first run; refuse any later run whose rules differ."""
    path = cfg.out_dir / "frozen.json"
    fp = fingerprint(cfg)
    if path.exists():
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec["fingerprint"] != fp:
            raise ValueError(f"the tracked rules changed since {rec['frozen_at']} (fingerprint {rec['fingerprint'][:12]} "
                             f"-> {fp[:12]}); a changed rule is a new experiment: use a new start_date and out_dir")
        return rec
    for r in cfg.rules:
        if r.family not in FAMILIES or r.universe not in cfg.universes:
            raise ValueError(f"unknown family or universe in rule {r.name!r}")
    rec = {"fingerprint": fp, "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
           "start_date": str(cfg.start_date), "rules": [r.model_dump(mode="json") for r in cfg.rules]}
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def last_complete_day(now: datetime) -> date:
    """The latest market date whose daily bar is final at ``now``: today after the close, else yesterday."""
    et = now.astimezone(MARKET_TZ)
    return et.date() if et.time() >= BAR_FINAL else date.fromordinal(et.date().toordinal() - 1)


def _cash_returns(p: Panel, cash_symbol: str) -> np.ndarray:
    c = p.select([cash_symbol]).close[:, 0]
    r = np.zeros_like(c)
    r[1:] = c[1:] / c[:-1] - 1
    return r


def score(cfg: TrackConfig, panels: dict[str, Panel]) -> dict[str, Any]:
    """Every rule and each universe's Hold, from ``start_date`` to the last day of that universe's panel
    (``panels[universe]`` holds the universe's symbols plus the cash symbol)."""
    out: dict[str, Any] = {"start_date": str(cfg.start_date), "universes": {}}
    for uni, syms in cfg.universes.items():
        p = panels[uni].select(syms)
        cash = _cash_returns(panels[uni], cfg.cash_symbol)
        s = next((i for i, d in enumerate(p.dates) if d >= cfg.start_date), len(p.dates))
        rows: dict[str, Any] = {}
        entries = [("Hold", *HOLD)] + [(r.name, r.family, r.genome) for r in cfg.rules if r.universe == uni]
        for name, fam, g in entries:
            x = exposures(p, fam, g)
            days = len(p.dates) - s
            res = simulate(p, x, s, len(p.dates), cfg.cost_bps, cash) if days else None
            last = x[1][-1] if days else np.zeros(len(syms))  # held overnight after the last close
            held = {p.symbols[j]: round(float(last[j]) / len(syms), 4) for j in np.flatnonzero(last > 1e-9)}
            rows[name] = {"days": days, "family": fam, "holding_last_day": held,
                          **({"total_return": float(np.prod(1 + res.returns) - 1),
                              **summary(res.returns, cash[s:]), "daily": [float(v) for v in res.returns]}
                             if res is not None else {})}
        out["universes"][uni] = {"symbols": syms, "first_day": str(p.dates[s]) if s < len(p.dates) else None,
                                 "last_day": str(p.dates[-1]), "rows": rows}
    return out


def render(cfg: TrackConfig, res: dict[str, Any], frozen: dict[str, Any]) -> str:
    L = ["# Swing forward tracker", "",
         f"Rules frozen {frozen['frozen_at'][:10]} (fingerprint `{frozen['fingerprint'][:12]}`); scored from "
         f"{cfg.start_date}, at {cfg.cost_bps:g} bps per trade; idle cash earns {cfg.cash_symbol}. Each position is "
         "valued as if sold at the last close (one exit cost charged).", "",
         "> A few weeks or months of results say almost nothing: a real edge needs about a year or more of days "
         "to show up above luck. Annual figures and Sharpe are shown only after 60 days.", ""]
    for uni, u in res["universes"].items():
        L += [f"## {uni} ({len(u['symbols'])} symbols)", ""]
        if u["first_day"] is None:
            L += [f"No completed days since {cfg.start_date} yet (data through {u['last_day']}).", ""]
            continue
        L += [f"Days {u['first_day']} to {u['last_day']}.", "",
              "| Strategy | Days | Return | vs Hold | Return/yr | Sharpe | Max DD | Holding after the last close |",
              "|---|---|---|---|---|---|---|---|"]
        hold = u["rows"]["Hold"]
        for name, r in u["rows"].items():
            long = r["days"] >= 60
            held = ", ".join(f"{s} {w:.0%}" for s, w in sorted(r["holding_last_day"].items(), key=lambda kv: -kv[1]))
            held = "all symbols, equal weight" if name == "Hold" else (held or "cash")
            L.append(f"| {name} | {r['days']} | {r['total_return']:+.2%} | "
                     f"{'' if name == 'Hold' else f'{r['total_return'] - hold['total_return']:+.2%}'} | "
                     f"{f'{r['ann_return']:+.1%}' if long else '-'} | {f'{r['sharpe']:.2f}' if long else '-'} | "
                     f"{r['max_drawdown']:.1%} | {held} |")
        L.append("")
    L += ["Research only: no order is placed anywhere. Holdings are what each rule would hold; they are not advice."]
    return "\n".join(L) + "\n"


def track(cfg: TrackConfig) -> Path:
    frozen = check_frozen(cfg)
    panels = {u: load_panel(cfg.dataset_dir, [*syms, cfg.cash_symbol]) for u, syms in cfg.universes.items()}
    res = score(cfg, panels)
    res["fingerprint"] = frozen["fingerprint"]
    res["generated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "tracker.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (cfg.out_dir / "tracker.md").write_text(render(cfg, res, frozen), encoding="utf-8")
    return cfg.out_dir / "tracker.md"
