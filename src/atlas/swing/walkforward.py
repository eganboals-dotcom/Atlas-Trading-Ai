"""Walk-forward evolution of swing bots on daily bars, scored against buy-and-hold on unseen days.

Rolling folds: evolve each family on ``train_days`` (fitness: Sharpe at base costs, for bots that are in the
market at least ``min_exposure`` of the time and trade at least ``min_trades_per_year``), skip ``embargo_days``,
score the winners on the next ``validation_days``. Out-of-sample days are stitched across folds. Everything
from ``holdout_start`` on is cut off before any computation, so it stays untouched for one final test.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from atlas.agents.genome import Genome, crossover, genome_hash, mutate_genome, sample_genome
from atlas.config import StrictModel
from atlas.swing.backtest import Result, block_bootstrap, run, sharpe, summary
from atlas.swing.data import DEFAULT_DIR, Panel, load_panel
from atlas.swing.families import FAMILIES, HOLD, exposures


class SwingConfig(StrictModel):
    experiment_name: str
    dataset_dir: Path = DEFAULT_DIR
    symbols: list[str]
    holdout_start: date
    warmup_days: int = 260  # longest lookback (250) plus a little
    train_days: int = 504
    embargo_days: int = 5
    validation_days: int = 126
    step_days: int = 126
    families: list[str] = Field(default_factory=lambda: list(FAMILIES))
    population: int = 30
    generations: int = 12
    elite: int = 4
    mutation_scale: float = 0.15
    mutation_rate: float = 0.4
    crossover_rate: float = 0.5
    seed: int = 7
    costs_bps: dict[str, float] = Field(default_factory=lambda: {"optimistic": 1.0, "base": 2.0, "pessimistic": 5.0})
    bootstrap_samples: int = 2000
    block_days: int = 20
    # A bot that barely trades can post a high Sharpe from a few lucky trades; such bots can't win.
    min_exposure: float = 0.25
    min_trades_per_year: float = 4.0


def fitness(r: Result, cfg: SwingConfig) -> float:
    """Sharpe, for bots that pass the activity gate; below it, a score under every passing bot's that
    still ranks more active bots higher, so evolution is pushed toward the gate."""
    per_year = r.trades / max(1, len(r.returns)) * 252
    if r.exposure >= cfg.min_exposure and per_year >= cfg.min_trades_per_year:
        return sharpe(r.returns)
    return -100.0 + min(r.exposure / cfg.min_exposure, 1.0) + min(per_year / cfg.min_trades_per_year, 1.0)


def folds(n_days: int, cfg: SwingConfig) -> list[tuple[int, int, int, int]]:
    out, s = [], cfg.warmup_days
    while True:
        v0 = s + cfg.train_days + cfg.embargo_days
        if v0 + cfg.validation_days > n_days:
            return out
        out.append((s, s + cfg.train_days, v0, v0 + cfg.validation_days))
        s += cfg.step_days


def evolve_family(p: Panel, family: str, start: int, end: int, cfg: SwingConfig, rng: np.random.Generator,
                  ) -> tuple[list[tuple[float, Genome]], int]:
    """Best genomes of one family on [start, end), by ``fitness`` at base costs; also the number of genomes tried."""
    space = FAMILIES[family].space
    cost = cfg.costs_bps["base"]
    seen: dict[str, tuple[float, Genome]] = {}

    def fit(g: Genome) -> float:
        h = genome_hash(g)
        if h not in seen:
            r = run(p, *exposures(p, family, g), start, end, cost)
            seen[h] = (fitness(r, cfg), g)
        return seen[h][0]

    pop = [sample_genome(space, rng) for _ in range(cfg.population)]
    for _ in range(cfg.generations):
        ranked = sorted(pop, key=fit, reverse=True)

        def pick() -> Genome:
            a, b, c = (ranked[int(i)] for i in rng.integers(len(ranked), size=3))
            return max((a, b, c), key=fit)

        nxt = ranked[:cfg.elite]
        while len(nxt) < cfg.population:
            child = crossover(pick(), pick(), rng) if rng.random() < cfg.crossover_rate else dict(pick())
            nxt.append(mutate_genome(child, space, rng, cfg.mutation_scale, cfg.mutation_rate)[0])
        pop = nxt
    for g in pop:
        fit(g)
    return sorted(seen.values(), key=lambda x: x[0], reverse=True), len(seen)


def walkforward(cfg: SwingConfig, out_root: Path = Path("artifacts/reports/swing")) -> Path:
    full = load_panel(cfg.dataset_dir, cfg.symbols)
    p = full.before(cfg.holdout_start)  # the holdout is never seen below
    holdout_days = len(full.dates) - len(p.dates)
    rng = np.random.default_rng(cfg.seed)
    hold_x = exposures(p, *HOLD)
    plan = folds(len(p.dates), cfg)
    if not plan:
        raise ValueError("not enough days for one fold")
    oos: dict[str, dict[str, list[np.ndarray]]] = {}  # strategy -> scenario -> per-fold returns
    fold_rows, trials = [], 0
    for k, (t0, t1, v0, v1) in enumerate(plan):
        best: dict[str, tuple[float, Genome]] = {}
        for fam in cfg.families:
            ranked, n = evolve_family(p, fam, t0, t1, cfg, rng)
            best[fam] = ranked[0]
            trials += n
        champ = max(best, key=lambda f: best[f][0])
        hold_is = sharpe(run(p, *hold_x, t0, t1, cfg.costs_bps["base"]).returns)
        row = {"fold": k, "train": [str(p.dates[t0]), str(p.dates[t1 - 1])],
               "validation": [str(p.dates[v0]), str(p.dates[v1 - 1])], "champion_family": champ,
               "champion_genome": best[champ][1], "in_sample_sharpe": best[champ][0], "hold_in_sample_sharpe": hold_is,
               "family_best": {f: {"in_sample_sharpe": s, "genome": g} for f, (s, g) in best.items()}, "oos": {}}
        strategies = {"champion": (champ, best[champ][1]), "hold": HOLD,
                      **{f"family:{f}": (f, g) for f, (_, g) in best.items()}}
        for name, (fam, g) in strategies.items():
            x = hold_x if name == "hold" else exposures(p, fam, g)
            for sc, cost in cfg.costs_bps.items():
                r = run(p, *x, v0, v1, cost)
                oos.setdefault(name, {}).setdefault(sc, []).append(r.returns)
                if sc == "base":
                    row["oos"][name] = {**summary(r.returns), "exposure": r.exposure, "trades": r.trades}
        fold_rows.append(row)
        print(f"fold {k}: {row['validation'][0]}..{row['validation'][1]} champion {champ} "
              f"IS Sharpe {best[champ][0]:.2f} | OOS champion {row['oos']['champion']['ann_return']:+.1%} "
              f"hold {row['oos']['hold']['ann_return']:+.1%}", flush=True)

    stitched = {n: {sc: np.concatenate(v) for sc, v in d.items()} for n, d in oos.items()}
    headline = {}
    for sc in cfg.costs_bps:
        a, h = stitched["champion"][sc], stitched["hold"][sc]
        ci_sharpe = block_bootstrap([a, h], lambda x, y: sharpe(x) - sharpe(y), cfg.bootstrap_samples,
                                    cfg.block_days, cfg.seed)
        ci_excess = block_bootstrap([a - h], lambda d: float(d.mean() * 252), cfg.bootstrap_samples,
                                    cfg.block_days, cfg.seed)
        headline[sc] = {"champion": summary(a), "hold": summary(h), "sharpe_diff": sharpe(a) - sharpe(h),
                        "sharpe_diff_ci90": ci_sharpe, "excess_return_ann": float((a - h).mean() * 252),
                        "excess_return_ci90": ci_excess}
    families = {n.split(":", 1)[1]: summary(v["base"]) for n, v in stitched.items() if n.startswith("family:")}
    run_id = f"{cfg.experiment_name}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    out = out_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    report = {"run_id": run_id, "config": json.loads(cfg.model_dump_json()), "symbols": list(p.symbols),
              "development_days": [str(p.dates[0]), str(p.dates[-1])], "holdout_days": holdout_days,
              "holdout_status": "untouched by this run", "trials": trials, "headline": headline,
              "families_base": families, "folds": fold_rows,
              "selection": {"mean_in_sample_sharpe": float(np.mean([r["in_sample_sharpe"] for r in fold_rows])),
                            "mean_oos_sharpe": float(np.mean([r["oos"]["champion"]["sharpe"] for r in fold_rows]))}}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    (out / "report.md").write_text(render(report), encoding="utf-8")
    return out


def _pct(x: float) -> str:
    return f"{x:+.1%}"


def render(r: dict[str, Any]) -> str:
    L = [f"# Swing walk-forward: {r['run_id']}", "",
         "> Research evidence under simple cost assumptions, not a promise of profit. Fills are assumed at the "
         "official open and close prices; costs are a flat bps charge per unit of exposure changed.", "",
         f"Symbols ({len(r['symbols'])}, equal weight): {', '.join(r['symbols'])}  ",
         f"Development days: {r['development_days'][0]} to {r['development_days'][1]}; "
         f"holdout: last {r['holdout_days']} days, {r['holdout_status']}.  ",
         f"Genomes evaluated: {r['trials']:,} across {len(r['folds'])} folds.", "",
         "## Out-of-sample: fold champions vs Hold (stitched unseen days)", "",
         "| Costs | Champion return/yr | Champion Sharpe | Champion max DD | Hold return/yr | Hold Sharpe | Hold max DD "
         "| Sharpe diff [90% CI] | Excess return/yr [90% CI] |", "|---|---|---|---|---|---|---|---|---|"]
    for sc, h in r["headline"].items():
        c, b = h["champion"], h["hold"]
        lo, hi = h["sharpe_diff_ci90"]
        elo, ehi = h["excess_return_ci90"]
        L.append(f"| {sc} | {_pct(c['ann_return'])} | {c['sharpe']:.2f} | {c['max_drawdown']:.1%} | "
                 f"{_pct(b['ann_return'])} | {b['sharpe']:.2f} | {b['max_drawdown']:.1%} | "
                 f"{h['sharpe_diff']:+.2f} [{lo:+.2f}, {hi:+.2f}] | {_pct(h['excess_return_ann'])} [{_pct(elo)}, {_pct(ehi)}] |")
    s = r["selection"]
    L += ["", f"Selection: mean in-sample Sharpe of champions {s['mean_in_sample_sharpe']:.2f}, "
              f"mean out-of-sample {s['mean_oos_sharpe']:.2f}.", "",
          "## Each family's best, out of sample (base costs)", "",
          "| Family | Return/yr | Sharpe | Max DD |", "|---|---|---|---|"]
    for f, m in sorted(r["families_base"].items(), key=lambda kv: -kv[1]["sharpe"]):
        L.append(f"| {f} | {_pct(m['ann_return'])} | {m['sharpe']:.2f} | {m['max_drawdown']:.1%} |")
    L += ["", "## Folds (base costs)", "",
          "| Fold | Validation | Champion | Mode | IS Sharpe | OOS return/yr | OOS Sharpe | Hold OOS return/yr | Exposure | Trades/sym |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    mode_name = {0.0: "held", 1.0: "overnight", 2.0: "day"}
    for f in r["folds"]:
        o, h, g = f["oos"]["champion"], f["oos"]["hold"], f["champion_genome"]
        mode = mode_name.get(g.get("mode", 2.0), "?")
        L.append(f"| {f['fold']} | {f['validation'][0]}..{f['validation'][1]} | {f['champion_family']} | {mode} | "
                 f"{f['in_sample_sharpe']:.2f} | {_pct(o['ann_return'])} | {o['sharpe']:.2f} | {_pct(h['ann_return'])} | "
                 f"{o['exposure']:.0%} | {o['trades']:.0f} |")
    L += ["", "## Limitations", "",
          "* Prices are split- and dividend-adjusted daily SIP bars; fills at the official open/close with no auction "
          "imbalance or size limits.",
          "* Decisions are made at the open only, from closes up to the day before and the day's open.",
          "* Long-only, no leverage, equal weight across symbols.",
          "* The symbol list was chosen today; all of them still trade, so there is survivorship in the universe.",
          "* Daily returns are not independent; the block bootstrap intervals are approximate."]
    return "\n".join(L) + "\n"
