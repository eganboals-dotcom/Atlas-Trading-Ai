"""Walk-forward evolution of swing bots on daily bars, scored against buy-and-hold on unseen days.

Rolling folds: evolve each family on ``train_days`` (fitness: Sharpe at base costs, for bots that are in the
market at least ``min_exposure`` of the time and trade at least ``min_trades_per_year``), skip ``embargo_days``,
score the winners on the next ``validation_days``. Out-of-sample days are stitched across folds. Everything
from ``holdout_start`` on is cut off before any computation, so it stays untouched for one final test.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from atlas.agents.genome import Genome, crossover, genome_hash, mutate_genome, sample_genome
from atlas.config import StrictModel
from atlas.swing.backtest import Result, block_bootstrap, sharpe, simulate, summary
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
    # Daily return earned by capital out of the market (a T-bill ETF such as BIL); None leaves idle cash at 0%.
    cash_symbol: str | None = None
    # Alpaca news headlines (``fetch-news``) for the news families; None runs without news.
    news_dir: Path | None = None


def fitness(r: Result, cfg: SwingConfig, rf: np.ndarray | None = None) -> float:
    """Sharpe (of the return above the cash rate ``rf``), for bots that pass the activity gate; below it, a score under every passing bot's that
    still ranks more active bots higher, so evolution is pushed toward the gate."""
    per_year = r.trades / max(1, len(r.returns)) * 252
    if r.exposure >= cfg.min_exposure and per_year >= cfg.min_trades_per_year:
        return sharpe(r.returns if rf is None else r.returns - rf)
    return -100.0 + min(r.exposure / cfg.min_exposure, 1.0) + min(per_year / cfg.min_trades_per_year, 1.0)


# Published rules, run exactly as fixed here and never tuned, so they carry no selection bias of ours.
TEXTBOOK: dict[str, tuple[str, Genome]] = {
    "200-day trend filter": ("session", {"mode": 0.0, "trend_filter": 1.0, "trend_n": 200}),
    "dual momentum top 3 (12-1 months)": ("rotation", {"mode": 0.0, "lookback": 231, "skip": 21, "top_k": 3,
                                                       "rebalance": 21, "abs_filter": 1.0}),
    "sector momentum top 3 (6 months)": ("rotation", {"mode": 0.0, "lookback": 126, "skip": 0, "top_k": 3,
                                                      "rebalance": 21, "abs_filter": 0.0}),
    "overnight only": ("session", {"mode": 1.0, "trend_filter": 0.0, "trend_n": 200}),
    "turn of the month (-1..+3)": ("calendar", {"mode": 0.0, "trend_filter": 0.0, "trend_n": 200, "before": 1,
                                                "after": 3}),
    "volatility target 15%": ("vol_target", {"mode": 0.0, "trend_filter": 0.0, "trend_n": 200, "vol_n": 20,
                                             "target_vol": 0.15}),
    # v4: fixed combinations, set once here and never tuned (they are ours, not published as such)
    "dual momentum + 200-day trend + 15% vol cap": ("risk_rotation", {
        "lookback": 231, "skip": 21, "top_k": 3, "rebalance": 21, "abs_filter": 1.0, "trend_filter": 1.0,
        "trend_n": 200, "vol_scale": 1.0, "vol_n": 20, "target_vol": 0.15, "overlay_every": 5.0}),
    "day trade +1% target / -1% stop": ("bracket", {"trend_filter": 0.0, "trend_n": 200, "entry": 0.0,
                                                    "drop_bps": 100.0, "tp_bps": 100.0, "sl_bps": 100.0,
                                                    "max_days": 1}),
    "swing +10% target / -5% stop, above 200-day trend, up to 60 days": ("wide_bracket", {
        "trend_filter": 1.0, "trend_n": 200, "entry": 0.0, "drop_bps": 100.0, "tp_bps": 1000.0, "sl_bps": 500.0,
        "max_days": 60}),
    "swing +5% target / -3% stop, up to 20 days": ("wide_bracket", {
        "trend_filter": 0.0, "trend_n": 200, "entry": 0.0, "drop_bps": 100.0, "tp_bps": 500.0, "sl_bps": 300.0,
        "max_days": 20}),
    "post-event drift (2% gap, 2x volume, 20 days)": ("event_drift", {
        "mode": 0.0, "trend_filter": 0.0, "trend_n": 200, "direction": 1.0, "gap_bps": 200.0, "vol_mult": 2.0,
        "confirm": 0.0, "hold_days": 20}),
}
NEWS_TEXTBOOK: dict[str, tuple[str, Genome]] = {  # added to TEXTBOOK when a run has news
    "news gaps: follow with news, fade without (1% gap, 5 days)": ("news_gap", {
        "trend_filter": 0.0, "trend_n": 200, "rule": 2.0, "gap_bps": 100.0, "min_articles": 1, "hold_days": 5}),
    "post-earnings drift (gap up on earnings, enter next day, 60 days)": ("earnings_drift", {
        "trend_filter": 0.0, "trend_n": 200, "direction": 1.0, "gap_bps": 0.0, "delay": 1.0, "hold_days": 60}),
    "analyst upgrade drift (buy on upgrades, 20 days)": ("news_tone", {
        "trend_filter": 0.0, "trend_n": 200, "source": 0.0, "side": 1.0, "delay": 0.0, "hold_days": 20}),
    "earnings surprise drift (EPS beat, enter next day, 60 days)": ("news_tone", {
        "trend_filter": 0.0, "trend_n": 200, "source": 1.0, "side": 1.0, "delay": 1.0, "hold_days": 60}),
}


def folds(n_days: int, cfg: SwingConfig) -> list[tuple[int, int, int, int]]:
    out, s = [], cfg.warmup_days
    while True:
        v0 = s + cfg.train_days + cfg.embargo_days
        if v0 + cfg.validation_days > n_days:
            return out
        out.append((s, s + cfg.train_days, v0, v0 + cfg.validation_days))
        s += cfg.step_days


def evolve_family(p: Panel, family: str, start: int, end: int, cfg: SwingConfig, rng: np.random.Generator,
                  cash: np.ndarray | None = None) -> tuple[list[tuple[float, Genome]], int]:
    """Best genomes of one family on [start, end), by ``fitness`` at base costs; also the number of genomes tried."""
    space = FAMILIES[family].space
    cost = cfg.costs_bps["base"]
    seen: dict[str, tuple[float, Genome]] = {}

    def fit(g: Genome) -> float:
        h = genome_hash(g)
        if h not in seen:
            r = simulate(p, exposures(p, family, g), start, end, cost, cash)
            seen[h] = (fitness(r, cfg, None if cash is None else cash[start:end]), g)
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
    syms = cfg.symbols + ([cfg.cash_symbol] if cfg.cash_symbol else [])
    full = load_panel(cfg.dataset_dir, syms).before(cfg.holdout_start)  # the holdout is never seen below
    holdout_days = len(load_panel(cfg.dataset_dir, syms).dates) - len(full.dates)
    p = full.select(cfg.symbols)
    rules = dict(TEXTBOOK)
    if cfg.news_dir:
        from atlas.swing.news import load_news, news_features

        p = dataclasses.replace(p, features=news_features(p, load_news(cfg.news_dir)))  # days before the holdout only
        rules |= NEWS_TEXTBOOK
    cash = np.zeros(len(p.dates))  # no interest unless a cash symbol is configured
    if cfg.cash_symbol:
        c = full.select([cfg.cash_symbol]).close[:, 0]
        cash = np.zeros_like(c)
        cash[1:] = c[1:] / c[:-1] - 1
    rng = np.random.default_rng(cfg.seed)
    hold_x = exposures(p, *HOLD)
    plan = folds(len(p.dates), cfg)
    if not plan:
        raise ValueError("not enough days for one fold")
    oos: dict[str, dict[str, list[np.ndarray]]] = {}  # strategy -> scenario -> per-fold returns
    fixed = {f"rule:{name}": (fam, g) for name, (fam, g) in rules.items()}  # never tuned
    fixed_x = {name: exposures(p, fam, g) for name, (fam, g) in fixed.items()}
    fold_rows, trials = [], 0
    for k, (t0, t1, v0, v1) in enumerate(plan):
        best: dict[str, tuple[float, Genome]] = {}
        for fam in cfg.families:
            ranked, n = evolve_family(p, fam, t0, t1, cfg, rng, cash)
            best[fam] = ranked[0]
            trials += n
        champ = max(best, key=lambda f: best[f][0])
        passing = [f for f, (fit_, _) in best.items() if fit_ > -50] or [champ]  # cleared the gate in training
        hold_is = sharpe(simulate(p, hold_x, t0, t1, cfg.costs_bps["base"], cash).returns - cash[t0:t1])
        row = {"fold": k, "train": [str(p.dates[t0]), str(p.dates[t1 - 1])],
               "validation": [str(p.dates[v0]), str(p.dates[v1 - 1])], "champion_family": champ,
               "champion_genome": best[champ][1], "in_sample_sharpe": best[champ][0], "hold_in_sample_sharpe": hold_is,
               "ensemble_families": passing,
               "family_best": {f: {"in_sample_sharpe": s, "genome": g} for f, (s, g) in best.items()}, "oos": {}}
        xs = {"champion": exposures(p, champ, best[champ][1]), "hold": hold_x,
              **{f"family:{f}": exposures(p, f, g) for f, (_, g) in best.items()}, **fixed_x}
        for sc, cost in cfg.costs_bps.items():
            rets = {}
            for name, x in xs.items():
                r = simulate(p, x, v0, v1, cost, cash)
                rets[name] = r.returns
                if sc == "base":
                    row["oos"][name] = {**summary(r.returns, cash[v0:v1]), "exposure": r.exposure, "trades": r.trades}
            # equal capital in each family's best bot that cleared the gate: no pick between families at all
            rets["ensemble"] = np.mean([rets[f"family:{f}"] for f in passing], axis=0)
            if sc == "base":
                row["oos"]["ensemble"] = summary(rets["ensemble"], cash[v0:v1])
            for name, r in rets.items():
                oos.setdefault(name, {}).setdefault(sc, []).append(r)
        fold_rows.append(row)
        print(f"fold {k}: {row['validation'][0]}..{row['validation'][1]} champion {champ} "
              f"IS Sharpe {best[champ][0]:.2f} | OOS champion {row['oos']['champion']['ann_return']:+.1%} "
              f"ensemble {row['oos']['ensemble']['ann_return']:+.1%} hold {row['oos']['hold']['ann_return']:+.1%}",
              flush=True)

    stitched = {n: {sc: np.concatenate(v) for sc, v in d.items()} for n, d in oos.items()}
    rf = np.concatenate([cash[v0:v1] for _, _, v0, v1 in plan])  # Sharpe ratios below are of returns above it
    headline: dict[str, dict[str, Any]] = {}
    for strat in ("champion", "ensemble"):
        for sc in cfg.costs_bps:
            a, h = stitched[strat][sc], stitched["hold"][sc]
            ci_sharpe = block_bootstrap([a, h, rf], lambda x, y, z: sharpe(x - z) - sharpe(y - z),
                                        cfg.bootstrap_samples, cfg.block_days, cfg.seed)
            ci_excess = block_bootstrap([a - h], lambda d: float(d.mean() * 252), cfg.bootstrap_samples,
                                        cfg.block_days, cfg.seed)
            headline.setdefault(strat, {})[sc] = {
                strat: summary(a, rf), "hold": summary(h, rf), "sharpe_diff": sharpe(a - rf) - sharpe(h - rf),
                "sharpe_diff_ci90": ci_sharpe, "excess_return_ann": float((a - h).mean() * 252),
                "excess_return_ci90": ci_excess}
    hold_base = summary(stitched["hold"]["base"], rf)

    def passes(m: dict[str, float]) -> bool:
        return m["sharpe"] > hold_base["sharpe"] and m["ann_return"] >= hold_base["ann_return"]

    others = {n: {sc: summary(v, rf) for sc, v in d.items()} for n, d in stitched.items()
              if n.startswith(("family:", "rule:"))}
    verdict = {n: passes(summary(stitched[n]["base"], rf)) for n in stitched if n != "hold"}
    run_id = f"{cfg.experiment_name}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    out = out_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    report = {"run_id": run_id, "config": json.loads(cfg.model_dump_json()), "symbols": list(p.symbols),
              "development_days": [str(p.dates[0]), str(p.dates[-1])], "holdout_days": holdout_days,
              "holdout_status": "untouched by this run", "trials": trials, "headline": headline,
              "hold_base": hold_base, "others": others, "passes": verdict,
              "pass_rule": "base costs, stitched unseen days: Sharpe above Hold's and annual return at least Hold's",
              "textbook": {n: {"family": f, "genome": g} for n, (f, g) in rules.items()}, "folds": fold_rows,
              "selection": {"mean_in_sample_sharpe": float(np.mean([r["in_sample_sharpe"] for r in fold_rows])),
                            "mean_oos_sharpe": float(np.mean([r["oos"]["champion"]["sharpe"] for r in fold_rows]))}}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    (out / "report.md").write_text(render(report), encoding="utf-8")
    return out


def _pct(x: float) -> str:
    return f"{x:+.1%}"


def render(r: dict[str, Any]) -> str:
    hb = r["hold_base"]
    L = [f"# Swing walk-forward: {r['run_id']}", "",
         "> Research evidence under simple cost assumptions, not a promise of profit. Fills are assumed at the "
         "official open and close prices; costs are a flat bps charge per unit of exposure changed.", "",
         f"Symbols ({len(r['symbols'])}, equal weight): {', '.join(r['symbols'])}  ",
         f"Development days: {r['development_days'][0]} to {r['development_days'][1]}; "
         f"holdout: last {r['holdout_days']} days, {r['holdout_status']}.  ",
         f"Genomes evaluated: {r['trials']:,} across {len(r['folds'])} folds.  ",
         f"Pass rule (fixed before the run): {r['pass_rule']}.", "",
         f"Hold on the same unseen days (base costs): {_pct(hb['ann_return'])}/yr, Sharpe {hb['sharpe']:.2f}, "
         f"max drawdown {hb['max_drawdown']:.1%}.", ""]
    titles = {"champion": "Fold champions (best family in training)",
              "ensemble": "Ensemble (equal capital in every family's best bot)"}
    for strat, rows in r["headline"].items():
        L += [f"## {titles[strat]} vs Hold, stitched unseen days", "",
              "| Costs (bps/side) | Return/yr | Sharpe | Max DD | Hold return/yr | Hold Sharpe | Sharpe diff [90% CI] "
              "| Excess return/yr [90% CI] |", "|---|---|---|---|---|---|---|---|"]
        for sc, h in rows.items():
            c, b = h[strat], h["hold"]
            lo, hi = h["sharpe_diff_ci90"]
            elo, ehi = h["excess_return_ci90"]
            L.append(f"| {sc} ({r['config']['costs_bps'][sc]:g}) | {_pct(c['ann_return'])} | {c['sharpe']:.2f} | "
                     f"{c['max_drawdown']:.1%} | {_pct(b['ann_return'])} | {b['sharpe']:.2f} | "
                     f"{h['sharpe_diff']:+.2f} [{lo:+.2f}, {hi:+.2f}] | {_pct(h['excess_return_ann'])} "
                     f"[{_pct(elo)}, {_pct(ehi)}] |")
        L.append("")
    s = r["selection"]
    L += [f"Selection: mean in-sample Sharpe of champions {s['mean_in_sample_sharpe']:.2f}, "
          f"mean out-of-sample {s['mean_oos_sharpe']:.2f}.", "",
          "## Every strategy on the unseen days", "",
          "Family rows are each family's best bot per fold; rule rows are published rules, never tuned. "
          "Many rows means many chances for one to look good by luck.", "",
          "| Strategy | Return/yr (base) | Sharpe (base) | Max DD | Return/yr (pessimistic) | Passes |",
          "|---|---|---|---|---|---|"]
    rows = sorted(r["others"].items(), key=lambda kv: -kv[1]["base"]["sharpe"])
    for name, m in rows:
        b, pe = m["base"], m["pessimistic"]
        L.append(f"| {name} | {_pct(b['ann_return'])} | {b['sharpe']:.2f} | {b['max_drawdown']:.1%} | "
                 f"{_pct(pe['ann_return'])} | {'yes' if r['passes'][name] else 'no'} |")
    L += ["", "## Folds (base costs)", "",
          "| Fold | Validation | Champion | Mode | IS Sharpe | OOS return/yr | OOS Sharpe | Ensemble OOS | Hold OOS |",
          "|---|---|---|---|---|---|---|---|---|"]
    mode_name = {0.0: "held", 1.0: "overnight", 2.0: "day"}
    for f in r["folds"]:
        o, h, e, g = f["oos"]["champion"], f["oos"]["hold"], f["oos"]["ensemble"], f["champion_genome"]
        fam = FAMILIES[f["champion_family"]]
        mode = "legs" if fam.legs else mode_name[g.get("mode", fam.fixed_mode)]
        L.append(f"| {f['fold']} | {f['validation'][0]}..{f['validation'][1]} | {f['champion_family']} | {mode} | "
                 f"{f['in_sample_sharpe']:.2f} | {_pct(o['ann_return'])} | {o['sharpe']:.2f} | "
                 f"{_pct(e['ann_return'])} | {_pct(h['ann_return'])} |")
    L += ["", "## Limitations", "",
          "* Prices are split- and dividend-adjusted daily SIP bars; fills at the official open/close with no auction "
          "imbalance or size limits.",
          "* Decisions are made at the open only, from closes up to the day before and the day's open.",
          "* Long-only, no leverage; Hold is equal weight across symbols, rebalanced daily.",
          (f"* Capital out of the market earns {r['config']['cash_symbol']}'s daily return (T-bills); Hold is always "
           "invested. Sharpe ratios are of the return above that cash rate, so sitting in cash doesn't raise them."
           if r["config"].get("cash_symbol") else "* Capital out of the market earns nothing."),
          "* Bracket bots fill stops and targets exactly at their levels (no slippage past a stop beyond the cost), "
          "and a day touching both is read from its shape (up day: stop first); this matched 1-minute paths "
          "within ~2 bps per trade.",
          "* The symbol list was chosen today; all of them still trade, so there is survivorship in the universe.",
          "* Daily returns are not independent; the block bootstrap intervals are approximate."]
    return "\n".join(L) + "\n"
