"""Nested walk-forward evaluation, holdout evaluation, and checkpoint stress tests.

Walk-forward procedure, for each fold (training sessions strictly before validation):

1. Evolve a fresh population for ``generations_per_fold`` generations on the fold's
   training sessions only (inner selection).
2. Take the top ``candidates_per_fold`` archived champions. Re-score them on the
   training sessions to get their in-sample numbers.
3. Score those champions and every fixed baseline on the fold's validation sessions,
   under every cost scenario (out-of-sample).

The outer holdout is never read here. ``evaluate_holdout`` reads it only for a frozen
design, through the guarded ledger.

Every selection trial is counted. The gap between in-sample and out-of-sample returns
is reported as a direct estimate of selection bias.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from atlas import __version__
from atlas.agents.base import AgentRecord
from atlas.agents.baselines import baseline_records
from atlas.config import DataConfig, EvolutionConfig, SimulatorConfig
from atlas.data.events import Quote
from atlas.data.providers import (
    DataSourceConfig, MarketDataProvider, SessionRef, load_session, session_validation,
)
from atlas.evaluation.holdout import HoldoutGuard, HoldoutLedger
from atlas.evaluation.metrics import latency_percentiles
from atlas.evaluation.scenarios import DEFAULT_SCENARIOS, EMPHASIZED, CostScenario, apply_scenario
from atlas.evaluation.splits import SplitConfig, SplitPlan, plan_splits
from atlas.evolution.evaluation import AgentEvaluation, evaluate_records
from atlas.evolution.runner import EvolutionRunner
from atlas.learning.settings import PPOSettings
from atlas.logging_utils import get_logger
from atlas.timeutils import NS_PER_S

log = get_logger(__name__)

DISCLAIMER = (
    "Out-of-sample results under this simulator's execution assumptions. They are research evidence, "
    "not a promise of profit, and say nothing about live fills. Synthetic data carries no real-market "
    "information."
)


class BootstrapConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    samples: int = Field(default=2_000, ge=100)
    seed: int = 7
    confidence: float = Field(default=0.90, gt=0, lt=1)


class WalkForwardConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    evolution_config: Path
    data_source: DataSourceConfig
    splits: SplitConfig
    generations_per_fold: int = Field(default=4, ge=1)
    candidates_per_fold: int = Field(default=3, ge=1)
    cost_scenarios: dict[str, CostScenario] = Field(default_factory=lambda: dict(DEFAULT_SCENARIOS))
    bootstrap: BootstrapConfig = BootstrapConfig()
    reports_dir: Path = Path("artifacts/reports/walkforward")
    holdout_ledger: Path = Path("artifacts/holdout_ledger.jsonl")
    ppo: PPOSettings = PPOSettings()
    # Which family supplies the frozen design's final candidate. PPO is never promoted
    # automatically: choosing it requires changing this setting, which changes the design fingerprint.
    final_candidate_source: Literal["evolution", "ppo"] = "evolution"

    @model_validator(mode="after")
    def _scenarios(self) -> WalkForwardConfig:
        missing = [s for s in EMPHASIZED if s not in self.cost_scenarios]
        if missing:
            raise ValueError(f"cost_scenarios must include {missing}")
        if self.final_candidate_source == "ppo" and not self.ppo.enabled:
            raise ValueError("final_candidate_source: ppo requires ppo.enabled: true")
        return self


# ----------------------------------------------------------------- helpers


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def bootstrap_mean_ci(values: list[float], cfg: BootstrapConfig) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "low": None, "high": None, "n": 0}
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(cfg.seed)
    means = arr[rng.integers(0, len(arr), size=(cfg.samples, len(arr)))].mean(axis=1)
    alpha = (1 - cfg.confidence) / 2
    return {"mean": float(arr.mean()), "low": float(np.quantile(means, alpha)),
            "high": float(np.quantile(means, 1 - alpha)), "n": len(values)}


def session_volatility_bps(ref: SessionRef) -> float:
    """Realized mid-price volatility in bps per sqrt(second), averaged across symbols."""
    by_symbol: dict[str, list[tuple[int, float]]] = {}
    for e in load_session(ref):
        if isinstance(e, Quote):
            by_symbol.setdefault(e.symbol, []).append((e.event_ts, e.mid))
    vols = []
    for pts in by_symbol.values():
        if len(pts) < 3:
            continue
        r2 = sum(math.log(b / a) ** 2 for (_, a), (_, b) in zip(pts, pts[1:]))
        span_s = (pts[-1][0] - pts[0][0]) / NS_PER_S
        if span_s > 0:
            vols.append(math.sqrt(r2 / span_s) * 10_000)
    return statistics.fmean(vols) if vols else float("nan")


def summarize(ev: AgentEvaluation) -> dict[str, Any]:
    w = ev.windows

    def mean_of(attr: str) -> float | None:
        vals = [getattr(m, attr) for m in w if getattr(m, attr) is not None]
        return statistics.fmean(vals) if vals else None

    symbol_pnl: dict[str, float] = {}
    for m in w:
        for s, v in m.symbol_pnl.items():
            symbol_pnl[s] = symbol_pnl.get(s, 0.0) + v
    return {
        "sessions": len(w),
        "error": ev.error,
        "violations": len(ev.violations),
        "mean_net_return": mean_of("net_return"),
        "worst_drawdown": max((m.max_drawdown for m in w), default=None),
        "mean_turnover": mean_of("turnover"),
        "sharpe": mean_of("sharpe"),
        "sortino": mean_of("sortino"),
        "profit_factor": mean_of("profit_factor"),
        "win_rate": mean_of("win_rate"),
        "mean_trade_pnl": mean_of("mean_trade_pnl"),
        "fill_rate": mean_of("fill_rate"),
        "avg_exposure": mean_of("avg_exposure"),
        "peak_exposure": max((m.peak_exposure for m in w), default=None),
        "fills": sum(m.fills for m in w),
        "round_trips": sum(m.round_trips for m in w),
        "fees": sum(m.fees for m in w),
        "session_returns": [m.net_return for m in w],
        "symbol_pnl": symbol_pnl,
    }


def scenario_evaluate(
    records: list[AgentRecord],
    refs: list[SessionRef],
    sim: SimulatorConfig,
    evo: EvolutionConfig,
    scenarios: dict[str, CostScenario],
) -> dict[str, dict[str, AgentEvaluation]]:
    out: dict[str, dict[str, AgentEvaluation]] = {}
    for name, sc in scenarios.items():
        evs = evaluate_records(records, refs, apply_scenario(sim, sc), evo.limits,
                               evo.evaluation.decision_interval_ms, evo.evaluation.workers)
        out[name] = {e.agent_id: e for e in evs}
    return out


def data_quality(refs: list[SessionRef]) -> list[dict[str, Any]]:
    return [{"session": r.session_date.isoformat(), "symbols": list(r.symbols),
             **session_validation(r).summary()} for r in refs]


# --------------------------------------------------------------- walk-forward


def design_fingerprint(wf: WalkForwardConfig, evo: EvolutionConfig, sim: SimulatorConfig, base: DataConfig,
                       plan: SplitPlan, final_candidate: AgentRecord | None) -> str:
    return _sha({
        "walkforward": wf.model_dump(mode="json", exclude={"reports_dir", "holdout_ledger"}),
        "evolution": evo.model_dump(mode="json", exclude={"checkpoint_dir", "reports_dir", "generations"}),
        "simulator": sim.model_dump(mode="json", exclude={"reports_dir", "demo_strategy"}),
        "data": base.model_dump(mode="json"),
        "plan": plan.to_dict(),
        "final_candidate": None if final_candidate is None else final_candidate.to_dict(),
    })


def dataset_id(provider: MarketDataProvider) -> str:
    d = provider.describe()
    return f"{d.get('provider')}:{d.get('dataset') or d.get('source')}:{_sha(d)}"


def run_walkforward(
    wf: WalkForwardConfig,
    evo: EvolutionConfig,
    sim: SimulatorConfig,
    base: DataConfig,
    provider: MarketDataProvider,
) -> tuple[dict[str, Any], Path]:
    plan = plan_splits(provider.sessions(), wf.splits)
    guard = HoldoutGuard(provider, plan.holdout)
    run_id = f"{wf.experiment_name}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    run_dir = wf.reports_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    baselines = baseline_records()
    scenarios = wf.cost_scenarios

    folds_out: list[dict[str, Any]] = []
    total_trials = 0
    final_candidate: AgentRecord | None = None
    for fold in plan.folds:
        train_refs, val_refs = guard.refs(fold.train), guard.refs(fold.validation)
        fold_evo = evo.model_copy(update={
            "experiment_name": f"fold_{fold.index:02d}",
            "seed": evo.seed + fold.index,
            "generations": wf.generations_per_fold,
            "checkpoint_dir": run_dir / "checkpoints",
            "reports_dir": run_dir / "evolution",
        })
        state = EvolutionRunner(fold_evo, sim, base, sessions=train_refs).run()
        total_trials += state.evaluations_run
        candidates = [e.record for e in state.archives.champions[: wf.candidates_per_fold]]
        log.info("fold evolved", extra={"fold": fold.index, "candidates": len(candidates),
                                        "trials": state.evaluations_run})
        fold_out: dict[str, Any] = {
            "index": fold.index,
            "train": [d.isoformat() for d in fold.train],
            "validation": [d.isoformat() for d in fold.validation],
            "evolution_status": state.stopped or "completed",
            "trials": state.evaluations_run,
            "candidates": [],
            "baselines": {},
        }
        ppo_record = None
        if wf.ppo.enabled:
            ppo_record, fold_out["ppo_training"] = _train_fold_ppo(wf, fold.index, train_refs, sim, fold_evo, run_dir)
        if wf.final_candidate_source == "evolution" and candidates:
            final_candidate = candidates[0]
        elif wf.final_candidate_source == "ppo" and ppo_record is not None:
            final_candidate = ppo_record
        learned = [ppo_record] if ppo_record else []
        in_sample = scenario_evaluate(candidates + learned, train_refs, sim, fold_evo,
                                      {"base": scenarios["base"]})["base"] if candidates or learned else {}
        oos = scenario_evaluate(candidates + learned + baselines, val_refs, sim, fold_evo, scenarios)
        if ppo_record is not None:
            fold_out["ppo"] = {
                "agent_id": ppo_record.agent_id, "artifact": ppo_record.artifact,
                "in_sample": summarize(in_sample[ppo_record.agent_id]),
                "out_of_sample": {name: summarize(evs[ppo_record.agent_id]) for name, evs in oos.items()},
                "latency": latency_percentiles([x for w in oos["base"][ppo_record.agent_id].decision_wall_ns for x in w]),
            }
        for rank, c in enumerate(candidates):
            fold_out["candidates"].append({
                "rank": rank, "agent_id": c.agent_id, "family": c.family, "genome": c.genome,
                "in_sample": summarize(in_sample[c.agent_id]),
                "out_of_sample": {name: summarize(evs[c.agent_id]) for name, evs in oos.items()},
                "latency": latency_percentiles(
                    [x for w in oos["base"][c.agent_id].decision_wall_ns for x in w]),
            })
        for b in baselines:
            fold_out["baselines"][b.agent_id] = {name: summarize(evs[b.agent_id]) for name, evs in oos.items()}
        folds_out.append(fold_out)

    report = _aggregate(wf, evo, sim, base, provider, plan, folds_out, total_trials, final_candidate)
    report["run_id"] = run_id
    if wf.ppo.enabled:
        report["ppo"] = _ppo_summary(wf, folds_out)
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    design = {
        "design_fingerprint": report["design_fingerprint"],
        "dataset_id": report["dataset_id"],
        "final_candidate": None if final_candidate is None else final_candidate.to_dict(),
        "walkforward_config": wf.model_dump(mode="json"),
        "evolution_config": evo.model_dump(mode="json"),
        "simulator_config": sim.model_dump(mode="json"),
        "data_config": base.model_dump(mode="json"),
        "plan": plan.to_dict(),
    }
    (run_dir / "design.json").write_text(json.dumps(design, indent=2, default=str), encoding="utf-8")
    from atlas.evaluation.report import write_markdown

    write_markdown(run_dir)
    return report, run_dir


def _train_fold_ppo(wf: WalkForwardConfig, fold_index: int, train_refs: list[SessionRef], sim: SimulatorConfig,
                    evo: EvolutionConfig, run_dir: Path) -> tuple[AgentRecord, dict[str, Any]]:
    """Train PPO on this fold's training sessions only (normalizer fit on the same sessions)."""
    from atlas.learning.ppo import train_ppo

    settings = wf.ppo.model_copy(update={"seed": wf.ppo.seed + fold_index})
    res = train_ppo(settings, train_refs, sim, evo.limits, evo.evaluation.decision_interval_ms,
                    run_dir / "ppo" / f"fold_{fold_index:02d}")
    record = AgentRecord(agent_id=f"ppo-fold{fold_index:02d}", family="ppo", generation=0, genome={},
                         origin="ppo", artifact=res.artifact)
    return record, {"summary": res.summary(), "diagnostics": res.diagnostics, "episodes": len(res.episodes),
                    "env_steps": settings.total_steps // settings.rollout_steps * settings.rollout_steps}


def _ppo_summary(wf: WalkForwardConfig, folds: list[dict[str, Any]]) -> dict[str, Any]:
    scenarios = list(wf.cost_scenarios)
    returns: dict[str, list[float]] = {s: [] for s in scenarios}
    comparison, in_s, oos_b, warnings = [], [], [], []
    for f in folds:
        p = f.get("ppo")
        if p is None:
            continue
        for s in scenarios:
            returns[s].extend(p["out_of_sample"][s]["session_returns"])
        in_s.append(p["in_sample"]["mean_net_return"])
        oos_b.append(p["out_of_sample"]["base"]["mean_net_return"])
        warnings += [f"fold {f['index']}: {w}" for w in f["ppo_training"]["summary"]["warnings"]]
        champ = f["candidates"][0] if f["candidates"] else None
        comparison.append({
            "fold": f["index"],
            "ppo_base": p["out_of_sample"]["base"]["mean_net_return"],
            "ppo_pessimistic": p["out_of_sample"]["pessimistic"]["mean_net_return"],
            "evolution_base": champ["out_of_sample"]["base"]["mean_net_return"] if champ else None,
            "evolution_pessimistic": champ["out_of_sample"]["pessimistic"]["mean_net_return"] if champ else None,
            "ppo_fills_base": p["out_of_sample"]["base"]["fills"],
            "reward_vs_realized_gap_bps": f["ppo_training"]["summary"]["mean_abs_reward_vs_realized_gap_bps"],
            "final_action_distribution": f["ppo_training"]["summary"]["final_action_distribution"],
            "max_approx_kl": f["ppo_training"]["summary"]["max_approx_kl"],
            "p50_us": p["latency"]["p50_us"], "p99_us": p["latency"]["p99_us"],
        })
    wins = {s: sum(1 for c in comparison if c[f"evolution_{s}"] is not None and c[f"ppo_{s}"] > c[f"evolution_{s}"])
            for s in ("base", "pessimistic")}
    return {
        "oos": {s: bootstrap_mean_ci(v, wf.bootstrap) for s, v in returns.items()},
        "selection_gap": {
            "mean_in_sample_return": statistics.fmean(in_s) if in_s else None,
            "mean_oos_base_return": statistics.fmean(oos_b) if oos_b else None,
        },
        "folds_beating_evolution_champion": wins,
        "folds_compared": len(comparison),
        "comparison": comparison,
        "training_warnings": warnings,
        "promotion": (
            "PPO is the configured final-candidate source (explicit setting)." if wf.final_candidate_source == "ppo"
            else "Not promoted. The final candidate comes from evolution; PPO is never promoted automatically."
        ),
    }


def _aggregate(wf, evo, sim, base, provider, plan, folds, total_trials, final_candidate) -> dict[str, Any]:
    scenarios = list(wf.cost_scenarios)
    champion_returns: dict[str, list[float]] = {s: [] for s in scenarios}
    baseline_returns: dict[str, dict[str, list[float]]] = {}
    in_sample, oos_base = [], []
    by_session: list[dict[str, Any]] = []
    symbol_pnl: dict[str, float] = {}
    for f in folds:
        if not f["candidates"]:
            continue
        champ = f["candidates"][0]
        in_sample.append(champ["in_sample"]["mean_net_return"])
        oos_base.append(champ["out_of_sample"]["base"]["mean_net_return"])
        for s in scenarios:
            champion_returns[s].extend(champ["out_of_sample"][s]["session_returns"])
        for sym, v in champ["out_of_sample"]["base"]["symbol_pnl"].items():
            symbol_pnl[sym] = symbol_pnl.get(sym, 0.0) + v
        for bid, res in f["baselines"].items():
            for s in scenarios:
                baseline_returns.setdefault(bid, {x: [] for x in scenarios})[s].extend(res[s]["session_returns"])
        for d, r in zip(f["validation"], champ["out_of_sample"]["base"]["session_returns"]):
            by_session.append({"session": d, "fold": f["index"], "champion": champ["agent_id"], "base_return": r})

    # Volatility regimes: terciles of realized vol across validation sessions.
    vols = {row["session"]: session_volatility_bps(provider.session_ref(date.fromisoformat(row["session"])))
            for row in by_session}
    ordered = sorted(v for v in vols.values() if not math.isnan(v))
    regimes: dict[str, list[float]] = {"low": [], "mid": [], "high": []}
    for row in by_session:
        v = vols[row["session"]]
        row["volatility_bps"] = v
        if ordered and not math.isnan(v):
            rank = ordered.index(v) / max(1, len(ordered) - 1)
            row["regime"] = "low" if rank < 1 / 3 else "mid" if rank < 2 / 3 else "high"
            regimes[row["regime"]].append(row["base_return"])

    dev_refs = [provider.session_ref(d) for d in plan.development]
    gaps = [i - o for i, o in zip(in_sample, oos_base) if i is not None and o is not None]
    return {
        "atlas_version": __version__,
        "experiment_name": wf.experiment_name,
        "design_fingerprint": design_fingerprint(wf, evo, sim, base, plan, final_candidate),
        "dataset_id": dataset_id(provider),
        "data_source": provider.describe(),
        "data_quality": data_quality(dev_refs),
        "plan": plan.to_dict(),
        "holdout": {"sessions": [d.isoformat() for d in plan.holdout], "status": "untouched by this run"},
        "cost_scenarios": {k: v.model_dump() for k, v in wf.cost_scenarios.items()},
        "emphasized_scenarios": list(EMPHASIZED),
        "selection_trials": {"agent_session_evaluations": total_trials, "folds": len(plan.folds),
                             "candidates_validated_per_fold": wf.candidates_per_fold},
        "folds": folds,
        "summary": {
            "champion_oos": {s: bootstrap_mean_ci(v, wf.bootstrap) for s, v in champion_returns.items()},
            "champion_positive_session_fraction": {
                s: (sum(r > 0 for r in v) / len(v) if v else None) for s, v in champion_returns.items()},
            "baselines_oos": {b: {s: bootstrap_mean_ci(v, wf.bootstrap) for s, v in res.items()}
                              for b, res in baseline_returns.items()},
            "selection_gap": {
                "mean_in_sample_return": statistics.fmean([x for x in in_sample if x is not None]) if in_sample else None,
                "mean_oos_base_return": statistics.fmean([x for x in oos_base if x is not None]) if oos_base else None,
                "mean_gap": statistics.fmean(gaps) if gaps else None,
                "note": "in-sample minus out-of-sample mean session return of each fold's top champion; "
                        "a large positive gap means selection fit noise in the training sessions",
            },
            "stability": {
                "by_session": by_session,
                "by_symbol_base_pnl": symbol_pnl,
                "by_volatility_regime": {k: (statistics.fmean(v) if v else None, len(v)) for k, v in regimes.items()},
            },
        },
        "limitations": limitations(provider, plan, wf, folds),
        "disclaimer": DISCLAIMER,
    }


def limitations(provider: MarketDataProvider, plan: SplitPlan, wf: WalkForwardConfig,
                folds: list[dict[str, Any]]) -> list[str]:
    d = provider.describe()
    out = []
    if d.get("provider") == "synthetic" or d.get("timestamp_kind") == "synthetic":
        out.append("Data is synthetic (a random walk with no exploitable structure); results test the pipeline, "
                   "not any real-market hypothesis.")
    out.append(f"Data fidelity: {d.get('fidelity')}. No order-book depth, queue position, or market impact "
               "beyond fixed slippage is modeled; fills are IOC against top of book.")
    out.append(f"Timestamps: {d.get('timestamp_kind')}. Sub-second horizons are only meaningful if the "
               "feed's timestamps and latency support them; this must be measured for any real feed.")
    out.append(f"Price adjustment: {d.get('price_adjustment')}")
    out.append(f"Universe: {d.get('universe')}" if d.get("universe") else
               f"Universe source: {d.get('universe_source')}. Delisted symbols are only covered if the vendor "
               "data includes them.")
    out.append("Holidays and early closes are not modeled in session calendars; halts are not modeled.")
    n_val = sum(len(f["validation"]) for f in folds)
    out.append(f"Out-of-sample evidence spans {n_val} validation sessions across {len(folds)} folds; "
               "per-session returns are not independent, so confidence intervals are approximate.")
    if any(not f["candidates"] for f in folds):
        out.append("Some folds produced no eligible candidate; their validation sessions contribute no "
                   "champion evidence.")
    out.append("Sharpe/Sortino come from 1-minute returns within short sessions and are noisy.")
    if wf.ppo.enabled:
        out.append(f"PPO trains per fold for {wf.ppo.total_steps} environment steps on that fold's training "
                   "sessions only. That is a small budget by RL standards, and results reflect it. It is "
                   "evaluated frozen and greedy (argmax), with the same costs, limits, and latency as every "
                   "other agent.")
        out.append("PPO is a separate, versioned family (ppo-v1) evaluated beside the 50-agent population; it "
                   "is not bred into the population and is never promoted automatically.")
    out.append(f"The outer holdout ({len(plan.holdout)} sessions) was not evaluated in this run.")
    return out


# ------------------------------------------------------------------ holdout


def evaluate_holdout(run_dir: Path, wf: WalkForwardConfig, evo: EvolutionConfig, sim: SimulatorConfig,
                     base: DataConfig, provider: MarketDataProvider, allow_reuse: bool = False) -> dict[str, Any]:
    design = json.loads((run_dir / "design.json").read_text(encoding="utf-8"))
    plan = plan_splits(provider.sessions(), wf.splits)
    candidate = None if design["final_candidate"] is None else AgentRecord.from_dict(design["final_candidate"])
    fp = design_fingerprint(wf, evo, sim, base, plan, candidate)
    if fp != design["design_fingerprint"]:
        raise ValueError("current configs/data do not match the frozen design; refusing holdout evaluation")
    if candidate is None:
        raise ValueError("the frozen design has no final candidate (no fold produced an eligible champion)")
    ledger = HoldoutLedger(wf.holdout_ledger)
    ds = dataset_id(provider)
    refs = HoldoutGuard(provider, plan.holdout).unlock(ledger, ds, fp, f"holdout evaluation of {run_dir.name}",
                                                        allow_reuse)
    records = [candidate] + baseline_records()
    res = scenario_evaluate(records, refs, sim, evo, wf.cost_scenarios)
    accesses = ledger.accesses(ds)
    out = {
        "run_id": run_dir.name,
        "design_fingerprint": fp,
        "dataset_id": ds,
        "holdout_sessions": [r.session_date.isoformat() for r in refs],
        "holdout_accesses_total": len(accesses),
        "holdout_reused": any(a["reuse"] for a in accesses),
        "candidate": {"agent_id": candidate.agent_id, "family": candidate.family, "genome": candidate.genome},
        "results": {s: {aid: summarize(ev) for aid, ev in evs.items()} for s, evs in res.items()},
        "champion_ci": {s: bootstrap_mean_ci([m.net_return for m in evs[candidate.agent_id].windows], wf.bootstrap)
                        for s, evs in res.items()},
        "data_quality": data_quality(refs),
        "disclaimer": DISCLAIMER,
    }
    (run_dir / "holdout_report.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    from atlas.evaluation.report import write_markdown

    write_markdown(run_dir)
    return out


# ------------------------------------------------------- checkpoint stress


def evaluate_checkpoint(checkpoint: Path, wf: WalkForwardConfig, evo: EvolutionConfig, sim: SimulatorConfig,
                        provider: MarketDataProvider, top: int, out_root: Path) -> tuple[dict[str, Any], Path]:
    """Stress-test a checkpoint's champions on sessions after its training data, under all scenarios."""
    ckpt = json.loads(checkpoint.read_text(encoding="utf-8"))
    champions = [AgentRecord.from_dict(e["record"]) for e in ckpt["archives"]["champions"][:top]]
    if not champions:
        raise ValueError(f"{checkpoint} has no archived champions to evaluate")
    plan = plan_splits(provider.sessions(), wf.splits)
    trained = sorted(date.fromisoformat(d) for d in ckpt.get("train_sessions", []))
    after = [d for d in plan.development if not trained or d > trained[-1]]
    if not after:
        raise ValueError("no non-holdout sessions after the checkpoint's training sessions")
    refs = HoldoutGuard(provider, plan.holdout).refs(after)
    res = scenario_evaluate(champions + baseline_records(), refs, sim, evo, wf.cost_scenarios)
    out = {
        "checkpoint": str(checkpoint),
        "train_sessions": [d.isoformat() for d in trained],
        "evaluation_sessions": [d.isoformat() for d in after],
        "results": {s: {aid: summarize(ev) for aid, ev in evs.items()} for s, evs in res.items()},
        "disclaimer": DISCLAIMER,
    }
    out_dir = out_root / f"stress-{checkpoint.parent.name}-{checkpoint.stem}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "stress_report.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out, out_dir
