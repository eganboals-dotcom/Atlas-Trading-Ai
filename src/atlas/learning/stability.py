"""Training-stability study: train PPO with several seeds on one fold's training sessions,
then score each frozen policy on that fold's validation sessions under identical costs.

The spread across seeds shows how much of any result is optimizer luck.
"""

from __future__ import annotations

import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from atlas.agents.base import AgentRecord
from atlas.config import EvolutionConfig, SimulatorConfig
from atlas.data.providers import MarketDataProvider
from atlas.evaluation.holdout import HoldoutGuard
from atlas.evaluation.splits import plan_splits
from atlas.evaluation.walkforward import WalkForwardConfig, scenario_evaluate, summarize
from atlas.learning.ppo import train_ppo


def seed_stability(wf: WalkForwardConfig, evo: EvolutionConfig, sim: SimulatorConfig, provider: MarketDataProvider,
                   fold_index: int, seeds: int, out_root: Path) -> tuple[dict[str, Any], Path]:
    plan = plan_splits(provider.sessions(), wf.splits)
    if not 0 <= fold_index < len(plan.folds):
        raise ValueError(f"fold {fold_index} out of range (0..{len(plan.folds) - 1})")
    fold = plan.folds[fold_index]
    guard = HoldoutGuard(provider, plan.holdout)
    train_refs, val_refs = guard.refs(fold.train), guard.refs(fold.validation)
    out_dir = out_root / f"{wf.experiment_name}-fold{fold_index:02d}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    runs = []
    for k in range(seeds):
        settings = wf.ppo.model_copy(update={"seed": wf.ppo.seed + 1000 * k, "enabled": True})
        res = train_ppo(settings, train_refs, sim, evo.limits, evo.evaluation.decision_interval_ms,
                        out_dir / f"seed_{settings.seed}")
        rec = AgentRecord(f"ppo-seed{settings.seed}", "ppo", 0, {}, origin="ppo", artifact=res.artifact)
        scen = {s: wf.cost_scenarios[s] for s in ("base", "pessimistic")}
        oos = scenario_evaluate([rec], val_refs, sim, evo, scen)
        runs.append({"seed": settings.seed, "training": res.summary(),
                     "validation": {s: summarize(evs[rec.agent_id]) for s, evs in oos.items()}})
    base = [r["validation"]["base"]["mean_net_return"] for r in runs]
    out = {
        "fold": fold_index,
        "train": [d.isoformat() for d in fold.train],
        "validation": [d.isoformat() for d in fold.validation],
        "seeds": [r["seed"] for r in runs],
        "base_return_by_seed_bps": [x * 1e4 for x in base],
        "base_return_spread_bps": (max(base) - min(base)) * 1e4 if base else None,
        "base_return_std_bps": statistics.pstdev(base) * 1e4 if len(base) > 1 else None,
        "runs": runs,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stability.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out, out_dir
