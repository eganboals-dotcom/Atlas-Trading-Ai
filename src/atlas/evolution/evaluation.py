"""Evaluate agents across sessions, serially or in worker processes.

Every agent sees the same sessions, data, costs, and simulator seed. Each run builds a
fresh agent instance from its record, so agents share no mutable state. Parallel and
serial evaluation produce identical results: every task is independent, and results
come back in submission order.

Sessions are passed as ``SessionRef`` objects, so the same code evaluates synthetic
windows, imported historical files, or walk-forward folds.
"""

from __future__ import annotations

import atexit
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from atlas.agents.adapter import AgentStrategy
from atlas.agents.base import AgentRecord
from atlas.agents.registry import create_agent
from atlas.config import AgentLimits, DataConfig, EvolutionConfig, SimulatorConfig, WindowConfig
from atlas.data.providers import SessionRef, load_session
from atlas.environment.portfolio import AccountingError
from atlas.environment.replay import ReplayEngine
from atlas.evaluation.metrics import RunMetrics, run_metrics
from atlas.timeutils import NS_PER_MS


@dataclass
class AgentEvaluation:
    agent_id: str
    windows: list[RunMetrics] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    error: str | None = None
    # Wall-clock decision timings per window; machine-dependent, so never checkpointed.
    decision_wall_ns: list[list[int]] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "windows": [w.to_dict() for w in self.windows],
            "violations": self.violations,
            "error": self.error,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> AgentEvaluation:
        return AgentEvaluation(d["agent_id"], [RunMetrics(**w) for w in d["windows"]], list(d["violations"]),
                               d["error"])


def window_data_config(base: DataConfig, window: WindowConfig, evaluation: Any) -> DataConfig:
    update: dict[str, Any] = {"session_date": window.session_date, "duration_s": evaluation.duration_s}
    if evaluation.symbols:
        update["symbols"] = evaluation.symbols
    synthetic = base.synthetic.model_copy(update={"seed": window.seed})
    return DataConfig.model_validate({**base.model_dump(), **update, "synthetic": synthetic.model_dump()})


def window_refs(evo: EvolutionConfig, base: DataConfig) -> list[SessionRef]:
    """SessionRefs for the explicit synthetic windows listed in an evolution config."""
    refs = []
    for w in evo.evaluation.windows:
        cfg = window_data_config(base, w, evo.evaluation)
        refs.append(SessionRef(kind="synthetic", session_date=cfg.session_date, symbols=tuple(cfg.symbols),
                               allowed_data_age_ms=cfg.allowed_data_age_ms, data_config=cfg.model_dump_json()))
    return refs


@dataclass(frozen=True)
class EvalTask:
    record: dict[str, Any]
    sessions: tuple[str, ...]  # SessionRef JSON per session
    simulator: str  # SimulatorConfig JSON
    limits: str  # AgentLimits JSON
    decision_interval_ms: float


def evaluate_task(task: EvalTask) -> AgentEvaluation:
    """Top-level (picklable) worker function."""
    record = AgentRecord.from_dict(task.record)
    sim_cfg = SimulatorConfig.model_validate_json(task.simulator)
    limits = AgentLimits.model_validate_json(task.limits)
    out = AgentEvaluation(record.agent_id)
    for i, key in enumerate(task.sessions):
        ref = SessionRef.model_validate_json(key)
        symbols = list(ref.symbols)
        strategy = AgentStrategy(
            create_agent(record),
            symbols,
            limits,
            decision_interval_ns=int(task.decision_interval_ms * NS_PER_MS),
            stale_ns=int(sim_cfg.stale_quote_ms * NS_PER_MS),
            seed=sim_cfg.seed + i,
        )
        try:
            result = ReplayEngine(sim_cfg, symbols).run(load_session(ref), strategy)
        except AccountingError as exc:
            out.error = f"session {ref.session_date}: accounting error: {exc}"
            return out
        out.windows.append(run_metrics(result, sim_cfg.initial_cash))
        out.decision_wall_ns.append(result.decision_wall_ns)
        out.violations.extend(f"session {ref.session_date}: {v}" for v in strategy.violations)
    return out


def resolve_workers(workers: int) -> int:
    return max(1, (os.cpu_count() or 2) - 1) if workers == 0 else workers


def evaluate_records(
    records: list[AgentRecord],
    refs: list[SessionRef],
    sim_cfg: SimulatorConfig,
    limits: AgentLimits,
    decision_interval_ms: float,
    workers: int = 1,
) -> list[AgentEvaluation]:
    keys = tuple(r.key for r in refs)
    tasks = [EvalTask(r.to_dict(), keys, sim_cfg.model_dump_json(), limits.model_dump_json(), decision_interval_ms)
             for r in records]
    n = resolve_workers(workers)
    if n <= 1 or len(tasks) <= 1:
        return [evaluate_task(t) for t in tasks]
    return list(_pool(n).map(evaluate_task, tasks, chunksize=max(1, len(tasks) // (n * 4))))


_POOLS: dict[int, ProcessPoolExecutor] = {}


def _pool(workers: int) -> ProcessPoolExecutor:
    """One long-lived pool per worker count. Workers keep their session caches warm across batches,
    and process start-up (slow on Windows) is paid once per run instead of once per batch."""
    if workers not in _POOLS:
        _POOLS[workers] = ProcessPoolExecutor(max_workers=workers)
        atexit.register(_POOLS[workers].shutdown)
    return _POOLS[workers]


class Evaluator:
    """Evaluates records on a fixed list of sessions with the evolution config's settings."""

    def __init__(self, evo: EvolutionConfig, sim_cfg: SimulatorConfig, base_data: DataConfig,
                 sessions: list[SessionRef] | None = None) -> None:
        self.evo = evo
        self.sim_cfg = sim_cfg
        self.sessions = sessions if sessions is not None else window_refs(evo, base_data)
        self.workers = resolve_workers(evo.evaluation.workers)

    def evaluate(self, records: list[AgentRecord]) -> list[AgentEvaluation]:
        return evaluate_records(records, self.sessions, self.sim_cfg, self.evo.limits,
                                self.evo.evaluation.decision_interval_ms, self.workers)
