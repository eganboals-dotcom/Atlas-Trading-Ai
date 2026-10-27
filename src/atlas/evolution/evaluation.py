"""Evaluate agents across fixed evaluation windows, serially or in worker processes.

Every agent sees the same windows, data, costs, and simulator seed. Each run builds a
fresh agent instance from its record, so agents share no mutable state. Parallel and
serial evaluation produce identical results because each task is independent and
results are returned in submission order.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from atlas.agents.adapter import AgentStrategy
from atlas.agents.base import AgentRecord
from atlas.agents.registry import create_agent
from atlas.config import DataConfig, EvolutionConfig, SimulatorConfig, WindowConfig
from atlas.data.events import MarketEvent
from atlas.data.synthetic import generate_events
from atlas.data.validation import validate_events
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


_EVENT_CACHE: dict[str, list[MarketEvent]] = {}


def window_events(cfg: DataConfig) -> list[MarketEvent]:
    """Validated events for a window, cached per process (data is immutable)."""
    key = cfg.model_dump_json()
    if key not in _EVENT_CACHE:
        _EVENT_CACHE[key] = validate_events(generate_events(cfg), cfg.allowed_data_age_ms).valid_events
    return _EVENT_CACHE[key]


@dataclass(frozen=True)
class EvalTask:
    record: dict[str, Any]
    windows: tuple[str, ...]  # DataConfig JSON per window
    simulator: str  # SimulatorConfig JSON
    limits: str  # AgentLimits JSON
    decision_interval_ms: float


def evaluate_task(task: EvalTask) -> AgentEvaluation:
    """Top-level (picklable) worker function."""
    from atlas.config import AgentLimits

    record = AgentRecord.from_dict(task.record)
    sim_cfg = SimulatorConfig.model_validate_json(task.simulator)
    limits = AgentLimits.model_validate_json(task.limits)
    out = AgentEvaluation(record.agent_id)
    for i, window_json in enumerate(task.windows):
        data_cfg = DataConfig.model_validate_json(window_json)
        events = window_events(data_cfg)
        strategy = AgentStrategy(
            create_agent(record),
            data_cfg.symbols,
            limits,
            decision_interval_ns=int(task.decision_interval_ms * NS_PER_MS),
            stale_ns=int(sim_cfg.stale_quote_ms * NS_PER_MS),
            seed=sim_cfg.seed + i,
        )
        try:
            result = ReplayEngine(sim_cfg, data_cfg.symbols).run(events, strategy)
        except AccountingError as exc:
            out.error = f"window {i}: accounting error: {exc}"
            return out
        out.windows.append(run_metrics(result, sim_cfg.initial_cash))
        out.violations.extend(f"window {i}: {v}" for v in strategy.violations)
    return out


class Evaluator:
    def __init__(self, evo: EvolutionConfig, sim_cfg: SimulatorConfig, base_data: DataConfig) -> None:
        self.evo = evo
        self.sim_cfg = sim_cfg
        self.window_configs = [window_data_config(base_data, w, evo.evaluation) for w in evo.evaluation.windows]
        self._windows_json = tuple(c.model_dump_json() for c in self.window_configs)
        workers = evo.evaluation.workers
        self.workers = max(1, (os.cpu_count() or 2) - 1) if workers == 0 else workers

    def tasks(self, records: list[AgentRecord]) -> list[EvalTask]:
        return [
            EvalTask(r.to_dict(), self._windows_json, self.sim_cfg.model_dump_json(),
                     self.evo.limits.model_dump_json(), self.evo.evaluation.decision_interval_ms)
            for r in records
        ]

    def evaluate(self, records: list[AgentRecord]) -> list[AgentEvaluation]:
        tasks = self.tasks(records)
        if self.workers <= 1 or len(tasks) <= 1:
            return [evaluate_task(t) for t in tasks]
        with ProcessPoolExecutor(max_workers=min(self.workers, len(tasks))) as pool:
            return list(pool.map(evaluate_task, tasks, chunksize=max(1, len(tasks) // (self.workers * 4))))
