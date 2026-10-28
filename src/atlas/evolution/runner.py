"""Generation loop with checkpoint/resume, lineage, and reports.

One generation = evaluate all 50 agents on every evaluation window -> score with frozen
gates/weights -> update archives -> select up to five diverse eligible parents ->
breed exactly 50 descendants -> checkpoint -> report.

If fewer eligible parents than required exist, nothing weak is promoted:
* ``hold``: stop, keep the current population and champion, and report why;
* ``explore``: the eligible parents breed their slots and every remaining slot is
  filled with fresh immigrants, explicitly labeled.

Checkpoints contain no wall-clock data, so an interrupted-and-resumed run produces
exactly the same checkpoint as an uninterrupted one.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from atlas import __version__
from atlas.agents.base import POPULATION_VERSION, AgentRecord
from atlas.agents.baselines import baseline_records
from atlas.config import DataConfig, EvolutionConfig, SimulatorConfig
from atlas.data.providers import SessionRef
from atlas.evolution.archive import Archives
from atlas.evolution.evaluation import AgentEvaluation, Evaluator
from atlas.evolution.fitness import FitnessResult, fitness_fingerprint, score
from atlas.evolution.reproduction import immigrant, initial_population, make_offspring
from atlas.evolution.selection import select_parents
from atlas.logging_utils import get_logger

log = get_logger(__name__)

CHECKPOINT_FORMAT = 1
DISCLAIMER = (
    "Synthetic-data research run. Scores measure behavior under this simulator's assumptions only "
    "and are not evidence of real-market profitability."
)


class CheckpointError(RuntimeError):
    pass


@dataclass
class EvolutionState:
    generation: int  # the generation whose population is waiting to be evaluated
    population: list[AgentRecord]
    rng_state: dict[str, Any]
    archives: Archives
    lineage: list[AgentRecord]
    history: list[dict[str, Any]]
    baselines: dict[str, dict[str, Any]]
    evaluations_run: int
    stopped: str | None = None

    def to_dict(self, fingerprint: str) -> dict[str, Any]:
        return {
            "format": CHECKPOINT_FORMAT,
            "config_fingerprint": fingerprint,
            "atlas_version": __version__,
            "population_version": POPULATION_VERSION,
            "generation": self.generation,
            "population": [r.to_dict() for r in self.population],
            "rng_state": self.rng_state,
            "archives": self.archives.to_dict(),
            "lineage": [r.to_dict() for r in self.lineage],
            "history": self.history,
            "baselines": self.baselines,
            "evaluations_run": self.evaluations_run,
            "stopped": self.stopped,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> EvolutionState:
        return EvolutionState(
            generation=d["generation"],
            population=[AgentRecord.from_dict(r) for r in d["population"]],
            rng_state=d["rng_state"],
            archives=Archives.from_dict(d["archives"]),
            lineage=[AgentRecord.from_dict(r) for r in d["lineage"]],
            history=d["history"],
            baselines=d["baselines"],
            evaluations_run=d["evaluations_run"],
            stopped=d.get("stopped"),
        )


def config_fingerprint(evo: EvolutionConfig, sim: SimulatorConfig, data: DataConfig,
                       sessions: list[SessionRef]) -> str:
    """Everything that affects results. Run length, worker count, and output paths are excluded."""
    payload = {
        "evolution": evo.model_dump(mode="json", exclude={"generations", "checkpoint_dir", "reports_dir"}),
        "simulator": sim.model_dump(mode="json", exclude={"reports_dir", "demo_strategy"}),
        "data": data.model_dump(mode="json"),
        "sessions": [r.model_dump(mode="json", exclude={"path"}) for r in sessions],
    }
    payload["evolution"]["evaluation"].pop("workers", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


class EvolutionRunner:
    def __init__(self, evo: EvolutionConfig, sim: SimulatorConfig, data: DataConfig,
                 sessions: list[SessionRef] | None = None) -> None:
        self.evo = evo
        self.sim = sim
        self.evaluator = Evaluator(evo, sim, data, sessions)
        self.sessions = self.evaluator.sessions
        self.fingerprint = config_fingerprint(evo, sim, data, self.sessions)
        self.ckpt_dir = evo.checkpoint_dir / evo.experiment_name
        self.report_dir = evo.reports_dir / evo.experiment_name
        self.n_windows = len(self.sessions)

    # ----------------------------------------------------------------- state

    def fresh_state(self) -> EvolutionState:
        rng = np.random.default_rng(self.evo.seed)
        population = initial_population(self.evo.families, self.evo.population_size, rng)
        d = self.evo.diversity
        return EvolutionState(
            generation=0,
            population=population,
            rng_state=rng.bit_generator.state,
            archives=Archives(d.champion_archive_size, d.novelty_archive_size, d.novelty_threshold),
            lineage=list(population),
            history=[],
            baselines={},
            evaluations_run=0,
        )

    def latest_checkpoint(self) -> Path | None:
        files = sorted(self.ckpt_dir.glob("gen_*.json"))
        return files[-1] if files else None

    def load_state(self, path: Path) -> EvolutionState:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") != CHECKPOINT_FORMAT:
            raise CheckpointError(f"{path}: unsupported checkpoint format {data.get('format')}")
        if data["config_fingerprint"] != self.fingerprint:
            raise CheckpointError(
                f"{path}: config fingerprint {data['config_fingerprint']} does not match current "
                f"config {self.fingerprint}; refusing to resume with different settings"
            )
        return EvolutionState.from_dict(data)

    def save_state(self, state: EvolutionState) -> Path:
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = self.ckpt_dir / f"gen_{state.generation:03d}.json"
        tmp = path.with_suffix(".tmp")
        payload = state.to_dict(self.fingerprint)
        payload["train_sessions"] = [r.session_date.isoformat() for r in self.sessions]
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)  # atomic: a crash never leaves a half-written checkpoint
        return path

    # ------------------------------------------------------------------- run

    def run(self, generations: int | None = None, resume: bool = False) -> EvolutionState:
        target = generations if generations is not None else self.evo.generations
        latest = self.latest_checkpoint()
        if resume and latest is not None:
            state = self.load_state(latest)
            log.info("resumed", extra={"checkpoint": str(latest), "generation": state.generation})
        elif latest is not None and not resume:
            raise CheckpointError(
                f"experiment {self.evo.experiment_name!r} already has checkpoints in {self.ckpt_dir}; "
                "pass --resume to continue it or choose a new experiment_name"
            )
        else:
            state = self.fresh_state()
            self.save_state(state)

        if not state.baselines:
            state.baselines = self._evaluate_baselines()
        while state.generation < target and state.stopped is None:
            self.step(state)
            self.save_state(state)
            self._write_lineage(state)
        self._write_summary(state)
        return state

    def _benchmark(self, state: EvolutionState | None = None) -> tuple[float, ...] | None:
        """Hold's per-window returns on the training sessions (for ``fitness.benchmark: buy_hold``)."""
        if self.evo.fitness.benchmark != "buy_hold":
            return None
        if getattr(self, "_hold_returns", None) is None:
            hold = next(r for r in baseline_records() if r.family == "baseline_buy_hold")
            ev = self.evaluator.evaluate([hold])[0]
            if ev.error:
                raise RuntimeError(f"benchmark evaluation failed: {ev.error}")
            self._hold_returns = tuple(m.net_return for m in ev.windows)
        return self._hold_returns

    def _evaluate_baselines(self) -> dict[str, dict[str, Any]]:
        records = baseline_records()
        out = {}
        for rec, ev in zip(records, self.evaluator.evaluate(records)):
            fit = score(ev, self.evo.fitness, self.n_windows, self._benchmark())
            out[rec.agent_id] = {"family": rec.family, "fitness": fit.to_dict(), "evaluation": ev.to_dict()}
        return out

    def step(self, state: EvolutionState) -> None:
        gen = state.generation
        population = state.population
        if len(population) != self.evo.population_size:
            raise RuntimeError(f"population has {len(population)} agents, expected {self.evo.population_size}")

        evaluations = self.evaluator.evaluate(population)
        state.evaluations_run += len(population) * self.n_windows
        bench = self._benchmark()
        fitness = {ev.agent_id: score(ev, self.evo.fitness, self.n_windows, bench) for ev in evaluations}
        state.archives.update(population, fitness, gen)
        selection = select_parents(population, fitness, self.evo.parent_count,
                                   self.evo.diversity.min_parent_distance,
                                   self.evo.diversity.max_parents_per_family)

        rng = np.random.default_rng()
        rng.bit_generator.state = state.rng_state
        parents = selection.parents
        next_gen = gen + 1
        if len(parents) == self.evo.parent_count:
            status = "promoted"
            children = make_offspring(parents, next_gen, self.evo.offspring, self.evo.families, rng)
        elif self.evo.on_insufficient_parents == "explore":
            status = "explore"
            children = make_offspring(parents, next_gen, self.evo.offspring, self.evo.families, rng)
            while len(children) < self.evo.population_size:
                children.append(immigrant(self.evo.families, next_gen, len(children), rng))
                children[-1].mutations.append("exploration fill: not enough eligible parents")
        else:
            status = "held"
            children = population
            state.stopped = (
                f"generation {gen}: only {len(parents)} eligible parents (need {self.evo.parent_count}); "
                "promotion held, current population and champion preserved"
            )
        state.rng_state = rng.bit_generator.state

        summary = self._generation_summary(gen, status, population, evaluations, fitness, selection, state)
        state.history.append({k: summary[k] for k in ("generation", "status", "eligible", "best_score",
                                                      "best_agent", "parents", "champion")})
        self._write_generation_report(gen, summary)
        log.info("generation complete", extra={k: summary[k] for k in ("generation", "status", "eligible",
                                                                       "best_score", "best_agent")})
        if status != "held":
            if len(children) != self.evo.population_size:
                raise RuntimeError(f"bred {len(children)} children, expected {self.evo.population_size}")
            state.population = children
            state.lineage.extend(children)
            state.generation = next_gen

    # --------------------------------------------------------------- reports

    def _generation_summary(
        self,
        gen: int,
        status: str,
        population: list[AgentRecord],
        evaluations: list[AgentEvaluation],
        fitness: dict[str, FitnessResult],
        selection: Any,
        state: EvolutionState,
    ) -> dict[str, Any]:
        by_id = {r.agent_id: r for r in population}
        rows = []
        for ev in evaluations:
            fit, rec = fitness[ev.agent_id], by_id[ev.agent_id]
            agg = fit.aggregate
            rows.append({
                "agent_id": rec.agent_id, "family": rec.family, "origin": rec.origin,
                "parents": "|".join(rec.parents), "eligible": fit.eligible, "score": fit.score,
                "mean_net_return": agg.mean_net_return if agg else None,
                "worst_drawdown": agg.worst_drawdown if agg else None,
                "mean_turnover": agg.mean_turnover if agg else None,
                "instability": agg.instability if agg else None,
                "fills": agg.total_fills if agg else None,
                "reasons": "; ".join(fit.reasons),
            })
        rows.sort(key=lambda r: (not r["eligible"], -(r["score"] or 0.0), r["agent_id"]))
        best = next((r for r in rows if r["eligible"]), None)
        families: dict[str, dict[str, Any]] = {}
        for r in rows:
            f = families.setdefault(r["family"], {"agents": 0, "eligible": 0, "best_score": None})
            f["agents"] += 1
            if r["eligible"]:
                f["eligible"] += 1
                f["best_score"] = r["score"] if f["best_score"] is None else max(f["best_score"], r["score"])
        champion = state.archives.champions[0] if state.archives.champions else None
        return {
            "generation": gen,
            "status": status,
            "fitness_fingerprint": fitness_fingerprint(self.evo.fitness),
            "config_fingerprint": self.fingerprint,
            "fitness_config": self.evo.fitness.model_dump(mode="json"),
            "eligible": sum(r["eligible"] for r in rows),
            "best_score": best["score"] if best else None,
            "best_agent": best["agent_id"] if best else None,
            "parents": [p.agent_id for p in selection.parents],
            "selection_notes": selection.notes,
            "champion": None if champion is None else {
                "agent_id": champion.record.agent_id, "family": champion.record.family,
                "score": champion.score, "generation_evaluated": champion.generation_evaluated,
            },
            "families": families,
            "baselines": {
                bid: {
                    "eligible": b["fitness"]["eligible"],
                    "score": b["fitness"]["score"],
                    "mean_net_return": (b["fitness"]["aggregate"] or {}).get("mean_net_return"),
                    "reasons": b["fitness"]["reasons"],
                }
                for bid, b in state.baselines.items()
            },
            "evaluations_run_total": state.evaluations_run,
            "leaderboard": rows,
            "stopped": state.stopped,
            "disclaimer": DISCLAIMER,
        }

    def _write_generation_report(self, gen: int, summary: dict[str, Any]) -> None:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        (self.report_dir / f"generation_{gen:03d}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with (self.report_dir / f"leaderboard_{gen:03d}.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary["leaderboard"][0]))
            writer.writeheader()
            writer.writerows(summary["leaderboard"])

    def _write_lineage(self, state: EvolutionState) -> None:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        with (self.report_dir / "lineage.jsonl").open("w", encoding="utf-8") as fh:
            for r in state.lineage:
                fh.write(json.dumps(r.to_dict(), sort_keys=True) + "\n")

    def _write_summary(self, state: EvolutionState) -> None:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "experiment_name": self.evo.experiment_name,
            "config_fingerprint": self.fingerprint,
            "fitness_fingerprint": fitness_fingerprint(self.evo.fitness),
            "generations_completed": state.generation,
            "stopped": state.stopped,
            "evaluations_run_total": state.evaluations_run,
            "history": state.history,
            "champions": [e.to_dict() for e in state.archives.champions[:5]],
            "novelty_archive_size": len(state.archives.novelty),
            "agents_in_lineage": len(state.lineage),
            "disclaimer": DISCLAIMER,
        }
        (self.report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
