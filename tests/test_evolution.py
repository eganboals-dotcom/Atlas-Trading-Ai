from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from helpers import data_config, sim_config

from atlas.config import EvolutionConfig
from atlas.evolution.evaluation import Evaluator
from atlas.evolution.runner import CheckpointError, EvolutionRunner


def evo_config(tmp_path: Path, **overrides: Any) -> EvolutionConfig:
    base: dict[str, Any] = {
        "experiment_name": "test_exp",
        "seed": 11,
        "generations": 2,
        "simulator_config": "unused.yaml",
        "data_config": "unused.yaml",
        "evaluation": {
            "windows": [{"session_date": "2026-10-05", "seed": 1}, {"session_date": "2026-10-06", "seed": 2}],
            "duration_s": 120,
            "symbols": ["SPY"],
            "workers": 1,
        },
        "fitness": {"gates": {"min_fills": 4, "min_windows_traded": 2}},
        "checkpoint_dir": str(tmp_path / "ckpt"),
        "reports_dir": str(tmp_path / "reports"),
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return EvolutionConfig.model_validate(base)


def runner(tmp_path: Path, **overrides: Any) -> EvolutionRunner:
    return EvolutionRunner(evo_config(tmp_path, **overrides), sim_config(stale_quote_ms=2_000.0), data_config())


def checkpoint(r: EvolutionRunner) -> dict[str, Any]:
    return json.loads(r.latest_checkpoint().read_text(encoding="utf-8"))


def test_config_requires_exactly_50_from_5_x_10() -> None:
    with pytest.raises(ValueError):
        evo_config(Path("."), parent_count=4)
    with pytest.raises(ValueError):
        evo_config(Path("."), offspring={"immigrants": 2})
    with pytest.raises(ValueError):
        evo_config(Path("."), population_size=40)


def test_generations_always_have_50_agents_and_full_lineage(tmp_path: Path) -> None:
    r = runner(tmp_path)
    state = r.run()
    assert state.generation == 2 and state.stopped is None
    assert len(state.population) == 50
    assert all(a.generation == 2 for a in state.population)

    lineage = {a.agent_id: a for a in state.lineage}
    assert len(state.lineage) == len(lineage) == 150  # initial + two bred generations, ids unique
    for a in state.lineage:
        for parent in a.parents:
            assert lineage[parent].generation == a.generation - 1
    for gen in (1, 2):
        kids = [a for a in state.lineage if a.generation == gen]
        assert len(kids) == 50
        assert all(a.origin != "initial" for a in kids)

    reports = r.report_dir
    for name in ("generation_000.json", "generation_001.json", "leaderboard_001.csv", "lineage.jsonl",
                 "summary.json"):
        assert (reports / name).is_file(), name
    report = json.loads((reports / "generation_001.json").read_text(encoding="utf-8"))
    assert len(report["leaderboard"]) == 50
    assert len(report["parents"]) == 5
    assert set(report["baselines"]) == {"baseline-cash", "baseline-buy_hold", "baseline-momentum",
                                        "baseline-mean_reversion", "baseline-random", "baseline-coin_flip"}
    assert report["fitness_fingerprint"] and report["disclaimer"]
    assert not any(a.agent_id.startswith("baseline") for a in state.lineage)


def test_resume_matches_uninterrupted_run(tmp_path: Path) -> None:
    straight = runner(tmp_path / "a")
    straight.run(generations=3)

    first = runner(tmp_path / "b")
    first.run(generations=1)
    resumed = runner(tmp_path / "b")
    resumed.run(generations=3, resume=True)
    assert checkpoint(resumed) == checkpoint(straight)


def test_refuses_to_overwrite_or_resume_with_changed_config(tmp_path: Path) -> None:
    runner(tmp_path).run(generations=1)
    with pytest.raises(CheckpointError, match="already has checkpoints"):
        runner(tmp_path).run(generations=2)
    changed = runner(tmp_path, seed=99)
    with pytest.raises(CheckpointError, match="fingerprint"):
        changed.run(generations=2, resume=True)


def test_hold_when_too_few_eligible_parents(tmp_path: Path) -> None:
    r = runner(tmp_path, fitness={"gates": {"min_fills": 10_000_000}})
    initial_ids = [a.agent_id for a in r.fresh_state().population]
    state = r.run(generations=3)
    assert state.generation == 0
    assert "promotion held" in state.stopped
    assert [a.agent_id for a in state.population] == initial_ids
    assert state.history[-1]["status"] == "held"


def test_explore_fills_with_labeled_immigrants(tmp_path: Path) -> None:
    r = runner(tmp_path, fitness={"gates": {"min_fills": 10_000_000}}, on_insufficient_parents="explore")
    state = r.run(generations=1)
    assert state.generation == 1
    assert len(state.population) == 50
    assert all(a.origin == "immigrant" and "exploration fill" in a.mutations[-1] for a in state.population)


def test_parallel_evaluation_matches_serial(tmp_path: Path) -> None:
    serial = runner(tmp_path)
    records = serial.fresh_state().population[::7]
    parallel = Evaluator(evo_config(tmp_path, evaluation={"workers": 2}), serial.sim, data_config())
    assert [e.to_dict() for e in serial.evaluator.evaluate(records)] == \
           [e.to_dict() for e in parallel.evaluate(records)]
