from __future__ import annotations

import numpy as np
import pytest

from atlas.agents.base import AgentRecord
from atlas.agents.registry import FAMILY_CLASSES
from atlas.config import FitnessConfig, OffspringConfig
from atlas.evaluation.metrics import RunMetrics, downside_deviation, max_drawdown
from atlas.evolution.archive import Archives
from atlas.evolution.evaluation import AgentEvaluation
from atlas.evolution.fitness import FitnessResult, fitness_fingerprint, score
from atlas.evolution.reproduction import initial_population, make_offspring
from atlas.evolution.selection import select_parents

CFG = FitnessConfig()


def m(ret=0.001, dd=0.002, down=0.0005, turn=2.0, fills=20) -> RunMetrics:
    return RunMetrics(ret, dd, down, turn, fills, fills, 0, 1.0, 100_000 * (1 + ret))


def ev(agent_id="a", windows=None, violations=(), error=None) -> AgentEvaluation:
    return AgentEvaluation(agent_id, windows if windows is not None else [m(), m(), m()], list(violations), error)


def test_metric_definitions() -> None:
    assert max_drawdown([100, 110, 99, 120, 108]) == pytest.approx(11 / 110)
    assert max_drawdown([100, 101, 102]) == 0.0
    assert downside_deviation([100, 100, 100]) == 0.0
    assert downside_deviation([100, 99]) == pytest.approx(0.01)


def test_eligible_agent_gets_documented_score() -> None:
    r = score(ev(), CFG, 3)
    assert r.eligible
    w, a = CFG.weights, CFG.anchors
    expected = (w.net_return * 0.001 / a.net_return - w.max_drawdown * 0.002 / a.max_drawdown
                - w.downside * 0.0005 / a.downside - w.turnover * 2.0 / a.turnover)
    assert r.score == pytest.approx(expected)


def test_high_return_never_offsets_a_drawdown_breach() -> None:
    huge = [m(ret=0.5, dd=0.05), m(ret=0.5), m(ret=0.5)]
    r = score(ev(windows=huge), CFG, 3)
    assert not r.eligible and r.score is None
    assert any("drawdown" in x for x in r.reasons)


@pytest.mark.parametrize(
    "evaluation, needle",
    [
        (ev(windows=[m(fills=1)] * 3), "insufficient evidence"),
        (ev(windows=[m(fills=40), m(fills=0), m(fills=0)]), "insufficient evidence"),
        (ev(violations=["bad"]), "illegal actions"),
        (ev(windows=[m(turn=500)] * 3), "turnover"),
        (ev(error="boom"), "invalid"),
        (ev(windows=[m(), m()]), "incomplete"),
    ],
)
def test_gates(evaluation, needle) -> None:
    r = score(evaluation, CFG, 3)
    assert not r.eligible and any(needle in x for x in r.reasons)


def test_instability_is_penalized() -> None:
    steady = score(ev(windows=[m(ret=0.001)] * 3), CFG, 3).score
    jumpy = score(ev(windows=[m(ret=-0.001), m(ret=0.001), m(ret=0.003)]), CFG, 3).score
    assert jumpy < steady


def test_fitness_fingerprint_changes_with_weights() -> None:
    other = FitnessConfig(weights={"net_return": 2.0})
    assert fitness_fingerprint(CFG) != fitness_fingerprint(other)


def _fit(scores: dict[str, float | None]) -> dict[str, FitnessResult]:
    return {k: FitnessResult(k, v is not None, v) for k, v in scores.items()}


def _rec(agent_id: str, family: str, genome=None) -> AgentRecord:
    return AgentRecord(agent_id, family, 0, genome or dict(FAMILY_CLASSES[family].default_genome))


def test_selection_skips_near_clones_and_respects_family_cap() -> None:
    recs = [_rec("m1", "momentum"), _rec("m2", "momentum"), _rec("r1", "mean_reversion"),
            _rec("o1", "order_flow"), _rec("l1", "liquidity"), _rec("x1", "regime_adaptive")]
    fit = _fit({"m1": 5, "m2": 4.9, "r1": 3, "o1": 2, "l1": 1, "x1": 0.5})
    sel = select_parents(recs, fit, 5, min_distance=0.15, max_per_family=2)
    ids = [p.agent_id for p in sel.parents]
    assert ids == ["m1", "r1", "o1", "l1", "x1"]  # m2 is a clone of m1
    assert any("m2" in n for n in sel.notes)


def test_selection_never_pads_with_ineligible() -> None:
    recs = [_rec("a", "momentum"), _rec("b", "liquidity"), _rec("c", "order_flow")]
    sel = select_parents(recs, _fit({"a": 1, "b": None, "c": None}), 5, 0.15, 2)
    assert [p.agent_id for p in sel.parents] == ["a"]
    assert "only 1 eligible" in sel.notes[-1]


def test_selection_relaxes_constraints_among_eligible_only() -> None:
    recs = [_rec(f"m{i}", "momentum") for i in range(5)]
    sel = select_parents(recs, _fit({f"m{i}": 5 - i for i in range(5)}), 5, 0.15, 2)
    assert len(sel.parents) == 5
    assert any("relaxed" in n for n in sel.notes)


def test_offspring_mix_and_lineage() -> None:
    rng = np.random.default_rng(0)
    from atlas.config import FAMILY_NAMES

    parents = initial_population(list(FAMILY_NAMES), 50, rng)[::10]
    cfg = OffspringConfig()
    kids = make_offspring(parents, 1, cfg, list(FAMILY_NAMES), rng)
    assert len(kids) == 50
    assert len({k.agent_id for k in kids}) == 50
    origins = [k.origin for k in kids]
    assert origins.count("elite") == 5
    assert origins.count("immigrant") == 5
    assert origins.count("large_mutation") == 10
    assert origins.count("small_mutation") + origins.count("crossover") == 30
    by_id = {p.agent_id: p for p in parents}
    for k in kids:
        if k.origin == "elite":
            assert k.genome == by_id[k.parents[0]].genome
        if k.origin in ("small_mutation", "large_mutation", "crossover"):
            assert k.mutations and all(by_id[p].family == k.family for p in k.parents)


def test_archives_keep_best_and_novel() -> None:
    recs = [_rec("a", "momentum"), _rec("b", "liquidity"), _rec("c", "order_flow")]
    arch = Archives(champion_size=2, novelty_size=10, novelty_threshold=0.5)
    arch.update(recs, _fit({"a": 3, "b": 1, "c": None}), generation=0)
    assert [e.record.agent_id for e in arch.champions] == ["a", "b"]
    assert {e.record.agent_id for e in arch.novelty} == {"a", "b"}
    # An elite copy (same genome) is not re-archived as a new champion.
    arch.update([_rec("a2", "momentum")], _fit({"a2": 3}), generation=1)
    assert [e.record.agent_id for e in arch.champions] == ["a", "b"]


def test_buy_hold_benchmark_scores_excess_return_and_keeps_old_fingerprints() -> None:
    plain = FitnessConfig()
    vs_hold = FitnessConfig(benchmark="buy_hold", drawdown_stat="mean")
    agent = ev(windows=[m(ret=0.004, dd=0.002), m(ret=0.002, dd=0.010), m(ret=0.003, dd=0.003)])
    hold = (0.003, 0.003, 0.003)
    r = score(agent, vs_hold, 3, hold)
    assert r.aggregate.mean_net_return == pytest.approx(0.0)  # exactly matches Hold on average
    assert r.aggregate.worst_drawdown == pytest.approx(0.005)  # mean of window drawdowns
    assert score(agent, plain, 3).aggregate.mean_net_return == pytest.approx(0.003)
    with pytest.raises(ValueError):
        score(agent, vs_hold, 3)  # benchmark returns are required
    # Configs that don't use the new options hash exactly as before.
    assert "benchmark" not in plain.model_dump() and fitness_fingerprint(plain) != fitness_fingerprint(vs_hold)
