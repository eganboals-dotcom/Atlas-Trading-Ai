from __future__ import annotations

import math

import numpy as np
import pytest
from helpers import ScriptedStrategy, data_config, quote, sim_config, trade, ts

from atlas.agents.adapter import AgentStrategy
from atlas.agents.base import Agent, AgentRecord, AgentRuntime, Observation
from atlas.agents.baselines import BASELINE_CLASSES, baseline_records
from atlas.agents.features import RETURN_HORIZONS_S, FeatureEngine
from atlas.agents.genome import (
    ParamSpec, crossover, genome_distance, mutate_genome, sample_genome, validate_genome,
)
from atlas.agents.registry import FAMILY_CLASSES, create_agent
from atlas.config import AgentLimits
from atlas.data.synthetic import generate_events
from atlas.data.validation import validate_events
from atlas.environment.market_state import MarketView, PortfolioState
from atlas.environment.replay import ReplayEngine
from atlas.evolution.reproduction import initial_population
from atlas.timeutils import NS_PER_MS

LIMITS = AgentLimits()


# ------------------------------------------------------------------ genome


def test_param_spec_roundtrip_and_bounds() -> None:
    spec = ParamSpec("float", 0.1, 100.0, log=True)
    for v in (0.1, 1.0, 37.5, 100.0):
        assert spec.from_unit(spec.to_unit(v)) == pytest.approx(v)
    assert spec.from_unit(2.0) == pytest.approx(100.0)
    assert ParamSpec("int", 1, 10).from_unit(0.5) in range(1, 11)
    with pytest.raises(ValueError):
        ParamSpec("float", 5, 1)


@pytest.mark.parametrize("family", sorted(FAMILY_CLASSES))
def test_family_defaults_and_samples_are_valid(family: str) -> None:
    cls = FAMILY_CLASSES[family]
    validate_genome(cls.default_genome, cls.param_space)
    rng = np.random.default_rng(0)
    for _ in range(50):
        g = sample_genome(cls.param_space, rng)
        validate_genome(g, cls.param_space)
        child, changes = mutate_genome(g, cls.param_space, rng, scale=0.5, rate=0.5)
        validate_genome(child, cls.param_space)
        assert changes


def test_mutation_is_seeded() -> None:
    space = FAMILY_CLASSES["momentum"].param_space
    g = FAMILY_CLASSES["momentum"].default_genome
    a = mutate_genome(g, space, np.random.default_rng(5), 0.1, 0.3)
    b = mutate_genome(g, space, np.random.default_rng(5), 0.1, 0.3)
    assert a == b


def test_crossover_and_distance() -> None:
    space = FAMILY_CLASSES["order_flow"].param_space
    rng = np.random.default_rng(1)
    a, b = sample_genome(space, rng), sample_genome(space, rng)
    child = crossover(a, b, rng)
    assert all(child[k] in (a[k], b[k]) for k in child)
    assert genome_distance(a, a, space) == 0.0
    assert 0.0 < genome_distance(a, b, space) <= 1.0
    with pytest.raises(ValueError):
        crossover(a, FAMILY_CLASSES["momentum"].default_genome, rng)


# ------------------------------------------------------------------ population


def test_initial_population_is_50_with_10_per_family() -> None:
    pop = initial_population(list(FAMILY_CLASSES), 50, np.random.default_rng(0))
    assert len(pop) == 50
    assert len({r.agent_id for r in pop}) == 50
    for family in FAMILY_CLASSES:
        assert sum(r.family == family for r in pop) == 10


def test_baselines_are_not_evolvable_families() -> None:
    assert not set(BASELINE_CLASSES) & set(FAMILY_CLASSES)
    assert all(r.origin == "baseline" for r in baseline_records())
    assert len(baseline_records()) == 5


def test_record_family_must_match_agent() -> None:
    rec = AgentRecord("x", "momentum", 0, dict(FAMILY_CLASSES["momentum"].default_genome))
    with pytest.raises(ValueError):
        FAMILY_CLASSES["liquidity"](rec)
    bad = AgentRecord("x", "momentum", 0, {"size_frac": 0.2})
    with pytest.raises(ValueError):
        create_agent(bad)


def test_checkpoint_roundtrip(tmp_path) -> None:
    rec = initial_population(["mean_reversion"], 2, np.random.default_rng(3))[1]
    agent = create_agent(rec)
    path = tmp_path / "agent.json"
    agent.save_checkpoint(path)
    loaded = Agent.load_checkpoint(path)
    assert loaded.metadata() == agent.metadata()
    assert dict(loaded.genome) == rec.genome


def test_rule_agents_do_not_learn() -> None:
    agent = create_agent(initial_population(["momentum"], 1, np.random.default_rng(0))[0])
    with pytest.raises(NotImplementedError):
        agent.learn([])


def test_metadata_fields() -> None:
    meta = create_agent(baseline_records()[0]).metadata()
    assert meta.agent_id == "baseline-cash" and meta.trainable is False
    assert meta.population_version == "v1-non-rl"


# ------------------------------------------------------------------ features


def _feed(events, symbols=("SPY",), stale_ms=1_000):
    view = MarketView(list(symbols), stale_ms * NS_PER_MS)
    fe = FeatureEngine(list(symbols), stale_ms * NS_PER_MS)
    for e in events:
        view._advance(e.event_ts)
        if e.event_type == "quote":
            view._receive_quote(e)
        else:
            view._receive_trade(e)
        fe.update(view)
    return fe, view


def test_returns_need_history_then_match_mid_change() -> None:
    fe, view = _feed([quote(0, 100.00, 100.02), quote(600, 100.10, 100.12)])
    f = fe.snapshot(view.now_ns)["SPY"]
    expected = math.log(100.11 / 100.01) * 10_000
    assert f.returns_bps[0.5] == pytest.approx(expected)
    assert math.isnan(f.returns_bps[5.0])  # not enough history yet


def test_trade_signing_and_flow_imbalance() -> None:
    events = [quote(0), trade(1, price=100.02, size=30), trade(2, price=100.00, size=10), trade(3, price=100.015, size=20)]
    fe, view = _feed(events)
    f = fe.snapshot(view.now_ns)["SPY"]
    assert f.trade_volume[1.0] == 60
    assert f.flow_imbalance[1.0] == pytest.approx((30 - 10 + 20) / 60)


def test_flow_window_expires() -> None:
    fe, _ = _feed([quote(0), trade(1, price=100.02, size=30), quote(1_500)])
    f = fe.snapshot(ts(1_500))["SPY"]
    assert f.trade_volume[1.0] == 0 and f.trade_volume[5.0] == 30


def test_l1_imbalance_and_staleness() -> None:
    fe, _ = _feed([quote(0, bid_size=300, ask_size=100)])
    assert fe.snapshot(ts(0))["SPY"].l1_imbalance == pytest.approx(0.5)
    assert fe.snapshot(ts(0))["SPY"].valid
    assert not fe.snapshot(ts(5_000))["SPY"].valid


def test_feature_snapshot_without_quotes_is_invalid() -> None:
    fe = FeatureEngine(["SPY"], 1_000 * NS_PER_MS)
    f = fe.snapshot(ts(0))["SPY"]
    assert not f.valid and all(math.isnan(f.returns_bps[h]) for h in RETURN_HORIZONS_S)


# ------------------------------------------------------------------ adapter


class _Fixed(Agent):
    family = "baseline_cash"  # reuse an empty param space
    param_space = {}
    default_genome = {}

    def __init__(self, targets) -> None:
        super().__init__(AgentRecord("t", "baseline_cash", 0, {}))
        self.targets = targets

    def act(self, obs: Observation):
        return dict(self.targets)


def _run_adapter(agent, events, **limits):
    lim = AgentLimits(**limits) if limits else LIMITS
    strat = AgentStrategy(agent, ["SPY"], lim, 100 * NS_PER_MS, 1_000 * NS_PER_MS, seed=0)
    result = ReplayEngine(sim_config(), ["SPY"]).run(events, strat)
    return strat, result


def test_adapter_records_illegal_targets() -> None:
    strat, result = _run_adapter(_Fixed({"SPY": -5, "TSLA": 10}), [quote(0)])
    assert result.orders == []
    assert any("illegal target" in v for v in strat.violations)
    assert any("unknown symbol" in v for v in strat.violations)


def test_adapter_clips_to_limits_and_order_size() -> None:
    events = [quote(t) for t in range(0, 2_000, 100)]
    strat, result = _run_adapter(_Fixed({"SPY": 10_000}), events, max_position_shares=700, max_order_shares=300,
                                 max_position_frac=1.0)
    assert strat.clipped > 0
    assert max(f.quantity for f in result.fills) <= 300
    assert result.final["positions"]["SPY"]["quantity"] == 700


def test_adapter_caps_position_value_as_fraction_of_equity() -> None:
    events = [quote(t) for t in range(0, 2_000, 100)]
    _, result = _run_adapter(_Fixed({"SPY": 10_000}), events, max_position_frac=0.5)
    assert result.final["positions"]["SPY"]["quantity"] == int(0.5 * 100_000 / 100.02)


def test_adapter_throttles_decisions() -> None:
    events = [quote(t) for t in range(0, 1_000, 10)]  # 100 events over 1 s, 100 ms interval
    strat, _ = _run_adapter(_Fixed({"SPY": 0}), events)
    assert 9 <= strat.decisions <= 11


@pytest.mark.parametrize("family", sorted(FAMILY_CLASSES) + sorted(BASELINE_CLASSES))
def test_every_family_runs_and_reconciles(family: str) -> None:
    cls = FAMILY_CLASSES.get(family) or BASELINE_CLASSES[family]
    events = validate_events(generate_events(data_config()), 2_000).valid_events
    agent = cls(AgentRecord("t", family, 0, dict(cls.default_genome)))
    strat = AgentStrategy(agent, ["SPY", "QQQ"], LIMITS, 500 * NS_PER_MS, 2_000 * NS_PER_MS, seed=1)
    result = ReplayEngine(sim_config(stale_quote_ms=2_000.0), ["SPY", "QQQ"]).run(events, strat)
    assert strat.violations == []
    for f in result.fills:
        assert f.quantity <= LIMITS.max_order_shares
    for p in result.final["positions"].values():
        assert 0 <= p["quantity"] <= LIMITS.max_position_shares


def test_agents_do_not_share_state() -> None:
    """Two instances of the same record behave identically regardless of what ran before."""
    events = validate_events(generate_events(data_config()), 2_000).valid_events
    rec = AgentRecord("t", "regime_adaptive", 0, dict(FAMILY_CLASSES["regime_adaptive"].default_genome))

    def run():
        strat = AgentStrategy(create_agent(rec), ["SPY", "QQQ"], LIMITS, 500 * NS_PER_MS, 2_000 * NS_PER_MS, 1)
        return ReplayEngine(sim_config(stale_quote_ms=2_000.0), ["SPY", "QQQ"]).run(events, strat).digest

    first = run()
    other = AgentRecord("o", "momentum", 0, dict(FAMILY_CLASSES["momentum"].default_genome))
    AgentStrategy(create_agent(other), ["SPY", "QQQ"], LIMITS, 500 * NS_PER_MS, 2_000 * NS_PER_MS, 1)
    assert run() == first


def test_agent_observation_is_read_only() -> None:
    agent = create_agent(baseline_records()[0])
    agent.initialize(AgentRuntime(("SPY",), LIMITS), 0)
    obs = agent.observe({}, PortfolioState.of(1.0, 1.0, {"SPY": 3}), 0)
    with pytest.raises(TypeError):
        obs.positions["SPY"] = 5  # type: ignore[index]
