from __future__ import annotations

from helpers import data_config, sim_config

from atlas.config import LatencyConfig, LatencyDistribution
from atlas.data.synthetic import generate_events
from atlas.simulation import dataset_hash, run_simulation

JITTERY = LatencyConfig(
    feed=LatencyDistribution(kind="uniform", low_ms=1, high_ms=5),
    decision=LatencyDistribution(kind="fixed", value_ms=1),
    order=LatencyDistribution(kind="lognormal", value_ms=20, sigma=0.5),
)


def sim(seed: int = 42):
    return sim_config(seed=seed, latency=JITTERY, demo_strategy={"hold_s": 10, "round_trips": 4})


def test_synthetic_generator_is_deterministic() -> None:
    assert generate_events(data_config()) == generate_events(data_config())


def test_different_data_seed_gives_different_data() -> None:
    other = data_config(synthetic={"seed": 4, "initial_prices": {"SPY": 580.0, "QQQ": 500.0}})
    assert dataset_hash(generate_events(data_config())) != dataset_hash(generate_events(other))


def test_same_data_config_seed_replays_identically() -> None:
    first, _, _ = run_simulation(sim(), data_config())
    second, _, _ = run_simulation(sim(), data_config())
    assert first.fills, "scenario should trade"
    assert first.digest == second.digest
    assert first.audit == second.audit
    assert first.equity_curve == second.equity_curve


def test_different_simulator_seed_changes_latency_draws() -> None:
    a, _, _ = run_simulation(sim(seed=1), data_config())
    b, _, _ = run_simulation(sim(seed=2), data_config())
    assert a.digest != b.digest


def test_no_state_leaks_between_runs() -> None:
    """Interleaving unrelated runs must not change a run's result."""
    baseline, _, _ = run_simulation(sim(), data_config())
    run_simulation(sim(seed=99), data_config(duration_s=60))
    again, _, _ = run_simulation(sim(), data_config())
    assert baseline.digest == again.digest
