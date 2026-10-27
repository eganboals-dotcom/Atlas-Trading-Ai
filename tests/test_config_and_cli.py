from __future__ import annotations

import json
from pathlib import Path

import pytest
from helpers import data_config, sim_config
from pydantic import ValidationError

from atlas.cli import EXIT_CONFIG, EXIT_DATA_INVALID, EXIT_OK, main
from atlas.config import ConfigError, DataConfig, SimulatorConfig, load_config

REPO = Path(__file__).resolve().parents[1]


def test_shipped_configs_load() -> None:
    sim = load_config(REPO / "configs/simulator.yaml", SimulatorConfig)
    data = load_config(REPO / "configs/data.yaml", DataConfig)
    assert data.symbols == ["SPY", "QQQ", "IWM"]
    assert sim.invalid_event_policy == "drop"


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        sim_config(slipage={"bps": 1})  # typo must not silently fall back to a default


@pytest.mark.parametrize(
    "override",
    [{"initial_cash": 0}, {"fees": {"per_share": -0.01}}, {"fill": {"max_participation": 1.5}},
     {"latency": {"order": {"kind": "uniform", "low_ms": 5, "high_ms": 1}}}],
)
def test_invalid_simulator_values_are_rejected(override) -> None:
    with pytest.raises(ValidationError):
        sim_config(**override)


@pytest.mark.parametrize(
    "override",
    [{"symbols": []}, {"symbols": ["SPY", "spy"]}, {"session_date": "2026-10-24"},
     {"timezone": "Mars/Olympus"}, {"provider": "schwab"}],
)
def test_invalid_data_values_are_rejected(override) -> None:
    with pytest.raises(ValidationError):
        data_config(**override)


def test_load_config_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml", DataConfig)
    bad = tmp_path / "bad.yaml"
    bad.write_text("provider: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(bad, DataConfig)


def _write_small_configs(tmp_path: Path) -> Path:
    data = tmp_path / "data.yaml"
    data.write_text(
        "provider: synthetic\nsymbols: [SPY]\nsession_date: 2026-10-20\nduration_s: 120\n", encoding="utf-8"
    )
    sim = tmp_path / "sim.yaml"
    sim.write_text(
        f"data_config: {data.name}\nreports_dir: {(tmp_path / 'reports').as_posix()}\n"
        "demo_strategy: {hold_s: 10, round_trips: 2}\n",
        encoding="utf-8",
    )
    return sim


def test_cli_simulate_writes_run_record(tmp_path: Path, capsys) -> None:
    assert main(["simulate", "--config", str(_write_small_configs(tmp_path))]) == EXIT_OK
    summary = json.loads(capsys.readouterr().out)
    out_dir = Path(summary["output_dir"])
    for name in ("summary.json", "config_snapshot.yaml", "fills.jsonl", "audit.jsonl",
                 "equity_curve.csv", "validation_issues.json"):
        assert (out_dir / name).is_file(), name
    assert summary["fills"] > 0


def test_cli_validate_data_strict_fails_on_anomalies(capsys) -> None:
    code = main(["validate-data", "--config", str(REPO / "configs/data_with_anomalies.yaml"), "--strict"])
    assert code == EXIT_DATA_INVALID
    assert json.loads(capsys.readouterr().out)["dropped_events"] > 0


def test_cli_fail_policy_stops_on_invalid_data(tmp_path: Path) -> None:
    sim = tmp_path / "sim.yaml"
    sim.write_text(
        f"data_config: {(REPO / 'configs/data_with_anomalies.yaml').as_posix()}\ninvalid_event_policy: fail\n",
        encoding="utf-8",
    )
    assert main(["simulate", "--config", str(sim), "--no-write"]) == EXIT_DATA_INVALID


def test_cli_bad_config_exits_with_config_error(tmp_path: Path) -> None:
    bad = tmp_path / "sim.yaml"
    bad.write_text("data_config: nope.yaml\nunknown_field: 1\n", encoding="utf-8")
    assert main(["simulate", "--config", str(bad)]) == EXIT_CONFIG
