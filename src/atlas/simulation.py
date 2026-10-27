"""High-level runner: generate/validate data, replay, and write a reproducible run record."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from atlas import __version__
from atlas.config import ConfigError, DataConfig, SimulatorConfig, load_config
from atlas.data.events import MarketEvent, event_to_dict
from atlas.data.synthetic import generate_events
from atlas.data.validation import DataValidationError, ValidationReport, validate_events
from atlas.environment.demo_strategy import BuyHoldSellDemo
from atlas.environment.replay import ReplayEngine, RunResult
from atlas.logging_utils import get_logger

log = get_logger(__name__)


def dataset_hash(events: list[MarketEvent]) -> str:
    h = hashlib.sha256()
    for e in events:
        h.update(json.dumps(event_to_dict(e), sort_keys=True).encode())
    return h.hexdigest()


def resolve_data_config(sim_cfg: SimulatorConfig, sim_path: Path) -> Path:
    """Look for ``data_config`` relative to the working directory, then the config's folder."""
    for candidate in (sim_cfg.data_config, sim_path.parent / sim_cfg.data_config):
        if candidate.is_file():
            return candidate
    raise ConfigError(f"data_config not found: {sim_cfg.data_config}")


def load_and_validate(data_cfg: DataConfig) -> tuple[list[MarketEvent], ValidationReport]:
    events = generate_events(data_cfg)
    report = validate_events(events, data_cfg.allowed_data_age_ms)
    log.info("data validated", extra={"dataset_hash": dataset_hash(events), **report.summary()})
    return events, report


def run_simulation(
    sim_cfg: SimulatorConfig,
    data_cfg: DataConfig,
    output_root: Path | None = None,
) -> tuple[RunResult, dict[str, Any], Path | None]:
    raw_events, report = load_and_validate(data_cfg)
    if sim_cfg.invalid_event_policy == "fail" and any(i.drops_event for i in report.issues):
        raise DataValidationError(report)

    symbol = sim_cfg.demo_strategy.symbol or data_cfg.symbols[0]
    if symbol not in data_cfg.symbols:
        raise ConfigError(f"demo_strategy.symbol {symbol!r} is not in data symbols {data_cfg.symbols}")

    engine = ReplayEngine(sim_cfg, data_cfg.symbols)
    result = engine.run(report.valid_events, BuyHoldSellDemo(sim_cfg.demo_strategy, symbol))

    summary: dict[str, Any] = {
        "atlas_version": __version__,
        "result_digest": result.digest,
        "dataset_hash": dataset_hash(raw_events),
        "data_provider": data_cfg.provider,
        "seed": sim_cfg.seed,
        "data_seed": data_cfg.synthetic.seed,
        "symbols": data_cfg.symbols,
        "validation": report.summary(),
        "orders": len(result.orders),
        "orders_by_status": _count(str(o.status) for o in result.orders),
        "fills": len(result.fills),
        "stats": result.stats,
        "portfolio": result.final,
        "disclaimer": "Synthetic plumbing demo. Results say nothing about real-market profitability.",
    }

    out_dir = None
    if output_root is not None:
        run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{result.digest[:8]}"
        out_dir = output_root / run_id
        _write_run(out_dir, run_id, sim_cfg, data_cfg, report, result, summary)
        summary["run_id"] = run_id
    return result, summary, out_dir


def _count(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


def _write_run(
    out_dir: Path,
    run_id: str,
    sim_cfg: SimulatorConfig,
    data_cfg: DataConfig,
    report: ValidationReport,
    result: RunResult,
    summary: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    snapshot = {"simulator": sim_cfg.model_dump(mode="json"), "data": data_cfg.model_dump(mode="json")}
    (out_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps({"run_id": run_id, **summary}, indent=2), encoding="utf-8")
    (out_dir / "validation_issues.json").write_text(
        json.dumps([{**asdict(i), "kind": str(i.kind)} for i in report.issues], indent=2), encoding="utf-8"
    )
    with (out_dir / "fills.jsonl").open("w", encoding="utf-8") as fh:
        for f in result.fills:
            fh.write(json.dumps({**asdict(f), "side": str(f.side)}) + "\n")
    with (out_dir / "audit.jsonl").open("w", encoding="utf-8") as fh:
        for rec in result.audit:
            fh.write(json.dumps(rec, default=str) + "\n")
    with (out_dir / "equity_curve.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ts_ns", "equity"])
        writer.writerows(result.equity_curve)
