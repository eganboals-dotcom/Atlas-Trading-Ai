"""Command-line entry point: ``python -m atlas.cli <command> --config <path>``.

Commands: ``validate-data``, ``simulate`` (Phase 1) and ``evolve`` (Phase 2). Later
phases add evaluate, paper, and report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from atlas.config import ConfigError, DataConfig, SimulatorConfig, load_config
from atlas.data.validation import DataValidationError
from atlas.logging_utils import configure_logging, get_logger

log = get_logger(__name__)

EXIT_OK, EXIT_DATA_INVALID, EXIT_CONFIG = 0, 1, 2


def _cmd_validate_data(args: argparse.Namespace) -> int:
    from atlas.simulation import dataset_hash, load_and_validate

    cfg = load_config(args.config, DataConfig)
    events, report = load_and_validate(cfg)
    out = {"dataset_hash": dataset_hash(events), **report.summary()}
    if args.show_issues:
        out["issues"] = [
            {"kind": str(i.kind), "index": i.index, "symbol": i.symbol, "detail": i.detail}
            for i in report.issues[: args.show_issues]
        ]
    print(json.dumps(out, indent=2))
    if args.strict and not report.is_clean:
        return EXIT_DATA_INVALID
    return EXIT_OK


def _cmd_simulate(args: argparse.Namespace) -> int:
    from atlas.simulation import resolve_data_config, run_simulation

    sim_path = Path(args.config)
    sim_cfg = load_config(sim_path, SimulatorConfig)
    data_cfg = load_config(resolve_data_config(sim_cfg, sim_path), DataConfig)
    output_root = None if args.no_write else Path(args.output or sim_cfg.reports_dir)
    _, summary, out_dir = run_simulation(sim_cfg, data_cfg, output_root)
    if out_dir is not None:
        summary["output_dir"] = str(out_dir)
    print(json.dumps(summary, indent=2))
    return EXIT_OK


def _resolve(path: Path, relative_to: Path) -> Path:
    for candidate in (path, relative_to.parent / path):
        if candidate.is_file():
            return candidate
    raise ConfigError(f"referenced config not found: {path}")


def _cmd_evolve(args: argparse.Namespace) -> int:
    from atlas.config import EvolutionConfig
    from atlas.evolution.runner import CheckpointError, EvolutionRunner

    evo_path = Path(args.config)
    evo = load_config(evo_path, EvolutionConfig)
    if args.workers is not None:
        evo = evo.model_copy(update={"evaluation": evo.evaluation.model_copy(update={"workers": args.workers})})
    sim = load_config(_resolve(evo.simulator_config, evo_path), SimulatorConfig)
    data = load_config(_resolve(evo.data_config, evo_path), DataConfig)
    runner = EvolutionRunner(evo, sim, data)
    try:
        state = runner.run(generations=args.generations, resume=args.resume)
    except CheckpointError as exc:
        print(f"checkpoint error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    print(json.dumps({
        "experiment": evo.experiment_name,
        "generations_completed": state.generation,
        "stopped": state.stopped,
        "history": state.history,
        "reports": str(runner.report_dir),
        "checkpoints": str(runner.ckpt_dir),
    }, indent=2))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlas", description="Project Atlas research CLI (simulation only).")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate-data", help="Generate/load data and validate schema, timestamps, and gaps.")
    p.add_argument("--config", required=True)
    p.add_argument("--strict", action="store_true", help="exit 1 if any issue is found")
    p.add_argument("--show-issues", type=int, default=20, metavar="N", help="print the first N issues")
    p.set_defaults(func=_cmd_validate_data)

    p = sub.add_parser("simulate", help="Run a deterministic synthetic replay with the demo strategy.")
    p.add_argument("--config", required=True)
    p.add_argument("--output", help="override reports directory")
    p.add_argument("--no-write", action="store_true", help="do not write run artifacts")
    p.set_defaults(func=_cmd_simulate)

    p = sub.add_parser("evolve", help="Run evolutionary generations of the 50-agent population.")
    p.add_argument("--config", required=True)
    p.add_argument("--generations", type=int, help="total generations to reach (default: config value)")
    p.add_argument("--resume", action="store_true", help="continue from the latest checkpoint")
    p.add_argument("--workers", type=int, help="parallel evaluation processes (0 = auto)")
    p.set_defaults(func=_cmd_evolve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        return args.func(args)
    except ConfigError as exc:
        log.error("configuration error", extra={"error": str(exc)})
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except DataValidationError as exc:
        log.error("data validation failed", extra=exc.report.summary())
        print(f"data validation failed: {exc}", file=sys.stderr)
        return EXIT_DATA_INVALID


if __name__ == "__main__":
    sys.exit(main())
