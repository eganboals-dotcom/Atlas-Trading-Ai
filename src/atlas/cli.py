"""Command-line entry point: ``python -m atlas.cli <command> --config <path>``.

Commands: ``validate-data``, ``simulate`` (Phase 1); ``evolve`` (Phase 2);
``export-synthetic``, ``import-data``, ``walkforward``, ``evaluate``, ``report`` (Phase 3).
Paper trading arrives in Phase 5.
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
    if args.dataset:
        return _validate_dataset(args)
    if not args.config:
        raise ConfigError("validate-data needs --config (synthetic DataConfig) or --dataset (imported dataset dir)")
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


def _validate_dataset(args: argparse.Namespace) -> int:
    """Check every session file against its manifest hash, then validate its events."""
    from atlas.data.providers import ParquetProvider, load_raw
    from atlas.data.validation import validate_events

    provider = ParquetProvider(args.dataset)
    sessions, clean = [], True
    for d in provider.sessions():
        ref = provider.session_ref(d)
        try:
            report = validate_events(load_raw(ref), ref.allowed_data_age_ms)
        except ValueError as exc:  # hash mismatch or unreadable file
            sessions.append({"session": d.isoformat(), "error": str(exc)})
            clean = False
            continue
        clean = clean and report.is_clean
        sessions.append({"session": d.isoformat(), "symbols": list(ref.symbols), **report.summary()})
    print(json.dumps({"dataset": provider.describe(), "sessions": sessions}, indent=2, default=str))
    if any("error" in s for s in sessions) or (args.strict and not clean):
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


def _cmd_export_synthetic(args: argparse.Namespace) -> int:
    from datetime import date

    from atlas.data.ingest import export_synthetic_raw
    from atlas.data.providers import SyntheticProvider

    base = load_config(args.config, DataConfig)
    provider = SyntheticProvider(base, date.fromisoformat(args.start), args.sessions, args.duration_s,
                                 args.symbols)
    paths = export_synthetic_raw(provider, Path(args.out))
    print(json.dumps({"raw_files": [str(p) for p in paths], "sessions": len(provider.sessions())}, indent=2))
    return EXIT_OK


def _cmd_import_data(args: argparse.Namespace) -> int:
    import yaml

    from atlas.data.ingest import DataImportError, ImportConfig, import_dataset

    path = Path(args.config)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        cfg = ImportConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except ValueError as exc:
        raise ConfigError(f"{path}: invalid import config:\n{exc}") from exc
    try:
        manifest = import_dataset(cfg)
    except DataImportError as exc:
        print(f"import error: {exc}", file=sys.stderr)
        return EXIT_DATA_INVALID
    print(json.dumps({
        "dataset": manifest["dataset"],
        "output": str(cfg.output_dir / cfg.dataset),
        "sessions": [{k: s[k] for k in ("date", "rows", "symbols")} | {"issues": s["validation"]["issue_counts"]}
                     for s in manifest["sessions"]],
    }, indent=2))
    return EXIT_OK


def _load_walkforward(path_str: str, workers: int | None):
    import yaml

    from atlas.config import EvolutionConfig
    from atlas.data.providers import make_provider
    from atlas.evaluation.walkforward import WalkForwardConfig

    path = Path(path_str)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        wf = WalkForwardConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except ValueError as exc:
        raise ConfigError(f"{path}: invalid walk-forward config:\n{exc}") from exc
    evo_path = _resolve(wf.evolution_config, path)
    evo = load_config(evo_path, EvolutionConfig)
    if workers is not None:
        evo = evo.model_copy(update={"evaluation": evo.evaluation.model_copy(update={"workers": workers})})
    sim = load_config(_resolve(evo.simulator_config, evo_path), SimulatorConfig)
    base = load_config(_resolve(evo.data_config, evo_path), DataConfig)
    return wf, evo, sim, base, make_provider(wf.data_source, path.parent)


def _cmd_walkforward(args: argparse.Namespace) -> int:
    from atlas.evaluation.walkforward import run_walkforward

    wf, evo, sim, base, provider = _load_walkforward(args.config, args.workers)
    report, run_dir = run_walkforward(wf, evo, sim, base, provider)
    print(json.dumps({
        "run_id": report["run_id"],
        "report": str(run_dir / "report.md"),
        "design_fingerprint": report["design_fingerprint"],
        "champion_oos": report["summary"]["champion_oos"],
        "selection_gap": report["summary"]["selection_gap"],
        "holdout": report["holdout"],
    }, indent=2))
    return EXIT_OK


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from atlas.evaluation.holdout import HoldoutAccessError
    from atlas.evaluation.walkforward import evaluate_checkpoint, evaluate_holdout

    wf, evo, sim, base, provider = _load_walkforward(args.config, args.workers)
    if args.checkpoint:
        if args.holdout:
            raise ConfigError("--holdout evaluates a frozen walk-forward design; use it with --run-id")
        out, out_dir = evaluate_checkpoint(Path(args.checkpoint), wf, evo, sim, provider, args.top,
                                           wf.reports_dir.parent / "evaluations")
        print(json.dumps({"output": str(out_dir), "evaluation_sessions": out["evaluation_sessions"]}, indent=2))
        return EXIT_OK
    if not args.run_id or not args.holdout:
        raise ConfigError("evaluate needs --checkpoint PATH, or --run-id ID --holdout")
    try:
        out = evaluate_holdout(wf.reports_dir / args.run_id, wf, evo, sim, base, provider,
                               allow_reuse=args.allow_holdout_reuse)
    except (HoldoutAccessError, ValueError) as exc:
        print(f"holdout evaluation refused: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    print(json.dumps({"report": str(wf.reports_dir / args.run_id / "report.md"),
                      "holdout_accesses_total": out["holdout_accesses_total"],
                      "holdout_reused": out["holdout_reused"], "champion_ci": out["champion_ci"]}, indent=2))
    return EXIT_OK


def _cmd_report(args: argparse.Namespace) -> int:
    from atlas.evaluation.report import write_markdown

    run_dir = Path(args.reports_dir) / args.run_id
    if not (run_dir / "report.json").is_file():
        raise ConfigError(f"no report.json in {run_dir}")
    out = write_markdown(run_dir)
    print(out.read_text(encoding="utf-8"))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlas", description="Project Atlas research CLI (simulation only).")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate-data", help="Validate data schema, timestamps, gaps, and manifest hashes.")
    p.add_argument("--config", help="synthetic data config (DataConfig YAML)")
    p.add_argument("--dataset", help="imported dataset directory containing manifest.json")
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

    p = sub.add_parser("export-synthetic", help="Write synthetic sessions as raw vendor-style CSVs.")
    p.add_argument("--config", required=True, help="DataConfig YAML used as the generator template")
    p.add_argument("--start", required=True, help="first session date, YYYY-MM-DD")
    p.add_argument("--sessions", type=int, required=True)
    p.add_argument("--duration-s", type=float)
    p.add_argument("--symbols", nargs="+")
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_export_synthetic)

    p = sub.add_parser("import-data", help="Import raw CSVs into normalized Parquet sessions plus a manifest.")
    p.add_argument("--config", required=True)
    p.set_defaults(func=_cmd_import_data)

    p = sub.add_parser("walkforward", help="Nested walk-forward evaluation (the outer holdout stays untouched).")
    p.add_argument("--config", required=True)
    p.add_argument("--workers", type=int)
    p.set_defaults(func=_cmd_walkforward)

    p = sub.add_parser("evaluate", help="Stress-test a checkpoint, or evaluate a frozen design on the holdout.")
    p.add_argument("--config", required=True, help="walk-forward config")
    p.add_argument("--checkpoint", help="evolution checkpoint to stress-test on later sessions")
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--run-id", help="walk-forward run whose frozen design to evaluate")
    p.add_argument("--holdout", action="store_true", help="evaluate on the outer holdout (recorded in the ledger)")
    p.add_argument("--allow-holdout-reuse", action="store_true")
    p.add_argument("--workers", type=int)
    p.set_defaults(func=_cmd_evaluate)

    p = sub.add_parser("report", help="Rebuild and print the Markdown report for a walk-forward run.")
    p.add_argument("--run-id", required=True)
    p.add_argument("--reports-dir", default="artifacts/reports/walkforward")
    p.set_defaults(func=_cmd_report)
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
