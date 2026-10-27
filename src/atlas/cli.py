"""Command-line entry point: ``python -m atlas.cli <command> --config <path>``.

Commands: ``validate-data``, ``simulate`` (Phase 1); ``evolve`` (Phase 2);
``export-synthetic``, ``import-data``, ``walkforward``, ``evaluate``, ``report`` (Phase 3);
``train-ppo`` (Phase 4); ``paper``, ``paper-kill`` (Phase 5); ``readiness`` (Phase 6).
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

EXIT_OK, EXIT_DATA_INVALID, EXIT_CONFIG, EXIT_CRASH = 0, 1, 2, 3


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


def _cmd_train_ppo(args: argparse.Namespace) -> int:
    try:
        from atlas.learning.stability import seed_stability
    except ImportError as exc:
        raise ConfigError(f"PPO needs the rl extra (pip install -e .[rl]): {exc}") from exc

    wf, evo, sim, _, provider = _load_walkforward(args.config, args.workers)
    out, out_dir = seed_stability(wf, evo, sim, provider, args.fold, args.seeds, Path(args.output))
    print(json.dumps({k: out[k] for k in ("fold", "train", "validation", "seeds", "base_return_by_seed_bps",
                                          "base_return_spread_bps")} | {"output": str(out_dir)}, indent=2))
    return EXIT_OK


def _cmd_paper(args: argparse.Namespace) -> int:
    import yaml

    from atlas.brokers.base import LiveTradingDisabledError
    from atlas.paper.config import PaperConfig, RiskConfig
    from atlas.paper.feed import SimulatedCrash
    from atlas.paper.session import HealthCheckError, PaperSession

    path = Path(args.config)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        cfg = PaperConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except ValueError as exc:
        raise ConfigError(f"{path}: refusing to start paper mode, invalid or unsafe config:\n{exc}") from exc
    sim = load_config(_resolve(cfg.simulator_config, path), SimulatorConfig)
    risk = load_config(_resolve(cfg.risk_config, path), RiskConfig)
    session = PaperSession(cfg, sim, risk, base_dir=path.parent, resume=args.resume,
                           reset_kill_switch=args.reset_kill_switch)
    try:
        report = session.run()
    except (HealthCheckError, LiveTradingDisabledError) as exc:
        print(f"paper session refused to start: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except SimulatedCrash as exc:
        print(f"paper session crashed ({exc}); state was flushed periodically — rerun with --resume",
              file=sys.stderr)
        return EXIT_CRASH
    print(json.dumps({
        "session_dir": str(session.dir),
        "report": str(session.dir / "daily_report.md"),
        "kill_switch": report["kill_switch"],
        "accounts": {a["account"]: {"net_return_bps": a["net_return"] * 1e4, "fills": a["fills"],
                                    "halted": a["halted"]} for a in report["accounts"]},
    }, indent=2))
    return EXIT_OK


def _cmd_paper_kill(args: argparse.Namespace) -> int:
    from atlas.paper.risk import request_kill

    path = request_kill(Path(args.session_dir), args.reason)
    print(f"kill switch file written: {path}")
    return EXIT_OK


def _cmd_readiness(args: argparse.Namespace) -> int:
    import yaml

    from atlas.readiness import ReadinessConfig, build_review

    path = Path(args.config)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        cfg = ReadinessConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    except ValueError as exc:
        raise ConfigError(f"{path}: invalid readiness config:\n{exc}") from exc
    if args.skip_drills or args.skip_tests:
        cfg = cfg.model_copy(update={"run_drills": cfg.run_drills and not args.skip_drills,
                                     "run_tests": cfg.run_tests and not args.skip_tests})
    review, out_dir = build_review(cfg)
    print(json.dumps({"verdict": review["verdict"], "gate_counts": review["gate_counts"],
                      "report": str(out_dir / "readiness_report.md"),
                      "docs": str(cfg.docs_path) if cfg.docs_path else None}, indent=2))
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

    p = sub.add_parser("train-ppo", help="Train PPO with several seeds on one fold and compare them out of sample.")
    p.add_argument("--config", required=True, help="walk-forward config with a `ppo:` section")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--output", default="artifacts/ppo")
    p.add_argument("--workers", type=int)
    p.set_defaults(func=_cmd_train_ppo)

    p = sub.add_parser("paper", help="Run a paper-trading session (simulated orders only; live orders impossible).")
    p.add_argument("--config", required=True)
    p.add_argument("--resume", action="store_true", help="continue a session after a crash or stop")
    p.add_argument("--reset-kill-switch", action="store_true", help="operator reset after reviewing a kill")
    p.set_defaults(func=_cmd_paper)

    p = sub.add_parser("paper-kill", help="Engage the kill switch of a paper session.")
    p.add_argument("--session-dir", required=True)
    p.add_argument("--reason", default="operator kill")
    p.set_defaults(func=_cmd_paper_kill)

    p = sub.add_parser("readiness", help="Readiness review: evidence, failure drills, gates (enables nothing).")
    p.add_argument("--config", required=True)
    p.add_argument("--skip-drills", action="store_true")
    p.add_argument("--skip-tests", action="store_true")
    p.set_defaults(func=_cmd_readiness)

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
