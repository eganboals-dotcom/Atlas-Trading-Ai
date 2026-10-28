"""Command-line entry point: ``python -m atlas.cli <command> --config <path>``.

Commands: ``validate-data``, ``simulate`` (Phase 1); ``evolve`` (Phase 2);
``export-synthetic``, ``import-data``, ``walkforward``, ``evaluate``, ``report`` (Phase 3);
``train-ppo`` (Phase 4); ``paper``, ``paper-check``, ``paper-window``, ``paper-watch``, ``paper-stop``, ``paper-kill``, ``dashboard``
(Phase 5); ``readiness`` (Phase 6).
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


def _load_paper(config: str):
    import yaml

    from atlas.paper.config import PaperConfig, RiskConfig

    path = Path(config)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        cfg = PaperConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except ValueError as exc:
        raise ConfigError(f"{path}: refusing to start paper mode, invalid or unsafe config:\n{exc}") from exc
    sim = load_config(_resolve(cfg.simulator_config, path), SimulatorConfig)
    risk = load_config(_resolve(cfg.risk_config, path), RiskConfig)
    return path, cfg, sim, risk


def _session_dir(args: argparse.Namespace) -> Path:
    """--session-dir, or today's session directory derived from --config."""
    if getattr(args, "session_dir", None):
        return Path(args.session_dir)
    if not getattr(args, "config", None):
        raise ConfigError("give --session-dir or --config")
    from atlas.paper.session import session_date_for

    _, cfg, _, _ = _load_paper(args.config)
    day = session_date_for(cfg, None)
    return cfg.output_dir / cfg.session_name.replace("{date}", day.isoformat())


def _cmd_paper(args: argparse.Namespace) -> int:
    from atlas.brokers.base import LiveTradingDisabledError
    from atlas.paper.feed import SimulatedCrash
    from atlas.paper.session import HealthCheckError, PaperSession

    path, cfg, sim, risk = _load_paper(args.config)
    session = PaperSession(cfg, sim, risk, base_dir=path.parent, resume=args.resume,
                           reset_kill_switch=args.reset_kill_switch, reset_fund=args.reset_fund)
    if args.auto_resume and (session.dir / "state.json").exists():
        session.resume = True  # same day, same config: continue the session instead of refusing

    def on_ready(s: PaperSession) -> None:
        print(f"paper session {s.dir.name}: startup checks passed; feed={cfg.feed.kind}; "
              f"{len(s.accounts)} accounts; simulated orders only", file=sys.stderr)
        mode = "terminal" if args.watch else cfg.dashboard.window
        if mode == "app":
            from atlas.dashboard.window import open_app_window, window_available

            if window_available():
                open_app_window(s.dir)
                print("leaderboard: opened in its own window (read-only)", file=sys.stderr)
            else:
                mode = "terminal"
                print("leaderboard window needs pywebview (pip install -e .[window]); using the terminal view",
                      file=sys.stderr)
        if mode == "terminal":
            from atlas.dashboard.terminal import open_window

            open_window(s.dir)
            print("leaderboard: opened in a new terminal window (read-only)", file=sys.stderr)
        if args.open_dashboard or cfg.dashboard.open_on_start:
            from atlas.dashboard.launch import launch

            url = launch(cfg.output_dir, s.dir.name, cfg.dashboard.port)
            print(f"live page: {url} (read-only; keeps running after the session ends)", file=sys.stderr)
        if cfg.feed.kind == "alpaca" and cfg.feed.alpaca is not None:
            from datetime import datetime

            from atlas.timeutils import MARKET_TZ

            now = datetime.now(MARKET_TZ).time()
            if now >= cfg.feed.alpaca.end_time:
                print("the market is already closed for today: this session ends immediately. "
                      "Start it tomorrow before 09:30 ET.", file=sys.stderr)
            else:
                when = "trading until" if now >= cfg.feed.alpaca.start_time else \
                    f"waiting for {cfg.feed.alpaca.start_time:%H:%M} ET, then trading until"
                print(f"{when} {cfg.feed.alpaca.end_time:%H:%M} ET. Keep this window open. Stop cleanly with Ctrl+C "
                      f"or `python -m atlas.cli paper-stop --config {path.as_posix()}`", file=sys.stderr)

    try:
        report = session.run(on_ready=on_ready)
    except (HealthCheckError, LiveTradingDisabledError) as exc:
        print(f"paper session refused to start: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except SimulatedCrash as exc:
        print(f"paper session crashed ({exc}); state was flushed periodically - rerun with --resume",
              file=sys.stderr)
        return EXIT_CRASH
    print(json.dumps({
        "session_dir": str(session.dir),
        "report": str(session.dir / "daily_report.md"),
        "stopped_by": report.get("stopped_by"),
        "kill_switch": report["kill_switch"],
        "accounts": {a["account"]: {"net_return_bps": a["net_return"] * 1e4, "fills": a["fills"],
                                    "halted": a["halted"]} for a in report["accounts"]},
    }, indent=2))
    return EXIT_OK


def _cmd_dashboard(args: argparse.Namespace) -> int:
    import time

    from atlas.dashboard.launch import launch

    url = launch(Path(args.paper_dir), None, args.port)
    print(f"live page: {url} (read-only)")
    time.sleep(5)  # let the browser open before this command exits
    return EXIT_OK


def _cmd_paper_kill(args: argparse.Namespace) -> int:
    from atlas.paper.risk import request_kill

    path = request_kill(_session_dir(args), args.reason)
    print(f"kill switch file written: {path} (no new orders; reset needed before restarting)")
    return EXIT_OK


def _cmd_paper_status(args: argparse.Namespace) -> int:
    from datetime import UTC, datetime

    from atlas.dashboard import data

    d = _session_dir(args)
    live = data.load_live(d)
    if not live:
        print(f"no live snapshot in {d} (session not started yet?)")
        return EXIT_DATA_INVALID
    age = (datetime.now(UTC) - datetime.fromisoformat(live["updated_wall"])).total_seconds()
    print(f"{live['session']} | status: {live['status']} | updated {age:.0f}s ago | feed {live['feed']} | "
          f"safe: {', '.join(live['safe']) or 'no'} | kill: {live['kill'] or 'no'} | simulated orders only")
    print(f"{'#':>2}  {'bot':<10} {'model':<38} {'P&L $':>10} {'return':>12} {'fills':>6}  status / positions")
    for r in data.leaderboard(live):
        rank = str(r["rank"]) if r["rank"] else "-"
        model = f"{r['agent_id']} ({r['family']})"[:38]
        pos = ", ".join(f"{s} {q}" for s, q in r["positions"].items()) or "flat"
        status = f"HALTED: {r['halted']}" if r["halted"] else pos
        print(f"{rank:>2}  {r['account']:<10} {model:<38} {r['net_pnl']:>10.2f} "
              f"{r['net_return'] * 1e4:>+9.2f}bps {r['fills']:>6}  {status}")
    return EXIT_OK


def _cmd_paper_window(args: argparse.Namespace) -> int:
    from atlas.dashboard.window import run_window, window_available

    if not window_available():
        raise ConfigError("the leaderboard window needs pywebview: pip install -e .[window]")
    return run_window(_session_dir(args))


def _cmd_paper_watch(args: argparse.Namespace) -> int:
    from atlas.dashboard.terminal import watch

    return watch(_session_dir(args), args.interval, args.once)


def _cmd_paper_stop(args: argparse.Namespace) -> int:
    from atlas.paper.session import STOP_FILE

    d = _session_dir(args)
    if not d.is_dir():
        raise ConfigError(f"no session directory {d}")
    (d / STOP_FILE).write_text("stop requested", encoding="utf-8")
    print(f"stop requested for {d.name}: the session finishes cleanly within a few seconds "
          "(cancel working orders, save state, write the daily report). Continue later with --resume.")
    return EXIT_OK


def _cmd_paper_check(args: argparse.Namespace) -> int:
    """Preflight: run every startup check against the real feed, without trading or writing a session."""
    import shutil
    import tempfile
    from datetime import datetime

    from atlas.paper.session import HealthCheckError, PaperSession, today_market_date
    from atlas.timeutils import MARKET_TZ

    path, cfg, sim, risk = _load_paper(args.config)
    checks: list[tuple[str, bool, str]] = [("config", True, f"{path} valid; mode=paper, live orders disabled")]
    tmp = Path(tempfile.mkdtemp(prefix="atlas-check-"))
    try:
        update: dict = {"output_dir": tmp, "alerts": cfg.alerts.model_copy(update={"console": False})}
        if cfg.fund is not None:  # check against the real ledger (read-only: the preflight never settles)
            update["fund"] = cfg.fund.model_copy(update={"ledger": cfg.fund.ledger or cfg.output_dir / "fund_ledger.json"})
        probe = cfg.model_copy(update=update)
        session = PaperSession(probe, sim, risk, base_dir=path.parent)
        try:
            session.health_checks()
            feed = session.feed
            detail = f"{cfg.feed.kind} feed connected; symbols {', '.join(session.symbols)}"
            quotes = getattr(feed, "connect_quotes", None)
            if quotes:
                detail += "; latest " + "; ".join(f"{s} {q['bid']}/{q['ask']}" for s, q in quotes.items())
            checks.append(("market data + keys", True, detail))
            offset = getattr(feed, "clock_offset_ms", None)
            if offset is not None and offset < -60_000:
                checks.append(("clock", True, f"latest quote is {-offset / 60_000:.0f} min old (market closed), so "
                               "the PC clock can't be verified now; every quote is checked against it during the "
                               "session. Sync it anyway: Settings > Time & language > Sync now"))
            elif offset is not None:
                checks.append(("clock", True, f"latest exchange timestamp is {offset:+.0f} ms vs this PC "
                               f"(limit +{risk.max_clock_skew_ms:.0f} ms)"))
            checks.append(("bots", True, ", ".join(f"{a.account_id}={a.model['agent_id']} ({a.model['family']})"
                                                   for a in session.accounts)))
            session.audit.close()
        except HealthCheckError as exc:
            checks.append(("startup checks", False, str(exc)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if cfg.dashboard.window == "app":
        from atlas.dashboard.window import window_available

        checks.append(("leaderboard window", True, "native window ready" if window_available() else
                       "pywebview missing (pip install -e .[window]); the terminal view will be used instead"))
    elif cfg.dashboard.window == "terminal":
        checks.append(("leaderboard window", True, "terminal view (paper-watch)"))
    if cfg.dashboard.open_on_start:
        try:
            import plotly  # noqa: F401
            import streamlit  # noqa: F401

            checks.append(("browser page", True, f"dashboard on http://localhost:{cfg.dashboard.port}"))
        except ImportError as exc:
            checks.append(("browser page", False, f"install the dashboard extra: {exc}"))
    if cfg.record.enabled:
        target = cfg.record.root / "raw" / cfg.record.dataset
        try:
            target.mkdir(parents=True, exist_ok=True)
            checks.append(("recording", True, f"market data will be saved to {target} (local only, never committed)"))
        except OSError as exc:
            checks.append(("recording", True, f"cannot write {target} ({exc}); the session will trade without recording"))
    if cfg.fund is not None:
        import json as _json

        from atlas.paper.session import PERMANENT_BASELINES

        permanent = set(PERMANENT_BASELINES.values())
        listed = [s for s in cfg.shadows if not (s.source == "baseline" and s.baseline in permanent)]
        from atlas.paper.session import ROLES

        bots = 1 + len(listed) + sum(ROLES[acct] != "benchmark" for acct in PERMANENT_BASELINES)  # + Flip
        ledger = cfg.fund.ledger or cfg.output_dir / "fund_ledger.json"
        if ledger.is_file():
            days = _json.loads(ledger.read_text(encoding="utf-8")).get("days", [])
            last = days[-1] if days else None
            detail = (f"ledger has {len(days)} day(s); fund at ${last['fund_end']:,.2f} after {last['date']}"
                      if last else "ledger exists, no days settled yet")
        else:
            detail = (f"new fund: ${cfg.fund.total_capital:,.2f} across {bots} bots "
                      f"(${cfg.fund.total_capital / bots:,.2f} each), compounding daily")
        checks.append(("fund", True, detail))
    if cfg.feed.kind == "alpaca" and cfg.feed.alpaca is not None:
        now = datetime.now(MARKET_TZ)
        a = cfg.feed.alpaca
        day = today_market_date()
        if now.time() < a.start_time:
            when = f"market opens at {a.start_time:%H:%M} ET today ({day}); start the session any time before"
        elif now.time() < a.end_time:
            when = "market is open now; a session started now trades until " + f"{a.end_time:%H:%M} ET"
        else:
            when = "market is closed for today; start the session tomorrow before 09:30 ET"
        checks.append(("schedule", True, when))
    ok = all(c[1] for c in checks)
    for name, passed, detail in checks:
        print(f"[{'OK' if passed else 'FAIL'}] {name}: {detail}")
    print("READY: paper session can start." if ok else "NOT READY: fix the FAIL lines above.")
    return EXIT_OK if ok else EXIT_CONFIG


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


def _cmd_fetch_history(args: argparse.Namespace) -> int:
    """Download real Alpaca history (read-only) as polled training sessions; resumable, skips done days."""
    import random
    import time
    from datetime import date as _date

    from atlas.data.alpaca_history import download_window, publish_dataset_session, random_window_start, sample_days
    from atlas.feeds.alpaca import HISTORY_PATHS, http_fetch, load_credentials

    fetch = http_fetch(load_credentials(Path(".env")), timeout_s=30, allowed=HISTORY_PATHS)
    out = Path("data/processed") / args.dataset
    done = set()
    if (out / "manifest.json").is_file():
        done = {s["date"] for s in json.loads((out / "manifest.json").read_text(encoding="utf-8"))["sessions"]}
    days = sample_days(_date.fromisoformat(args.first), _date.fromisoformat(args.last), args.days, args.seed)
    meta = {"source": f"alpaca-{args.feed}", "feed": args.feed,
            "license_notes": "Alpaca historical market data, downloaded for personal research; check Alpaca's data "
                             "terms before sharing or redistributing",
            "timestamp_kind": "vendor",
            "timestamp_notes": "event_ts = Alpaca's timestamp; receive_ts = emulated poll time + 80 ms",
            "fidelity": f"{args.feed.upper()} Level 1 history, re-sampled like the live poller (latest quote every "
                        f"1 s, latest trade every 2 s); one {args.minutes}-minute window per day",
            "price_adjustment": "none: raw intraday prices (each session is simulated flat at its start)",
            "corporate_actions": [], "universe_source": "symbols passed to fetch-history",
            "chronology_notes": "events in emulated arrival order (by poll, then source time)",
            "order_scope": "symbol_type"}
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    gate, last = threading.Lock(), [0.0]

    def paced(path: str, params: dict[str, str]):  # all threads together stay under ~170 requests/min
        with gate:
            wait = last[0] + 60 / 170 - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            last[0] = time.monotonic()
        return fetch(path, params)

    publish = threading.Lock()

    def one(i: int, day):
        start = random_window_start(random.Random(f"{args.seed}-{day}"), args.minutes)
        t0 = time.monotonic()
        events, units = download_window(paced, args.symbols, day, start, args.minutes, args.feed, min_interval_s=0)
        if len({e.symbol for e in events if e.event_type == "quote"}) < len(args.symbols):  # holiday / no data
            return f"[{i}/{len(days)}] {day} skipped: no data"
        with publish:
            entry = publish_dataset_session(out, args.dataset, day, events, args.symbols, meta,
                                            {"window_start_et": start.strftime("%H:%M"),
                                             "window_minutes": args.minutes, "quote_size_unit": units})
        return f"[{i}/{len(days)}] {day} {start:%H:%M} ET: {entry['rows']} events in {time.monotonic() - t0:.0f}s"

    todo = [(i, d) for i, d in enumerate(days, 1) if d.isoformat() not in done]
    print(f"{len(days)} days sampled, {len(todo)} to download", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, i, d) for i, d in todo]
        for fut in as_completed(futures):
            try:
                print(fut.result(), flush=True)
            except Exception as exc:  # one bad day must not lose the rest; re-running retries it
                print(f"day failed: {type(exc).__name__}: {exc}", flush=True)
    return EXIT_OK


def _cmd_fetch_bars(args: argparse.Namespace) -> int:
    """Download real 1-minute bars (read-only) as full-day sessions for slow (minutes-to-hours) bots."""
    from datetime import date as _date

    from atlas.data.alpaca_history import bar_events, merge_bar_sessions, publish_dataset_session, regular_hours_days
    from atlas.feeds.alpaca import HISTORY_BARS_PATH, HISTORY_PATHS, http_fetch, iter_history, load_credentials

    fetch = http_fetch(load_credentials(Path(".env")), timeout_s=60, allowed=HISTORY_PATHS)
    by_symbol = {}
    for sym in args.symbols:
        bars = iter_history(fetch, HISTORY_BARS_PATH, sym, f"{args.first}T00:00:00Z", f"{args.last}T23:59:59Z",
                            args.feed, extra={"timeframe": "1Min", "adjustment": "raw"})
        by_symbol[sym] = regular_hours_days(bars)
        print(f"{sym}: {len(by_symbol[sym])} days", flush=True)
    out = Path("data/processed") / args.dataset
    meta = {"source": f"alpaca-{args.feed}-bars", "feed": args.feed,
            "license_notes": "Alpaca historical market data, downloaded for personal research; check Alpaca's data "
                             "terms before sharing or redistributing",
            "timestamp_kind": "vendor",
            "timestamp_notes": "event_ts = bar end (start + 60 s): a bar is only known once it closes",
            "fidelity": f"1-minute {args.feed.upper()} bars, regular hours; one trade (close, volume) and one quote "
                        f"per minute, with a modeled {args.spread_bps} bps spread around the close",
            "price_adjustment": "none: raw intraday prices (each session is simulated flat at the open)",
            "corporate_actions": [], "universe_source": "symbols passed to fetch-bars",
            "chronology_notes": "per minute: trades then quotes, symbols interleaved",
            "order_scope": "symbol_type", "spread_bps": args.spread_bps}
    days = sorted(set.intersection(*(set(d) for d in by_symbol.values())))
    full = [d for d in days if all(len(by_symbol[s][d]) >= args.min_bars for s in args.symbols)]
    for day in full:
        events = merge_bar_sessions({s: bar_events(by_symbol[s][day], s, args.spread_bps) for s in args.symbols})
        publish_dataset_session(out, args.dataset, day, events, args.symbols, meta,
                                {"bars": {s: len(by_symbol[s][day]) for s in args.symbols}})
    print(f"{len(full)} full sessions written to {out} ({len(days) - len(full)} short days skipped)", flush=True)
    return EXIT_OK


def _cmd_fetch_daily(args: argparse.Namespace) -> int:
    """Download split- and dividend-adjusted daily bars (read-only) for swing (multi-day) research."""
    from atlas.feeds.alpaca import HISTORY_BARS_PATH, HISTORY_PATHS, http_fetch, iter_history, load_credentials
    from atlas.swing.data import daily_rows, save_daily

    fetch = http_fetch(load_credentials(Path(".env")), timeout_s=60, allowed=HISTORY_PATHS)
    rows = {}
    for sym in args.symbols:  # ~1 request per symbol, paced well under the free plan's limit
        rows[sym] = daily_rows(iter_history(fetch, HISTORY_BARS_PATH, sym, f"{args.first}T00:00:00Z",
                                            f"{args.last}T23:59:59Z", args.feed, min_interval_s=1.0,
                                            extra={"timeframe": "1Day", "adjustment": "all"}))
        print(f"{sym}: {len(rows[sym])} days", flush=True)
    out = Path(args.out)
    save_daily(rows, out, {"source": f"alpaca-{args.feed}-1Day", "adjustment": "all (splits and dividends)",
                           "license_notes": "Alpaca historical market data, downloaded for personal research; check "
                                            "Alpaca's data terms before sharing or redistributing"})
    print(f"daily bars written to {out}", flush=True)
    return EXIT_OK


def _cmd_fetch_news(args: argparse.Namespace) -> int:
    """Download news headlines and timestamps (read-only) for swing research, one file per year (resumable)."""
    from atlas.feeds.alpaca import NEWS_PATHS, http_fetch, iter_news, load_credentials
    from atlas.swing.news import news_rows, save_news, write_manifest

    fetch = http_fetch(load_credentials(Path(".env")), timeout_s=60, allowed=NEWS_PATHS)
    out = Path(args.out)
    for year in range(int(args.first[:4]), int(args.last[:4]) + 1):
        path = out / f"{year}.parquet"
        if path.exists():
            print(f"{year}: already downloaded", flush=True)
            continue
        first, last = max(args.first, f"{year}-01-01"), min(args.last, f"{year}-12-31")
        rows = news_rows(iter_news(fetch, args.symbols, f"{first}T00:00:00Z", f"{last}T23:59:59Z",
                                   min_interval_s=args.interval))
        save_news(rows, path)
        print(f"{year}: {len(rows)} articles", flush=True)
    write_manifest(out, {"source": "alpaca-news (v1beta1)", "symbols": args.symbols, "first": args.first,
                         "last": args.last, "fields": "id, created_at, symbols, headline, source",
                         "license_notes": "Alpaca news data, downloaded for personal research; check Alpaca's terms "
                                          "before sharing or redistributing"})
    print(f"news written to {out}", flush=True)
    return EXIT_OK


def _cmd_swing_track(args: argparse.Namespace) -> int:
    """Refetch daily bars (read-only) and score the frozen swing rules against Hold since their start date."""
    from datetime import UTC, datetime, timedelta

    from atlas.swing.track import TrackConfig, check_frozen, last_complete_day, track

    cfg = load_config(Path(args.config), TrackConfig)
    check_frozen(cfg)  # refuse changed rules before downloading anything
    if not args.no_fetch:
        from atlas.feeds.alpaca import HISTORY_BARS_PATH, HISTORY_PATHS, http_fetch, iter_history, load_credentials
        from atlas.swing.data import daily_rows, save_daily

        now = datetime.now(UTC)
        last = last_complete_day(now)
        end = (now - timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ")  # the free plan has no SIP data newer than 15 min
        fetch = http_fetch(load_credentials(Path(".env")), timeout_s=60, allowed=HISTORY_PATHS)
        rows = {}
        for sym in cfg.symbols():  # ~1 request per symbol, paced well under the free plan's limit
            got = daily_rows(iter_history(fetch, HISTORY_BARS_PATH, sym, f"{cfg.history_start}T00:00:00Z", end, "sip",
                                          min_interval_s=1.0, extra={"timeframe": "1Day", "adjustment": "all"}))
            rows[sym] = [r for r in got if r["date"] <= last]  # today's bar only once it is final
        save_daily(rows, cfg.dataset_dir, {"source": "alpaca-sip-1Day", "adjustment": "all (splits and dividends)",
                                           "through": str(last), "license_notes": "Alpaca historical market data, "
                                           "downloaded for personal research; check Alpaca's data terms before sharing"})
        print(f"daily bars through {last} for {len(rows)} symbols", flush=True)
    out = track(cfg)
    print(out.read_text(encoding="utf-8"))
    print(f"report: {out}")
    return EXIT_OK


def _cmd_swing_walkforward(args: argparse.Namespace) -> int:
    from atlas.swing.walkforward import SwingConfig, walkforward

    out = walkforward(load_config(Path(args.config), SwingConfig))
    print(f"report: {out / 'report.md'}")
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

    p = sub.add_parser("fetch-history", help="Download real Alpaca history (read-only) as polled training sessions.")
    p.add_argument("--dataset", default="alpaca_sip_hist")
    p.add_argument("--feed", default="sip", choices=["sip", "iex"])
    p.add_argument("--symbols", nargs="+", default=["SPY", "QQQ"])
    p.add_argument("--first", default="2016-06-01")
    p.add_argument("--last", default="2026-10-23")
    p.add_argument("--days", type=int, default=40, help="random days, spread evenly (stress days are added)")
    p.add_argument("--minutes", type=int, default=30, help="window length per day")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--workers", type=int, default=6, help="parallel downloads (shared rate limit)")
    p.set_defaults(func=_cmd_fetch_history)

    p = sub.add_parser("fetch-bars", help="Download real 1-minute bars (read-only) as full-day training sessions.")
    p.add_argument("--dataset", default="alpaca_bars_1min")
    p.add_argument("--feed", default="sip", choices=["sip", "iex"])
    p.add_argument("--symbols", nargs="+", default=["SPY", "QQQ"])
    p.add_argument("--first", default="2016-01-04")
    p.add_argument("--last", default="2026-10-26")
    p.add_argument("--spread-bps", type=float, default=1.0, help="modeled quoted spread (live IEX median ~0.75)")
    p.add_argument("--min-bars", type=int, default=380, help="skip early closes and gappy days")
    p.set_defaults(func=_cmd_fetch_bars)

    p = sub.add_parser("fetch-daily", help="Download real adjusted daily bars (read-only) for swing research.")
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--feed", default="sip", choices=["sip", "iex"])
    p.add_argument("--first", default="2016-01-04")
    p.add_argument("--last", default="2026-10-27")
    p.add_argument("--out", default="data/processed/alpaca_daily")
    p.set_defaults(func=_cmd_fetch_daily)

    p = sub.add_parser("fetch-news", help="Download news headlines (read-only Alpaca market data) for swing research.")
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--first", default="2016-01-01")
    p.add_argument("--last", default="2026-10-27")
    p.add_argument("--interval", type=float, default=1.0, help="seconds between requests (rate limit)")
    p.add_argument("--out", default="data/raw/alpaca_news/stocks30")
    p.set_defaults(func=_cmd_fetch_news)

    p = sub.add_parser("swing-track", help="Score frozen swing rules against Hold on days since they were frozen.")
    p.add_argument("--config", default="configs/swing_track.yaml")
    p.add_argument("--no-fetch", action="store_true", help="score the bars already downloaded")
    p.set_defaults(func=_cmd_swing_track)

    p = sub.add_parser("swing-walkforward", help="Walk-forward test of multi-day (swing) bots vs buy-and-hold.")
    p.add_argument("--config", required=True)
    p.set_defaults(func=_cmd_swing_walkforward)

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
    p.add_argument("--open-dashboard", action="store_true", help="open the live page in the browser")
    p.add_argument("--auto-resume", action="store_true", help="resume today's session if it already exists")
    p.add_argument("--watch", action="store_true", help="open the leaderboard in a new terminal window")
    p.add_argument("--reset-fund", action="store_true", help="archive the fund ledger and split capital afresh")
    p.set_defaults(func=_cmd_paper)

    p = sub.add_parser("paper-window", help="Open the live leaderboard window for a paper session.")
    p.add_argument("--session-dir")
    p.add_argument("--config", help="paper config; targets today's session")
    p.set_defaults(func=_cmd_paper_window)

    p = sub.add_parser("paper-watch", help="Live leaderboard in this terminal (refreshes every 2 s).")
    p.add_argument("--session-dir")
    p.add_argument("--config", help="paper config; targets today's session")
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--once", action="store_true", help="draw once and exit")
    p.set_defaults(func=_cmd_paper_watch)

    p = sub.add_parser("paper-status", help="Print the live leaderboard of a paper session.")
    p.add_argument("--session-dir")
    p.add_argument("--config", help="paper config; targets today's session")
    p.set_defaults(func=_cmd_paper_status)

    p = sub.add_parser("dashboard", help="Open the read-only live page (leaderboard, P&L, trades).")
    p.add_argument("--paper-dir", default="artifacts/paper")
    p.add_argument("--port", type=int, default=8501)
    p.set_defaults(func=_cmd_dashboard)

    p = sub.add_parser("paper-kill", help="Engage the kill switch of a paper session (halts trading).")
    p.add_argument("--session-dir")
    p.add_argument("--config", help="paper config; targets today's session")
    p.add_argument("--reason", default="operator kill")
    p.set_defaults(func=_cmd_paper_kill)

    p = sub.add_parser("paper-stop", help="Stop a running paper session cleanly (resumable).")
    p.add_argument("--session-dir")
    p.add_argument("--config", help="paper config; targets today's session")
    p.set_defaults(func=_cmd_paper_stop)

    p = sub.add_parser("paper-check", help="Preflight: keys, clock, bots, live page; does not trade.")
    p.add_argument("--config", required=True)
    p.set_defaults(func=_cmd_paper_check)

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
