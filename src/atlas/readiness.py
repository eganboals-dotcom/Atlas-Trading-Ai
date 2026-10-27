"""Phase 6 readiness review: evidence, failure drills, gates, and open issues.

This module only *reads* research and paper artifacts, and runs paper-mode drills. It
does not implement, enable, or configure live trading. A "ready" verdict is a
recommendation for a human decision. It never activates anything: a broker adapter
(Phase 7) and any real-money use (Phase 8) each need separate explicit approval.

Gate statuses:
* PASS — evidence meets the criterion;
* FAIL — evidence exists and does not meet it;
* MISSING — the evidence needed does not exist yet;
* PENDING — needs a recorded human sign-off (reviewer, date, document).

The verdict is READY only if every required gate passes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from atlas import __version__
from atlas.logging_utils import get_logger

log = get_logger(__name__)

Status = Literal["PASS", "FAIL", "MISSING", "PENDING"]


class SignOff(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    reviewer: str
    date: date
    document: str
    outcome: Literal["approved", "rejected"]


class Criteria(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    min_oos_sessions: int = Field(default=60, ge=1)
    min_oos_ci_lower_bps: float = 0.0  # base AND pessimistic lower bound must exceed this
    min_paper_sessions: int = Field(default=20, ge=1)
    max_selection_gap_bps: float = 1.0


class ReadinessConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    walkforward_reports_dir: Path = Path("artifacts/reports/walkforward")
    walkforward_experiment: str = "wf_synthetic_v1"
    ppo_experiment: str | None = "wf_synthetic_ppo_v1"
    paper_dir: Path = Path("artifacts/paper")
    holdout_ledger: Path = Path("artifacts/holdout_ledger.jsonl")
    paper_config: Path = Path("configs/paper.yaml")
    run_drills: bool = True
    run_tests: bool = True
    criteria: Criteria = Criteria()
    signoffs: dict[Literal["independent_risk_review", "legal_regulatory_tax_review", "broker_terms_review",
                           "operator_runbook_review"], SignOff | None] = Field(default_factory=lambda: {
        "independent_risk_review": None, "legal_regulatory_tax_review": None, "broker_terms_review": None,
        "operator_runbook_review": None})
    output_dir: Path = Path("artifacts/readiness")
    docs_path: Path | None = Path("docs/READINESS_REVIEW.md")


@dataclass
class Gate:
    id: str
    area: str
    criterion: str
    status: Status
    evidence: str
    required: bool = True


# Unverified broker/API items (spec sections 13 and 23). These are questions to answer
# from current official documentation. They are not answers.
BROKER_OPEN_ITEMS: list[dict[str, str]] = [
    {"item": "API product access", "question": "Is the Trader API (Individual) approved for this account, covering "
     "both the Market Data Production and the Accounts and Trading Production products? Access to one does not "
     "imply the other.", "reference": "https://developer.schwab.com/products/trader-api--individual"},
    {"item": "App approval and OAuth", "question": "App registration status, the OAuth authorization-code flow, "
     "access and refresh token lifetimes, re-authorization cadence, and secure token storage requirements.",
     "reference": "https://developer.schwab.com/user-guides/get-started/authenticate-with-oauth"},
    {"item": "Account permissions", "question": "Which order types, instruments, and account types are enabled. "
     "Initial scope is long-only, cash-funded US ETFs, with no margin, shorting, or options.",
     "reference": "https://developer.schwab.com/products/trader-api--individual"},
    {"item": "Supported order types and time-in-force", "question": "Is IOC (or an equivalent) supported? What "
     "limit/market semantics, order preview, and replace/cancel behavior apply? Atlas assumes IOC fills.",
     "reference": "https://developer.schwab.com/products/trader-api--individual"},
    {"item": "Rate limits", "question": "Request and order-submission limits per minute, and the throttling and "
     "error behavior. These bound order frequency and may rule out sub-second strategies.",
     "reference": "https://developer.schwab.com/products/trader-api--individual"},
    {"item": "Market-data entitlements and streaming", "question": "Real-time vs delayed Level 1 data, symbol and "
     "connection limits, message formats, heartbeat behavior, and the precision and semantics of exchange, vendor, "
     "and local timestamps.", "reference": "https://developer.schwab.com/products/trader-api--individual"},
    {"item": "Sandbox coverage", "question": "Which API methods the developer sandbox supports for this app. It "
     "returns engineered sample data with simulated OAuth, so it is not a realistic paper-trading fill simulator.",
     "reference": "https://developer.schwab.com/user-guides/apis-and-apps/test-in-sandbox"},
    {"item": "Terms of use", "question": "Whether the terms permit automated trading by this account holder, and "
     "any restrictions on data redistribution, storage, and derived use.",
     "reference": "https://developer.schwab.com/user-guides/individual-developer/become-individual-developer"},
    {"item": "Latency and throughput", "question": "Measured order round-trip latency and fill-report latency. No "
     "retail API should be assumed suitable for HFT without measurement.", "reference": "measurement required"},
    {"item": "Reconciliation and recovery", "question": "How fills, orders, and positions can be queried for "
     "reconciliation; duplicate-submission and idempotency behavior; and broker-side manual kill and recovery "
     "procedures.", "reference": "https://developer.schwab.com/products/trader-api--individual"},
]


# ------------------------------------------------------------------ evidence


def _latest_run(reports_dir: Path, experiment: str) -> Path | None:
    runs = sorted(p for p in reports_dir.glob(f"{experiment}-*") if (p / "report.json").is_file())
    return runs[-1] if runs else None


def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def collect_evidence(cfg: ReadinessConfig) -> dict[str, Any]:
    ev: dict[str, Any] = {}
    run = _latest_run(cfg.walkforward_reports_dir, cfg.walkforward_experiment)
    ev["walkforward_run"] = None if run is None else run.name
    ev["walkforward"] = _load(run / "report.json") if run else None
    ev["holdout"] = _load(run / "holdout_report.json") if run else None
    ppo_run = _latest_run(cfg.walkforward_reports_dir, cfg.ppo_experiment) if cfg.ppo_experiment else None
    ppo_report = _load(ppo_run / "report.json") if ppo_run else None
    ev["ppo_run"] = None if ppo_run is None else ppo_run.name
    ev["ppo"] = (ppo_report or {}).get("ppo")
    ledger = cfg.holdout_ledger
    ev["holdout_ledger"] = [json.loads(x) for x in ledger.read_text(encoding="utf-8").splitlines() if x.strip()] \
        if ledger.is_file() else []
    sessions = []
    if cfg.paper_dir.is_dir():
        for d in sorted(cfg.paper_dir.iterdir()):
            r = _load(d / "daily_report.json")
            if r:
                sessions.append(r)
    ev["paper_sessions"] = sessions
    return ev


# -------------------------------------------------------------------- drills


@dataclass
class DrillResult:
    name: str
    scenario: str
    expected: str
    observed: str
    passed: bool
    error: str | None = None


def run_drills(paper_config_path: Path, out_dir: Path) -> list[DrillResult]:
    """Standard failure drills, run in paper mode against the shipped paper configuration."""
    import yaml

    from atlas.config import SimulatorConfig, load_config
    from atlas.paper.config import FaultConfig, PaperConfig, RiskConfig
    from atlas.paper.feed import ReplayLiveFeed, SimulatedCrash
    from atlas.paper.risk import request_kill
    from atlas.paper.session import HealthCheckError, PaperSession

    base_dir = paper_config_path.parent

    def resolve(p: Path) -> Path:
        return p if p.is_file() else base_dir / p

    raw = yaml.safe_load(paper_config_path.read_text(encoding="utf-8"))
    base = PaperConfig.model_validate(raw)
    sim = load_config(resolve(base.simulator_config), SimulatorConfig)
    risk = load_config(resolve(base.risk_config), RiskConfig)
    drills_dir = out_dir / "drills"

    def make(name: str, faults: dict[str, Any] | None = None, risk_cfg: RiskConfig | None = None,
             resume: bool = False, feed=None, reset: bool = False) -> PaperSession:
        cfg = base.model_copy(update={
            "session_name": f"drill_{name}", "output_dir": drills_dir,
            "alerts": base.alerts.model_copy(update={"console": False}),
            "feed": base.feed.model_copy(update={"faults": FaultConfig(**(faults or {}))}),
        })
        return PaperSession(cfg, sim, risk_cfg or risk, base_dir=Path("."), feed=feed, resume=resume,
                            reset_kill_switch=reset)

    def audit(s: PaperSession, kind: str) -> list[dict[str, Any]]:
        lines = (s.dir / "audit.jsonl").read_text(encoding="utf-8").splitlines()
        return [r for r in map(json.loads, lines) if r.get("type") == kind]

    results: list[DrillResult] = []

    def drill(name: str, scenario: str, expected: str, fn: Callable[[], tuple[bool, str]]) -> None:
        try:
            ok, observed = fn()
            results.append(DrillResult(name, scenario, expected, observed, ok))
        except Exception as exc:  # a drill that crashes is a failed drill, reported, never hidden
            results.append(DrillResult(name, scenario, expected, "drill raised an exception", False, repr(exc)))

    baseline: dict[str, Any] = {}

    def clean() -> tuple[bool, str]:
        r = make("baseline").run()
        baseline.update(r)
        return r["kill_switch"] is None, f"{r['counters']['events']} events, kill switch {r['kill_switch']}"

    drill("baseline", "clean session", "completes, no kill switch", clean)

    def disconnect() -> tuple[bool, str]:
        r = make("disconnect", {"disconnect_at_events": [1000], "disconnect_failures": 2}).run()
        c = r["counters"]
        ok = c["disconnects"] == 1 and c["reconnect_attempts"] == 3 and r["kill_switch"] is None
        return ok, f"disconnects {c['disconnects']}, reconnect attempts {c['reconnect_attempts']}, " \
                   f"kill {r['kill_switch']}"

    drill("disconnect_recovers", "feed drops at event 1000; 2 reconnects fail",
          "safe state, recovers on attempt 3, session continues", disconnect)

    def exhausted() -> tuple[bool, str]:
        r = make("reconnect_exhausted", {"disconnect_at_events": [1000], "disconnect_failures": 99}).run()
        ok = bool(r["kill_switch"]) and "feed lost" in r["kill_switch"] and r["counters"]["events"] <= 1001
        return ok, f"kill switch: {r['kill_switch']}; events processed {r['counters']['events']}"

    drill("reconnect_exhausted", "feed drops and never comes back",
          f"kill switch after {risk.reconnect_max_retries} attempts; session stops", exhausted)

    def stale() -> tuple[bool, str]:
        s = make("stale_feed", {"stale_gaps": [[2000, 10_000]]})
        r = s.run()
        periods = [p for p in r["safe_state_periods"] if p["reason"] == "stale or missing data"]
        orders = audit(s, "order")
        inside = [o for p in periods for o in orders if p["start"] < o["ts"] < p["end"]]
        longest = max((p["duration_ms"] for p in periods), default=0)
        return bool(periods) and longest > risk.stale_data_ms and not inside, \
            f"{len(periods)} stale safe-state periods, longest {longest:.0f} ms, orders during them: {len(inside)}"

    drill("stale_feed", "10 s of total feed silence at event 2000",
          "safe state from the moment data went stale; no orders until fresh data", stale)

    def clock() -> tuple[bool, str]:
        r = make("clock", {"clock_backwards_at": [1500], "future_timestamp_at": [3000]}).run()
        n = sum(p["reason"] == "clock anomaly" for p in r["safe_state_periods"])
        return r["counters"]["clock_anomalies"] == 2 and n == 2, \
            f"clock anomalies {r['counters']['clock_anomalies']}, clock safe-state periods {n}"

    drill("clock_anomaly", "local clock runs backwards; an event is stamped 10 s in the future",
          "both detected; safe state until ordering is trustworthy again", clock)

    def crash() -> tuple[bool, str]:
        s = make("crash", {"crash_at_event": 2345})
        try:
            s.run()
            return False, "no crash occurred"
        except SimulatedCrash:
            pass
        r = make("crash", resume=True).run()
        orders = Counter((o["account"], o["order_id"]) for o in audit(s, "order"))
        ok = r["restarts"] == 1 and r["counters"]["events"] == baseline.get("counters", {}).get("events") \
            and max(orders.values(), default=1) == 1
        return ok, f"restarts {r['restarts']}, events {r['counters']['events']} " \
                   f"(clean run {baseline.get('counters', {}).get('events')}), duplicated order records " \
                   f"{sum(v > 1 for v in orders.values())}"

    drill("crash_and_resume", "process dies at event 2345 (last state flush at 2000), then --resume",
          "ledgers restored, in-flight orders cancelled, no events skipped or double-logged", crash)

    def manual_kill() -> tuple[bool, str]:
        probe = make("probe")
        probe.health_checks()
        ref = probe.feed.ref
        target = drills_dir / "drill_manual_kill"

        class KillAt(ReplayLiveFeed):
            def next_event(self):
                if self.position == 2000:
                    request_kill(target, "readiness drill")
                return super().next_event()

        s = make("manual_kill", feed=KillAt(ref))
        r = s.run()
        kill = audit(s, "kill_switch")
        after = [o for o in audit(s, "order") if kill and o["ts"] > kill[0]["ts"]]
        try:
            make("manual_kill", resume=True).health_checks()
            refused = False
        except HealthCheckError:
            refused = True
        ok = bool(r["kill_switch"]) and r["kill_switch"].startswith("manual") and not after and refused
        return ok, f"kill: {r['kill_switch']}; orders after kill {len(after)}; restart refused without reset: {refused}"

    drill("manual_kill", "operator writes the KILL file mid-session",
          "no new orders, working orders cancelled, restart refused until reset", manual_kill)

    def daily_loss() -> tuple[bool, str]:
        s = make("daily_loss", risk_cfg=risk.model_copy(update={"max_daily_loss": 0.0001}))
        r = s.run()
        halted = {a["account"]: a["halted"] for a in r["accounts"] if a["halted"]}
        halts = audit(s, "risk_halt")
        after = [o for o in audit(s, "order") for h in halts if o["account"] == h["account"] and o["ts"] > h["ts"]]
        return bool(halted) and not after, f"halted accounts {sorted(halted)}; orders after halt {len(after)}"

    drill("daily_loss_halt", "daily-loss limit tightened to 0.01%",
          "breaching accounts halt; no orders after the halt", daily_loss)

    def live_refused() -> tuple[bool, str]:
        from pydantic import ValidationError

        refused = []
        for override in ({"live_orders_enabled": True}, {"mode": "live"}, {"api_key": "x"}):
            try:
                PaperConfig.model_validate({**raw, **override})
                refused.append(False)
            except ValidationError:
                refused.append(True)
        return all(refused), f"unsafe configs refused: {sum(refused)}/{len(refused)}"

    drill("live_config_refused", "config with live_orders_enabled: true, mode: live, or an api_key",
          "paper command refuses to start", live_refused)
    return results


def run_test_suite() -> dict[str, Any]:
    # No extra -q: pyproject addopts already passes it, and -qq hides the summary line.
    proc = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-W", "ignore"],
                          capture_output=True, text=True)
    tail = [ln for ln in proc.stdout.splitlines() if " passed" in ln or " failed" in ln or " error" in ln]
    return {"exit_code": proc.returncode, "summary": tail[-1] if tail else proc.stdout[-300:]}


# --------------------------------------------------------------------- gates


def _bps(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 1e4:+.2f} bps"


def evaluate_gates(cfg: ReadinessConfig, ev: dict[str, Any], drills: list[DrillResult] | None,
                   tests: dict[str, Any] | None) -> list[Gate]:
    c = cfg.criteria
    wf = ev["walkforward"]
    gates: list[Gate] = []

    def add(*a, **k) -> None:
        gates.append(Gate(*a, **k))

    # Data fidelity
    if wf is None:
        add("data.real", "Data fidelity", "Research uses real, licensed historical market data", "MISSING",
            "no walk-forward run found")
    else:
        ds = wf["data_source"]
        synthetic = ds.get("provider") == "synthetic" or ds.get("timestamp_kind") == "synthetic"
        add("data.real", "Data fidelity", "Research uses real, licensed historical market data",
            "FAIL" if synthetic else "PASS", f"provider {ds.get('provider')}, timestamps {ds.get('timestamp_kind')}")
        ts_ok = ds.get("timestamp_kind") in ("exchange", "vendor")
        add("data.timestamps", "Data fidelity", "Exchange/vendor timestamps with measured precision and feed latency",
            "PASS" if ts_ok else "FAIL", f"timestamp provenance: {ds.get('timestamp_kind')}; precision and "
            "latency not yet measured on a real feed")
        add("data.point_in_time", "Data fidelity", "Point-in-time universe including delisted symbols",
            "MISSING" if synthetic else "PENDING", ds.get("universe") or ds.get("universe_source") or "unknown")

    # Out-of-sample performance and cost sensitivity
    if wf is None:
        for gid, crit in (("oos.base", "Out-of-sample base-cost return CI lower bound > threshold"),
                          ("oos.pessimistic", "Out-of-sample pessimistic-cost return CI lower bound > threshold"),
                          ("oos.sample", "Enough out-of-sample sessions"), ("oos.selection_bias", "Selection gap")):
            add(gid, "Out-of-sample", crit, "MISSING", "no walk-forward run found")
    else:
        s = wf["summary"]
        for scen, gid, area in (("base", "oos.base", "Out-of-sample"),
                                ("pessimistic", "oos.pessimistic", "Cost sensitivity")):
            ci = s["champion_oos"].get(scen, {})
            low = ci.get("low")
            ok = low is not None and low * 1e4 > c.min_oos_ci_lower_bps
            add(gid, area, f"{scen} costs: champion OOS mean-return CI lower bound > {c.min_oos_ci_lower_bps:+.2f} bps",
                "PASS" if ok else "FAIL", f"mean {_bps(ci.get('mean'))}, CI [{_bps(low)}, {_bps(ci.get('high'))}], "
                f"n={ci.get('n')}")
        cash = s["baselines_oos"].get("baseline-cash", {}).get("base", {}).get("mean")
        mean = s["champion_oos"].get("base", {}).get("mean")
        add("oos.beats_cash", "Out-of-sample", "Champion beats the cash baseline out of sample (base costs)",
            "PASS" if mean is not None and cash is not None and mean > cash else "FAIL",
            f"champion {_bps(mean)} vs cash {_bps(cash)}")
        n = s["champion_oos"].get("base", {}).get("n") or 0
        add("oos.sample", "Out-of-sample", f"At least {c.min_oos_sessions} out-of-sample sessions",
            "PASS" if n >= c.min_oos_sessions else "FAIL", f"{n} validation sessions")
        gap = s["selection_gap"]["mean_gap"]
        add("oos.selection_bias", "Out-of-sample", f"Selection gap ≤ {c.max_selection_gap_bps} bps, trials counted",
            "PASS" if gap is not None and gap * 1e4 <= c.max_selection_gap_bps else "FAIL",
            f"gap {_bps(gap)}; {wf['selection_trials']['agent_session_evaluations']} agent-session evaluations")

    # Outer holdout
    h = ev["holdout"]
    if h is None:
        add("holdout.evaluated", "Outer holdout", "Frozen design evaluated once on the untouched outer holdout",
            "MISSING", "holdout not evaluated" + (f" for {ev['walkforward_run']}" if ev["walkforward_run"] else ""))
    else:
        reused = h.get("holdout_reused")
        add("holdout.evaluated", "Outer holdout", "Frozen design evaluated once on the untouched outer holdout",
            "FAIL" if reused else "PASS", f"{h['holdout_accesses_total']} recorded access(es); reused: {reused}")
        for scen in ("base", "pessimistic"):
            ci = h["champion_ci"].get(scen, {})
            low = ci.get("low")
            add(f"holdout.{scen}", "Outer holdout", f"{scen} costs: holdout CI lower bound > "
                f"{c.min_oos_ci_lower_bps:+.2f} bps", "PASS" if low is not None and low * 1e4 > c.min_oos_ci_lower_bps
                else "FAIL", f"mean {_bps(ci.get('mean'))}, CI [{_bps(low)}, {_bps(ci.get('high'))}], n={ci.get('n')}")

    # Paper trading
    sessions = ev["paper_sessions"]
    live = [r for r in sessions if r["feed"]["kind"] != "replay"]
    add("paper.live_feed", "Paper trading", "Paper sessions ran on a live market-data feed",
        "PASS" if live else ("FAIL" if sessions else "MISSING"),
        f"{len(sessions)} session(s), {len(live)} on a live feed (others were replays)")
    add("paper.duration", "Paper trading", f"At least {c.min_paper_sessions} forward paper sessions on live data",
        "PASS" if len(live) >= c.min_paper_sessions else "FAIL", f"{len(live)} live-data session(s)")
    champ = [a for r in sessions for a in r["accounts"] if a["role"] == "champion"]
    if champ:
        pos = sum(a["excess_return_vs_benchmark"] > 0 for a in champ)
        add("paper.vs_benchmark", "Paper trading", "Champion net P&L positive over the paper period",
            "PASS" if sum(a["net_pnl"] for a in champ) > 0 else "FAIL",
            f"total champion net P&L {sum(a['net_pnl'] for a in champ):+.2f} over {len(champ)} session(s); "
            f"beat benchmark in {pos}")
    else:
        add("paper.vs_benchmark", "Paper trading", "Champion net P&L positive over the paper period", "MISSING",
            "no paper sessions")

    # Risk and operations
    halts = [(r["session"], a["account"], a["halted"]) for r in sessions for a in r["accounts"] if a["halted"]]
    kills = [(r["session"], r["kill_switch"]) for r in sessions if r["kill_switch"]]
    add("risk.paper_incidents", "Risk", "No unexplained risk halts or kill-switch events in paper sessions",
        "PASS" if sessions and not halts and not kills else ("MISSING" if not sessions else "FAIL"),
        f"halts {halts or 'none'}; kills {kills or 'none'}")
    if drills is None:
        add("ops.drills", "Operational reliability", "All failure drills pass", "MISSING", "drills not run")
    else:
        failed = [d.name for d in drills if not d.passed]
        add("ops.drills", "Operational reliability", "All failure drills pass", "FAIL" if failed else "PASS",
            f"{len(drills) - len(failed)}/{len(drills)} passed" + (f"; failed: {failed}" if failed else ""))
    if tests is None:
        add("ops.tests", "Operational reliability", "Full automated test suite passes", "MISSING", "not run")
    else:
        add("ops.tests", "Operational reliability", "Full automated test suite passes",
            "PASS" if tests["exit_code"] == 0 else "FAIL", tests["summary"])
    lat = [a["decision_latency"]["p99_us"] for r in sessions for a in r["accounts"]
           if a["decision_latency"]["p99_us"] is not None]
    add("ops.local_latency", "Operational reliability", "Local p99 decision latency well below the decision interval",
        "PASS" if lat else "MISSING", f"max p99 {max(lat):.0f} µs" if lat else "no paper latency samples",
        required=False)
    add("ops.broker_latency", "Operational reliability", "Network and broker order round-trip latency measured",
        "MISSING", "no broker connection exists (by design until Phase 7)")
    add("ops.live_feed_adapter", "Operational reliability",
        "Vendor live-feed adapter with heartbeats/timeouts implemented and tested", "MISSING",
        "only the replay feed exists; vendor streaming specs unverified")

    # Broker and governance
    add("broker.capabilities", "Broker", "Broker API products, permissions, order types, and limits verified from "
        "current official documentation", "MISSING", f"{len(BROKER_OPEN_ITEMS)} open items (see below)")
    for key, label in (("broker_terms_review", "Broker terms and data-use review"),
                       ("independent_risk_review", "Independent risk review"),
                       ("legal_regulatory_tax_review", "Legal, regulatory, and tax review"),
                       ("operator_runbook_review", "Operational runbook and broker-side recovery plan reviewed")):
        so = cfg.signoffs.get(key)
        if so is None:
            add(f"signoff.{key}", "Governance", label, "PENDING", "no sign-off recorded")
        else:
            add(f"signoff.{key}", "Governance", label, "PASS" if so.outcome == "approved" else "FAIL",
                f"{so.reviewer}, {so.date}, {so.document}: {so.outcome}")
    return gates


def open_issues(ev: dict[str, Any], gates: list[Gate]) -> list[str]:
    issues = [f"[{g.status}] {g.area}: {g.criterion}. {g.evidence}" for g in gates
              if g.required and g.status != "PASS"]
    if ev.get("ppo"):
        p = ev["ppo"]
        issues.append(f"[INFO] PPO family: OOS base {_bps(p['oos']['base']['mean'])}; "
                      f"{p['promotion']}")
    issues += [
        "[INFO] The risk manager runs in paper mode only; backtests use agent limits and fitness gates.",
        "[INFO] Session calendars do not model holidays, early closes, or halts.",
        "[INFO] Fill model: IOC against top of book only; no queue position, depth, or market impact beyond fixed "
        "slippage.",
    ]
    return issues


# -------------------------------------------------------------------- report


def build_review(cfg: ReadinessConfig) -> tuple[dict[str, Any], Path]:
    ev = collect_evidence(cfg)
    out_dir = cfg.output_dir / f"review-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    out_dir.mkdir(parents=True, exist_ok=False)
    drills = run_drills(cfg.paper_config, out_dir) if cfg.run_drills else None
    tests = run_test_suite() if cfg.run_tests else None
    gates = evaluate_gates(cfg, ev, drills, tests)
    required = [g for g in gates if g.required]
    verdict = "READY FOR HUMAN REVIEW" if all(g.status == "PASS" for g in required) else "NOT READY"
    counts = Counter(g.status for g in required)
    review = {
        "generated_at": datetime.now(UTC).isoformat(),
        "atlas_version": __version__,
        "verdict": verdict,
        "verdict_note": ("This verdict is a recommendation for a human decision. It enables nothing. Any broker "
                         "adapter (Phase 7) or real-money activation (Phase 8) needs separate explicit approval."),
        "gate_counts": dict(counts),
        "gates": [asdict(g) for g in gates],
        "drills": None if drills is None else [asdict(d) for d in drills],
        "tests": tests,
        "evidence": {
            "walkforward_run": ev["walkforward_run"], "ppo_run": ev["ppo_run"],
            "holdout_evaluated": ev["holdout"] is not None, "holdout_ledger_entries": len(ev["holdout_ledger"]),
            "paper_sessions": [r["session"] for r in ev["paper_sessions"]],
        },
        "walkforward_summary": None if ev["walkforward"] is None else {
            "champion_oos": ev["walkforward"]["summary"]["champion_oos"],
            "baselines_oos": {k: v.get("base") for k, v in ev["walkforward"]["summary"]["baselines_oos"].items()},
            "selection_gap": ev["walkforward"]["summary"]["selection_gap"],
            "limitations": ev["walkforward"]["limitations"],
        },
        "ppo_summary": ev["ppo"] and {k: ev["ppo"][k] for k in ("oos", "folds_beating_evolution_champion",
                                                                "folds_compared", "promotion")},
        "paper_summary": [{"session": r["session"], "feed": r["feed"]["kind"],
                           "accounts": {a["account"]: {"net_pnl": a["net_pnl"],
                                                       "excess_vs_benchmark": a["excess_return_vs_benchmark"],
                                                       "halted": a["halted"]} for a in r["accounts"]}}
                          for r in ev["paper_sessions"]],
        "broker_open_items": BROKER_OPEN_ITEMS,
        "open_issues": open_issues(ev, gates),
    }
    (out_dir / "readiness_report.json").write_text(json.dumps(review, indent=2, default=str), encoding="utf-8")
    md = render(review)
    (out_dir / "readiness_report.md").write_text(md, encoding="utf-8")
    if cfg.docs_path:
        cfg.docs_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.docs_path.write_text(md, encoding="utf-8")
    return review, out_dir


def render(r: dict[str, Any]) -> str:
    icon = {"PASS": "PASS", "FAIL": "FAIL", "MISSING": "MISSING", "PENDING": "PENDING"}
    gc = r["gate_counts"]
    lines = [
        "# Project Atlas — readiness review", "",
        f"Generated {r['generated_at'][:19]}Z · atlas {r['atlas_version']}", "",
        f"## Verdict: **{r['verdict']}**", "",
        f"> {r['verdict_note']}", "",
        f"Required gates: {gc.get('PASS', 0)} pass, {gc.get('FAIL', 0)} fail, {gc.get('MISSING', 0)} missing, "
        f"{gc.get('PENDING', 0)} pending sign-off.", "",
        "**Live trading is not implemented and remains disabled.** Nothing in this review changes that.", "",
        "## Gates", "",
        "| Status | Area | Criterion | Evidence |", "|---|---|---|---|",
    ]
    for g in r["gates"]:
        lines.append(f"| {icon[g['status']]}{'' if g['required'] else ' (info)'} | {g['area']} | {g['criterion']} "
                     f"| {g['evidence']} |")
    if r["walkforward_summary"]:
        w = r["walkforward_summary"]
        lines += ["", "## Out-of-sample performance and cost sensitivity", "",
                  f"Walk-forward run `{r['evidence']['walkforward_run']}`. Champion mean per-session net return on "
                  "validation sessions:", "", "| Scenario | Mean | 90% CI |", "|---|---|---|"]
        for scen, ci in w["champion_oos"].items():
            lines.append(f"| {scen} | {_bps(ci['mean'])} | [{_bps(ci['low'])}, {_bps(ci['high'])}] |")
        cash = (w["baselines_oos"].get("baseline-cash") or {}).get("mean")
        bh = (w["baselines_oos"].get("baseline-buy_hold") or {}).get("mean")
        g = w["selection_gap"]
        lines += ["", f"Baselines (base): cash {_bps(cash)}, buy & hold {_bps(bh)}. In-sample "
                  f"{_bps(g['mean_in_sample_return'])} vs out-of-sample {_bps(g['mean_oos_base_return'])} "
                  f"(selection gap {_bps(g['mean_gap'])})."]
    if r["ppo_summary"]:
        p = r["ppo_summary"]
        lines += ["", f"PPO (`{r['evidence']['ppo_run']}`): OOS base {_bps(p['oos']['base']['mean'])}, pessimistic "
                  f"{_bps(p['oos']['pessimistic']['mean'])}; beat the evolution champion in "
                  f"{p['folds_beating_evolution_champion']['base']}/{p['folds_compared']} folds (base). "
                  f"{p['promotion']}"]
    lines += ["", "## Paper trading", ""]
    if r["paper_summary"]:
        lines += ["| Session | Feed | Account | Net P&L | vs benchmark | Halted |", "|---|---|---|---|---|---|"]
        for s in r["paper_summary"]:
            for acct, a in s["accounts"].items():
                lines.append(f"| {s['session']} | {s['feed']} | {acct} | {a['net_pnl']:+.2f} | "
                             f"{_bps(a['excess_vs_benchmark'])} | {a['halted'] or 'no'} |")
    else:
        lines.append("No paper sessions recorded.")
    if r["drills"] is not None:
        lines += ["", "## Failure drills", "", "| Drill | Scenario | Expected | Observed | Result |",
                  "|---|---|---|---|---|"]
        for d in r["drills"]:
            obs = d["observed"] + (f" ({d['error']})" if d["error"] else "")
            lines.append(f"| {d['name']} | {d['scenario']} | {d['expected']} | {obs} | "
                         f"{'PASS' if d['passed'] else 'FAIL'} |")
    if r["tests"]:
        lines += ["", f"Test suite: {r['tests']['summary']} (exit code {r['tests']['exit_code']})."]
    lines += ["", "## Unresolved broker and API items", "",
              "Each must be answered from the broker's **current official documentation** (and, where noted, by "
              "measurement) before any adapter is designed. None is assumed.", "",
              "| Item | Question | Reference |", "|---|---|---|"]
    for b in r["broker_open_items"]:
        lines.append(f"| {b['item']} | {b['question']} | {b['reference']} |")
    lines += ["", "## Open issues", ""] + [f"* {x}" for x in r["open_issues"]]
    lines += ["", "## What would change the verdict", "",
              "1. Import licensed historical data, with exchange/vendor timestamps and point-in-time coverage, and "
              "rerun the walk-forward evaluation on it.",
              "2. Find a design whose out-of-sample confidence interval clears zero under base **and** pessimistic "
              "costs on enough sessions. Then freeze it and evaluate the outer holdout exactly once.",
              "3. Build and test a vendor live-feed adapter with heartbeats, then run an extended forward "
              "paper-trading period on live data.",
              "4. Answer every broker item above from official documentation, and record the independent risk, "
              "legal/regulatory/tax, broker-terms, and runbook sign-offs.",
              "5. Only then request explicit approval for Phase 7 (a disabled adapter, tested against mocks and the "
              "documented sandbox). Phase 8 (real money) is a separate decision.", ""]
    return "\n".join(lines)
