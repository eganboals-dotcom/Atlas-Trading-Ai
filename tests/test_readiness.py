from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml
from paper_helpers import REPO, RISK, paper_dict

from atlas.readiness import BROKER_OPEN_ITEMS, ReadinessConfig, evaluate_gates, run_drills

CI = lambda mean, low, high, n: {"mean": mean, "low": low, "high": high, "n": n}  # noqa: E731


def evidence(**kw):
    ev = {
        "walkforward_run": "wf-x",
        "walkforward": {
            "data_source": {"provider": "synthetic", "timestamp_kind": "synthetic", "universe": "fixed"},
            "summary": {
                "champion_oos": {"base": CI(-4e-5, -7e-5, -2e-6, 12), "pessimistic": CI(-3e-4, -4e-4, -2e-4, 12)},
                "baselines_oos": {"baseline-cash": {"base": CI(0.0, 0.0, 0.0, 12)}},
                "selection_gap": {"mean_gap": 2.6e-5, "mean_in_sample_return": -1e-5, "mean_oos_base_return": -4e-5},
            },
            "selection_trials": {"agent_session_evaluations": 4800},
            "limitations": [],
        },
        "holdout": None, "ppo_run": None, "ppo": None, "holdout_ledger": [],
        "paper_sessions": [{
            "session": "p1", "feed": {"kind": "replay"}, "kill_switch": None,
            "accounts": [{"account": "champion", "role": "champion", "net_pnl": -50.0, "halted": None,
                          "excess_return_vs_benchmark": 0.002, "decision_latency": {"p99_us": 5.0}}],
        }],
    }
    ev.update(kw)
    return ev


def status(gates, gid):
    return next(g.status for g in gates if g.id == gid)


def test_current_evidence_is_not_ready() -> None:
    gates = evaluate_gates(ReadinessConfig(), evidence(), drills=None, tests=None)
    required = [g for g in gates if g.required]
    assert not all(g.status == "PASS" for g in required)
    for gid, expected in [("data.real", "FAIL"), ("oos.base", "FAIL"), ("oos.pessimistic", "FAIL"),
                          ("oos.beats_cash", "FAIL"), ("holdout.evaluated", "MISSING"),
                          ("paper.live_feed", "FAIL"), ("broker.capabilities", "MISSING"),
                          ("signoff.independent_risk_review", "PENDING"), ("ops.drills", "MISSING")]:
        assert status(gates, gid) == expected, gid


def test_missing_evidence_is_reported_as_missing_not_pass() -> None:
    ev = evidence(walkforward=None, walkforward_run=None, paper_sessions=[])
    gates = evaluate_gates(ReadinessConfig(), ev, None, None)
    assert {status(gates, g) for g in ("data.real", "oos.base", "paper.live_feed", "risk.paper_incidents")} == {"MISSING"}


def test_gates_can_pass_only_with_complete_positive_evidence_and_signoffs() -> None:
    good = copy.deepcopy(evidence())
    wf = good["walkforward"]
    wf["data_source"] = {"provider": "parquet", "timestamp_kind": "exchange", "universe_source": "universe file"}
    wf["summary"]["champion_oos"] = {"base": CI(3e-4, 1e-4, 5e-4, 80), "pessimistic": CI(2e-4, 5e-5, 4e-4, 80)}
    wf["summary"]["selection_gap"]["mean_gap"] = 5e-5
    good["holdout"] = {"holdout_accesses_total": 1, "holdout_reused": False,
                       "champion_ci": {"base": CI(2e-4, 1e-5, 4e-4, 20), "pessimistic": CI(1e-4, 1e-6, 3e-4, 20)}}
    session = good["paper_sessions"][0] | {"feed": {"kind": "vendor"}}
    session["accounts"] = [session["accounts"][0] | {"net_pnl": 10.0}]
    good["paper_sessions"] = [session] * 20
    signed = {"reviewer": "R", "date": "2027-01-01", "document": "doc", "outcome": "approved"}
    cfg = ReadinessConfig(signoffs={k: signed for k in ("independent_risk_review", "legal_regulatory_tax_review",
                                                        "broker_terms_review", "operator_runbook_review")})

    class D:
        passed = True
        name = "d"

    gates = evaluate_gates(cfg, good, [D()], {"exit_code": 0, "summary": "ok"})
    remaining = [g.id for g in gates if g.required and g.status != "PASS"]
    # Even perfect research evidence cannot clear items that need a broker connection or vendor adapter,
    # which by design do not exist before an approved Phase 7.
    assert set(remaining) == {"data.point_in_time", "ops.broker_latency", "ops.live_feed_adapter",
                              "broker.capabilities"}


def test_reused_holdout_fails() -> None:
    ev = evidence(holdout={"holdout_accesses_total": 3, "holdout_reused": True,
                           "champion_ci": {"base": CI(1e-4, 1e-5, 2e-4, 6), "pessimistic": CI(1e-4, 1e-5, 2e-4, 6)}})
    assert status(evaluate_gates(ReadinessConfig(), ev, None, None), "holdout.evaluated") == "FAIL"


def test_broker_items_cover_the_spec_checklist() -> None:
    items = {b["item"] for b in BROKER_OPEN_ITEMS}
    for needed in ("API product access", "App approval and OAuth", "Rate limits", "Sandbox coverage",
                   "Supported order types and time-in-force", "Terms of use"):
        assert needed in items


def test_failure_drills_all_pass(tmp_path: Path) -> None:
    (tmp_path / "risk.yaml").write_text(yaml.safe_dump(RISK), encoding="utf-8")
    cfg = paper_dict(tmp_path)
    cfg.update(simulator_config=str(REPO / "configs/simulator.yaml"), risk_config=str(tmp_path / "risk.yaml"))
    path = tmp_path / "paper.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    results = run_drills(path, tmp_path / "review")
    failed = [(d.name, d.observed, d.error) for d in results if not d.passed]
    assert failed == []
    assert {d.name for d in results} == {"baseline", "disconnect_recovers", "reconnect_exhausted", "stale_feed",
                                         "clock_anomaly", "crash_and_resume", "manual_kill", "daily_loss_halt",
                                         "live_config_refused"}


def test_readiness_config_rejects_unknown_signoffs() -> None:
    with pytest.raises(ValueError):
        ReadinessConfig.model_validate({"signoffs": {"made_up_review": None}})
