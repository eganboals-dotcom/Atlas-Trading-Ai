from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from paper_helpers import audit, risk_config, session

from atlas.environment.portfolio import Portfolio
from atlas.paper.feed import SimulatedCrash
from atlas.paper.session import HealthCheckError


def test_session_produces_accounts_audit_and_daily_report(tmp_path: Path) -> None:
    s = session(tmp_path)
    report = s.run()
    roles = {a["account"]: a["role"] for a in report["accounts"]}
    assert roles == {"champion": "champion", "shadow-1": "shadow", "benchmark": "benchmark"}
    bench = next(a for a in report["accounts"] if a["role"] == "benchmark")
    assert bench["excess_return_vs_benchmark"] == 0.0
    for a in s.accounts:
        a.broker.portfolio.assert_reconciled()
    for name in ("audit.jsonl", "state.json", "daily_report.json", "daily_report.md", "equity.csv"):
        assert (s.dir / name).is_file(), name

    decisions = audit(s, "decision")
    assert decisions and all("quote_age_ms" in v for d in decisions for v in d["snapshot"].values())
    fills = [r for r in audit(s, "execution") if r.get("fill_qty")]
    assert fills and all(r["fill_assumption"] and "cash" in r and "positions" in r for r in fills)
    assert len(audit(s, "risk_check")) >= len(audit(s, "order"))
    assert report["accounts"][0]["model"]["genome_hash"]
    assert report["accounts"][0]["decision_latency"]["samples"] > 0


def test_same_inputs_give_the_same_session(tmp_path: Path) -> None:
    a = session(tmp_path / "a").run()
    b = session(tmp_path / "b").run()
    strip = lambda r: [(x["account"], x["net_pnl"], x["fills"]) for x in r["accounts"]]  # noqa: E731
    assert strip(a) == strip(b)


def test_crash_then_resume_restores_state_without_duplicates(tmp_path: Path) -> None:
    clean = session(tmp_path / "clean").run()

    crashing = session(tmp_path / "r", faults={"crash_at_event": 2345})
    with pytest.raises(SimulatedCrash):
        crashing.run()
    state = json.loads((crashing.dir / "state.json").read_text(encoding="utf-8"))
    assert state["feed_position"] == 2000  # last periodic flush before the crash

    with pytest.raises(HealthCheckError, match="already exists"):
        session(tmp_path / "r").health_checks()

    resumed = session(tmp_path / "r", resume=True)
    report = resumed.run()
    assert report["restarts"] == 1
    assert report["counters"]["events"] == clean["counters"]["events"]  # nothing skipped, nothing doubled
    for acct in state["accounts"].values():
        Portfolio.from_state(acct["broker"]["portfolio"])  # persisted ledgers reconcile
    for a in resumed.accounts:
        a.broker.portfolio.assert_reconciled()

    orders = Counter((r["account"], r["order_id"]) for r in audit(resumed, "order"))
    assert max(orders.values()) == 1  # replayed events after the last flush were not double-logged
    restarts = [r for r in audit(resumed, "cancel") if "process restart" in (r.get("reason") or "")]
    working_at_flush = sum(len(a["broker"]["working"]) for a in state["accounts"].values())
    assert len(restarts) == working_at_flush


def test_resume_refuses_changed_configuration(tmp_path: Path) -> None:
    with pytest.raises(SimulatedCrash):
        session(tmp_path, faults={"crash_at_event": 1200}).run()
    with pytest.raises(HealthCheckError, match="does not match"):
        session(tmp_path, resume=True, risk=risk_config(max_order_shares=10)).run()


def test_resume_needs_saved_state(tmp_path: Path) -> None:
    with pytest.raises(HealthCheckError, match="nothing to resume"):
        session(tmp_path, resume=True).health_checks()


def test_unknown_session_date_is_refused(tmp_path: Path) -> None:
    with pytest.raises(HealthCheckError, match="not available"):
        session(tmp_path, feed={"session_date": "2026-06-02"}).health_checks()
