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
    assert roles == {"champion": "champion", "shadow-1": "shadow", "Flip": "control", "Hold": "benchmark"}
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


class _StopAfter:
    """Wraps a feed and requests a graceful stop (paper-stop) after n events."""

    def __init__(self, inner, n: int, session_dir: Path) -> None:
        self.inner, self.n, self.session_dir = inner, n, session_dir
        self.symbols = inner.symbols

    def connect(self):
        self.inner.connect()

    def reconnect(self):
        self.inner.reconnect()

    @property
    def position(self):
        return self.inner.position

    def seek(self, p):
        self.inner.seek(p)

    def next_event(self):
        if self.inner.position == self.n:
            from atlas.paper.session import STOP_FILE

            (self.session_dir / STOP_FILE).write_text("stop", encoding="utf-8")
        return self.inner.next_event()


def test_graceful_stop_writes_report_and_resumes(tmp_path: Path) -> None:
    probe = session(tmp_path / "probe")
    probe.health_checks()
    target = tmp_path / "s" / "paper" / "test_session"
    s = session(tmp_path / "s", feed_override=_StopAfter(probe.feed, 1500, target))
    report = s.run()
    assert report["stopped_by"] == "operator (paper-stop)"
    assert report["kill_switch"] is None
    assert report["counters"]["events"] < 1700
    assert not (s.dir / "STOP").exists()
    assert json.loads((s.dir / "live.json").read_text(encoding="utf-8"))["status"].startswith("stopped by")
    resumed = session(tmp_path / "s", resume=True).run()
    assert resumed["restarts"] == 1 and resumed["stopped_by"] is None
    assert resumed["counters"]["events"] > report["counters"]["events"]


def test_paper_status_and_check_commands(tmp_path: Path, capsys) -> None:
    import yaml
    from paper_helpers import RISK, REPO, paper_dict

    from atlas.cli import EXIT_OK, main

    (tmp_path / "risk.yaml").write_text(yaml.safe_dump(RISK), encoding="utf-8")
    cfg = paper_dict(tmp_path)
    cfg.update(simulator_config=str(REPO / "configs/simulator.yaml"), risk_config=str(tmp_path / "risk.yaml"))
    path = tmp_path / "paper.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    assert main(["paper-check", "--config", str(path)]) == EXIT_OK
    assert "READY" in capsys.readouterr().out
    assert not (tmp_path / "paper").exists()  # the preflight never creates a session

    s = session(tmp_path)
    s.run()
    assert main(["paper-status", "--config", str(path)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "champion" in out and "Hold" in out and "simulated orders only" in out


def test_second_process_cannot_run_the_same_session(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys as _sys

    from atlas.paper.session import LOCK_FILE, pid_alive

    s = session(tmp_path)
    s.dir.mkdir(parents=True, exist_ok=True)
    other = subprocess.Popen([_sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (s.dir / LOCK_FILE).write_text(str(other.pid), encoding="utf-8")
        assert pid_alive(other.pid)
        with pytest.raises(HealthCheckError, match="already running"):
            s.health_checks()
    finally:
        other.kill()
        other.wait()
    assert not pid_alive(other.pid)
    s2 = session(tmp_path)  # stale lock from a dead process: taken over
    s2.run()
    assert not (s2.dir / LOCK_FILE).exists()  # released at the end
    assert pid_alive(os.getpid())


def test_buy_and_hold_and_coin_flip_are_permanent(tmp_path: Path) -> None:
    """They're always present; a config can neither remove nor duplicate them."""
    no_shadows = session(tmp_path / "a", shadows=[])
    no_shadows.health_checks()
    assert {a.account_id: a.model["family"] for a in no_shadows.accounts} == {
        "champion": "baseline_momentum", "Flip": "baseline_coin_flip", "Hold": "baseline_buy_hold"}

    duplicates = session(tmp_path / "b", shadows=[{"source": "baseline", "baseline": "baseline_coin_flip"},
                                                  {"source": "baseline", "baseline": "baseline_buy_hold"},
                                                  {"source": "baseline", "baseline": "baseline_random"}])
    duplicates.health_checks()
    families = [a.model["family"] for a in duplicates.accounts]
    assert families.count("baseline_coin_flip") == 1 and families.count("baseline_buy_hold") == 1
    assert [a.account_id for a in duplicates.accounts] == ["champion", "shadow-1", "Flip", "Hold"]


def test_bots_can_be_named_and_reserved_names_are_refused(tmp_path: Path) -> None:
    named = session(tmp_path / "a", candidate={"source": "baseline", "baseline": "baseline_momentum", "name": "Momo"},
                    shadows=[{"source": "baseline", "baseline": "baseline_random", "name": "Dice"}])
    named.health_checks()
    assert [a.account_id for a in named.accounts] == ["Momo", "Dice", "Flip", "Hold"]
    assert [a.role for a in named.accounts] == ["champion", "shadow", "control", "benchmark"]
    clash = session(tmp_path / "b", shadows=[{"source": "baseline", "baseline": "baseline_random", "name": "flip"}])
    with pytest.raises(HealthCheckError, match="unique"):
        clash.health_checks()
