"""Paper mode must be technically unable to place a live order."""

from __future__ import annotations

import ast
import importlib.util
import json
import socket
from pathlib import Path

import pytest
from paper_helpers import REPO, audit, paper_config, paper_dict, risk_config, session
from pydantic import ValidationError

from atlas.brokers import BrokerExecution, LiveTradingDisabledError
from atlas.paper.broker import PaperBroker
from atlas.paper.config import PaperConfig
from atlas.paper.session import PaperSession, ensure_not_live, make_broker

SRC = REPO / "src" / "atlas"
NETWORK_MODULES = {"socket", "ssl", "http", "urllib", "requests", "httpx", "aiohttp", "websocket", "websockets",
                   "ftplib", "smtplib", "grpc", "schwab", "tda"}


@pytest.mark.parametrize("override", [
    {"mode": "live"},
    {"mode": "Paper"},
    {"live_orders_enabled": True},
    {"live_orders_enabled": "false"},
    {"api_key": "x"},
    {"broker": "schwab"},
    {"credentials": {"token": "x"}},
])
def test_config_refuses_anything_but_paper_with_live_orders_off(tmp_path: Path, override) -> None:
    with pytest.raises(ValidationError):
        PaperConfig.model_validate({**paper_dict(tmp_path), **override})


@pytest.mark.parametrize("missing", ["mode", "live_orders_enabled", "risk_config", "candidate", "agent_limits"])
def test_missing_safety_settings_fail_closed(tmp_path: Path, missing: str) -> None:
    d = paper_dict(tmp_path)
    del d[missing]
    with pytest.raises(ValidationError):
        PaperConfig.model_validate(d)


def test_session_refuses_a_config_that_bypassed_validation(tmp_path: Path) -> None:
    cfg = paper_config(tmp_path)
    forged = PaperConfig.model_construct(**{**dict(cfg), "live_orders_enabled": True})
    from helpers import sim_config

    with pytest.raises(LiveTradingDisabledError):
        PaperSession(forged, sim_config(), risk_config())


def test_only_paper_broker_can_be_built_or_accepted() -> None:
    from helpers import sim_config

    broker = make_broker("a", sim_config(), 0, lambda r: None)
    assert isinstance(broker, PaperBroker) and type(broker).is_live is False

    class FakeLive(PaperBroker):
        is_live = True

    class OtherBroker(BrokerExecution):
        is_live = False

        def submit(self, *a):  # pragma: no cover
            raise AssertionError

        def process_arrivals(self, *a):  # pragma: no cover
            raise AssertionError

        def cancel_all(self, *a):  # pragma: no cover
            raise AssertionError

        def account(self):  # pragma: no cover
            raise AssertionError

    for bad in (FakeLive("a", sim_config(), 0, lambda r: None), OtherBroker()):
        with pytest.raises(LiveTradingDisabledError):
            ensure_not_live(bad)


def test_no_live_broker_module_exists() -> None:
    for name in ("atlas.brokers.schwab", "atlas.brokers.live", "atlas.paper.live_broker"):
        assert importlib.util.find_spec(name) is None
    import atlas.brokers as brokers

    assert set(brokers.__all__) == {"BrokerExecution", "LiveTradingDisabledError"}


def test_no_source_module_imports_a_network_client() -> None:
    offenders = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                if n.split(".")[0] in NETWORK_MODULES:
                    offenders.append(f"{path.relative_to(SRC)}: {n}")
    assert offenders == []


def test_no_source_module_reads_credentials_from_the_environment() -> None:
    hits = [str(p.relative_to(SRC)) for p in SRC.rglob("*.py")
            if "os.environ" in (t := p.read_text(encoding="utf-8")) or "getenv" in t]
    assert hits == []


def test_full_session_runs_with_networking_disabled(tmp_path: Path, monkeypatch) -> None:
    def no_network(*a, **k):
        raise AssertionError("paper session attempted a network connection")

    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    s = session(tmp_path)
    report = s.run()
    assert report["mode"] == "paper" and report["live_orders_enabled"] is False
    executions = audit(s, "execution")
    assert executions and all(r["simulated"] is True for r in executions)
    assert all(r["simulated"] is True for r in audit(s, "order"))
    assert all(a["simulated"] is True for a in report["accounts"])
    assert "No order was sent to any broker" in (s.dir / "daily_report.md").read_text(encoding="utf-8")
    saved = json.loads((s.dir / "state.json").read_text(encoding="utf-8"))
    assert "live" not in json.dumps(saved["accounts"]).lower().replace("live_", "")
