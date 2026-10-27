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


DATA_ONLY_MODULE = Path("feeds") / "alpaca.py"  # the single module allowed to touch the network


def _imports(path: Path) -> list[str]:
    names = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_only_the_market_data_module_imports_a_network_client() -> None:
    offenders = [f"{p.relative_to(SRC)}: {n}" for p in SRC.rglob("*.py") if p.relative_to(SRC) != DATA_ONLY_MODULE
                 for n in _imports(p) if n.split(".")[0] in NETWORK_MODULES]
    assert offenders == []
    allowed = {n.split(".")[0] for n in _imports(SRC / DATA_ONLY_MODULE)} & NETWORK_MODULES
    assert allowed == {"urllib"}  # stdlib HTTP only; no broker SDKs, no sockets, no websockets


def test_only_the_market_data_module_reads_the_environment() -> None:
    hits = [str(p.relative_to(SRC)) for p in SRC.rglob("*.py")
            if ("os.environ" in (t := p.read_text(encoding="utf-8")) or "getenv" in t)
            and p.relative_to(SRC) != DATA_ONLY_MODULE]
    assert hits == []


def test_market_data_module_can_only_GET_two_read_only_endpoints() -> None:
    from atlas.feeds import alpaca

    text = (SRC / DATA_ONLY_MODULE).read_text(encoding="utf-8")
    assert alpaca.DATA_HOST == "https://data.alpaca.markets"
    assert alpaca.ALLOWED_PATHS == {"/v2/stocks/quotes/latest", "/v2/stocks/trades/latest"}
    assert 'method="GET"' in text and "POST" not in text and "DELETE" not in text and "PATCH" not in text
    fetch = alpaca.http_fetch(alpaca.AlpacaCredentials("k", "s"))
    for bad in ("/v2/orders", "/v2/positions", "/v2/account"):
        with pytest.raises(PermissionError):
            fetch(bad, {})


def test_history_download_adds_only_two_read_only_market_data_endpoints() -> None:
    from atlas.feeds import alpaca

    assert alpaca.HISTORY_PATHS == {"/v2/stocks/quotes", "/v2/stocks/trades"}
    live = alpaca.http_fetch(alpaca.AlpacaCredentials("k", "s"))  # what paper sessions use
    with pytest.raises(PermissionError):
        live("/v2/stocks/quotes", {})
    hist = alpaca.http_fetch(alpaca.AlpacaCredentials("k", "s"), allowed=alpaca.HISTORY_PATHS)
    for bad in ("/v2/orders", "/v2/positions", "/v2/account", "/v2/stocks/quotes/latest"):
        with pytest.raises(PermissionError):
            hist(bad, {})
    with pytest.raises(PermissionError):
        alpaca.http_fetch(alpaca.AlpacaCredentials("k", "s"), allowed=frozenset({"/v2/orders"}))
    with pytest.raises(PermissionError):
        next(alpaca.iter_history(lambda p, q: {}, "/v2/orders", "SPY", "a", "b"))


def test_no_trading_host_or_order_endpoint_appears_in_source() -> None:
    forbidden = ("api.alpaca.markets", "paper-api.alpaca.markets", "/v2/orders", "/v1/orders",
                 "trader/v1", "accounts/{", "/orders")
    hits = [(str(p.relative_to(SRC)), f) for p in SRC.rglob("*.py") for f in forbidden
            if f in p.read_text(encoding="utf-8").replace("data.alpaca.markets", "")]
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


def test_every_bot_file_the_configs_use_is_committed() -> None:
    """Configs must never depend on untracked local files (e.g. git-ignored checkpoints)."""
    import subprocess

    import yaml

    tracked = set(subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True).stdout.split())
    for cfg in ("configs/paper_live.yaml", "configs/paper.yaml"):
        raw = yaml.safe_load((REPO / cfg).read_text(encoding="utf-8"))
        for src in [raw["candidate"], *raw.get("shadows", [])]:
            if src.get("path"):
                assert src["path"] in tracked, f"{cfg} uses {src['path']}, which is not committed"
