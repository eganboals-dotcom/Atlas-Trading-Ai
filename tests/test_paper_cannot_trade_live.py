"""Paper mode must be technically unable to place a live order."""

from __future__ import annotations

import ast
import re
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
NETWORK_MODULES = {"socket", "ssl", "http", "urllib", "requests", "httpx", "httpx2", "aiohttp", "websocket",
                   "websockets", "ftplib", "smtplib", "grpc", "schwab", "tda", "anthropic", "alpaca", "alpaca_trade_api"}


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


DATA_ONLY_MODULE = Path("feeds") / "alpaca.py"  # market data: the only module allowed to read the environment
AI_READER_MODULE = Path("feeds") / "anthropic_api.py"  # the AI news reader's Claude API calls
SEC_MODULE = Path("feeds") / "sec.py"  # company filings from SEC EDGAR
LOCAL_AI_MODULE = Path("feeds") / "local_llm.py"  # the free AI model on this PC (Ollama, loopback only)
FINRA_MODULE = Path("feeds") / "finra.py"  # FINRA's public short-interest files
NETWORK_ALLOWED = {DATA_ONLY_MODULE: {"urllib"}, AI_READER_MODULE: {"anthropic"}, SEC_MODULE: {"urllib"},
                   LOCAL_AI_MODULE: {"urllib"}, FINRA_MODULE: {"urllib"}}  # module -> the only network client it may import


def _imports(path: Path) -> list[str]:
    names = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_only_the_market_data_module_imports_a_network_client() -> None:
    offenders = [f"{p.relative_to(SRC)}: {n}" for p in SRC.rglob("*.py") if p.relative_to(SRC) not in NETWORK_ALLOWED
                 for n in _imports(p) if n.split(".")[0] in NETWORK_MODULES]
    assert offenders == []
    for module, only in NETWORK_ALLOWED.items():
        # alpaca.py: stdlib HTTP only; anthropic_api.py: Anthropic's package only. No broker SDKs, no sockets.
        assert {n.split(".")[0] for n in _imports(SRC / module)} & NETWORK_MODULES == only, module


def test_ai_reader_module_can_only_ask_claude_about_headlines(monkeypatch) -> None:
    """The AI reader may make one call, messages.create, to Anthropic's default API host; nothing else."""
    text = (SRC / AI_READER_MODULE).read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):  # drop docstrings and other bare strings, keep all code and real string values
        if hasattr(node, "body") and isinstance(node.body, list):
            node.body = [n for n in node.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    code = ast.unparse(tree)
    assert text.count("client.messages.create(") == 1
    assert set(re.findall(r"client\.([a-z_.]+)\(", code)) == {"messages.create"}
    for banned in ("base_url", "http://", "https://", ".beta", "batches", "files.", "tools=", "/v2/", "alpaca.markets",
                   "os.environ", "getenv", "submit", "order"):
        assert banned not in code, banned
    from atlas.feeds import anthropic_api

    monkeypatch.delenv(anthropic_api.KEY_ENV, raising=False)
    with pytest.raises(PermissionError):  # no key: refuses before any client exists
        anthropic_api.claude_classifier("claude-haiku-5-5", env_file=None)


def test_sec_module_can_only_GET_public_edgar_files(monkeypatch) -> None:
    """Two JSON endpoints and, since 2026-10-28, the quarterly insider-transactions zips; GET only."""
    from atlas.feeds import sec

    text = (SRC / SEC_MODULE).read_text(encoding="utf-8")
    assert set(re.findall(r"https://[a-z.]+", text)) == {"https://www.sec.gov", "https://data.sec.gov"}
    assert 'method="GET"' in text and "POST" not in text and "DELETE" not in text and "PATCH" not in text
    get = sec.http_get("Atlas test test@example.com")
    get_file = sec.http_get_file("Atlas test test@example.com")
    zip_url = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/2016q1_form345.zip"
    for bad in ("https://api.alpaca.markets/v2/orders", "https://www.sec.gov/cgi-bin/browse-edgar",
                "https://data.sec.gov/submissions/CIK0000320193.json", "http://data.sec.gov/api/xbrl/companyfacts/CIK1",
                "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json?x=1", zip_url):
        with pytest.raises(PermissionError):
            get(bad)  # the JSON getter never fetches a zip
    for bad in ("https://www.sec.gov/files/company_tickers.json", zip_url + "?x", zip_url.replace("2016q1", "2016q5"),
                "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/../../cgi-bin/x.zip",
                zip_url.replace("https://www.sec.gov", "https://evil.example"), zip_url.replace("https", "http")):
        with pytest.raises(PermissionError):
            get_file(bad)
    monkeypatch.delenv(sec.UA_ENV, raising=False)
    with pytest.raises(PermissionError):  # no contact address: refuses before any request
        sec.user_agent(env_file=None)


def test_local_ai_module_can_only_reach_ollama_on_this_pc(monkeypatch) -> None:
    """Loopback only, two Ollama paths only, no proxies (an HTTP_PROXY would carry headlines off the PC)."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from atlas.feeds import local_llm

    text = (SRC / LOCAL_AI_MODULE).read_text(encoding="utf-8")
    code = text.replace(ast.get_docstring(ast.parse(text), clean=False), "")
    assert set(re.findall(r"https?://[a-z0-9.]+", code)) == {"http://127.0.0.1"}
    for banned in ("os.environ", "getenv", "alpaca", "submit", "order", "socket"):
        assert banned not in code, banned
    for bad in ("http://example.com:11434", "https://127.0.0.1:11434", "http://127.0.0.1.evil.com:11434",
                "http://user@127.0.0.1:11434", "http://10.0.0.5:11434", "http://127.0.0.1:11434/x"):
        with pytest.raises(PermissionError):
            local_llm.check_host(bad)
        with pytest.raises(PermissionError):
            local_llm.local_classifier("m", host=bad)
    with pytest.raises(PermissionError):
        local_llm._request(local_llm.HOST, "/api/pull", {}, 1.0)
    with pytest.raises(PermissionError):
        local_llm._request(local_llm.HOST, "/api/delete", None, 1.0)

    seen: list[tuple[str, str, dict]] = []

    class Fake(BaseHTTPRequestHandler):  # stands in for Ollama on 127.0.0.1
        def log_message(self, *a):
            pass

        def _send(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            seen.append(("GET", self.path, {}))
            self._send({"models": [{"name": "gemma3:12b"}]})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(("POST", self.path, req))
            notes = [{"i": 0, "company": "Company A", "direction": 0.7, "importance": 0.9, "category": "earnings"}]
            self._send({"message": {"content": json.dumps({"notes": notes})}, "done_reason": "stop"})

    server = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("HTTP_PROXY", "http://10.255.255.1:9")  # would hang or fail if the proxy were used
    monkeypatch.setenv("NO_PROXY", "")
    try:
        assert local_llm.not_ready("gemma3:12b", host) is None
        assert "ollama pull qwen3:14b" in local_llm.not_ready("qwen3:14b", host)
        notes = local_llm.local_classifier("gemma3:12b", host=host)([{"i": 0, "headline": "Company A beats",
                                                                     "companies": ["Company A"]}])
    finally:
        server.shutdown()
    assert notes[0]["direction"] == 0.7
    assert [(m, p) for m, p, _ in seen] == [("GET", "/api/tags"), ("GET", "/api/tags"), ("POST", "/api/chat")]
    sent = seen[-1][2]
    assert sent["model"] == "gemma3:12b" and sent["format"]["required"] == ["notes"] and sent["stream"] is False
    assert "Company A beats" in sent["messages"][1]["content"]
    assert "Ollama is not running" in local_llm.not_ready("gemma3:12b", "http://127.0.0.1:9")  # nothing there


def test_finra_module_can_only_GET_short_interest_files() -> None:
    from atlas.feeds import finra

    text = (SRC / FINRA_MODULE).read_text(encoding="utf-8")
    code = text.replace(ast.get_docstring(ast.parse(text), clean=False), "")
    assert set(re.findall(r"https://[a-z.]+", code)) == {"https://cdn.finra.org"}
    assert 'method="GET"' in code and "POST" not in code and "DELETE" not in code and "PATCH" not in code
    for banned in ("os.environ", "getenv", "alpaca", "order", "socket", "Authorization"):
        assert banned not in code, banned
    get = finra.http_get_file()
    ok = "https://cdn.finra.org/equity/otcmarket/biweekly/shrt20240115.csv"
    for bad in (ok + "?x=1", ok.replace("shrt20240115", "shrt2024011"), ok.replace("biweekly", "daily"),
                ok.replace("https://cdn.finra.org", "https://api.finra.org"), ok.replace("https", "http"),
                "https://cdn.finra.org/equity/otcmarket/biweekly/../../../x.csv", "https://api.alpaca.markets/v2/orders"):
        with pytest.raises(PermissionError):
            get(bad)


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


def test_history_download_adds_only_three_read_only_market_data_endpoints() -> None:
    from atlas.feeds import alpaca

    assert alpaca.HISTORY_PATHS == {"/v2/stocks/quotes", "/v2/stocks/trades", "/v2/stocks/bars"}
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


def test_news_download_adds_only_one_read_only_market_data_endpoint() -> None:
    from atlas.feeds import alpaca

    assert alpaca.NEWS_PATHS == {"/v1beta1/news"}
    live = alpaca.http_fetch(alpaca.AlpacaCredentials("k", "s"))  # what paper sessions use
    with pytest.raises(PermissionError):
        live("/v1beta1/news", {})
    news = alpaca.http_fetch(alpaca.AlpacaCredentials("k", "s"), allowed=alpaca.NEWS_PATHS)
    for bad in ("/v2/orders", "/v2/positions", "/v2/account", "/v2/stocks/quotes/latest", "/v2/stocks/bars"):
        with pytest.raises(PermissionError):
            news(bad, {})
    calls = []
    pages = iter([{"news": [{"id": 1}], "next_page_token": "x"}, {"news": [{"id": 2}], "next_page_token": None}])
    got = list(alpaca.iter_news(lambda p, q: (calls.append(p), next(pages))[1], ["AAPL"], "a", "b",
                                sleep=lambda s: None))
    assert [a["id"] for a in got] == [1, 2] and set(calls) == {"/v1beta1/news"}


def test_crypto_download_adds_only_one_read_only_market_data_endpoint() -> None:
    from atlas.feeds import alpaca

    assert alpaca.CRYPTO_PATHS == {"/v1beta3/crypto/us/bars"}
    live = alpaca.http_fetch(alpaca.AlpacaCredentials("k", "s"))  # what paper sessions use
    with pytest.raises(PermissionError):
        live("/v1beta3/crypto/us/bars", {})
    crypto = alpaca.http_fetch(alpaca.AlpacaCredentials("k", "s"), allowed=alpaca.CRYPTO_PATHS)
    for bad in ("/v2/orders", "/v2/positions", "/v2/account", "/v1beta3/crypto/us/orders", "/v2/stocks/bars",
                "/v1beta3/crypto/us/latest/orderbooks"):
        with pytest.raises(PermissionError):
            crypto(bad, {})
    calls = []
    pages = iter([{"bars": {"BTC/USD": [{"c": 1}]}, "next_page_token": "x"},
                  {"bars": {"BTC/USD": [{"c": 2}]}, "next_page_token": None}])
    got = list(alpaca.iter_crypto_bars(lambda p, q: (calls.append(p), next(pages))[1], "BTC/USD", "a", "b",
                                       sleep=lambda s: None))
    assert [b["c"] for b in got] == [1, 2] and set(calls) == {"/v1beta3/crypto/us/bars"}


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
