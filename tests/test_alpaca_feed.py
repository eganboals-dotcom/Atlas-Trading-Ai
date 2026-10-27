"""Alpaca market-data feed, driven by a fake Alpaca API and a fake clock (no network, no keys)."""

from __future__ import annotations

import json
import math
from datetime import date, datetime, time, timezone
from pathlib import Path

import pytest
from paper_helpers import REPO, risk_config
from helpers import sim_config

from atlas.dashboard import data
from atlas.feeds.alpaca import (
    QUOTES_PATH, TRADES_PATH, AlpacaAuthError, AlpacaPollingFeed, load_credentials, parse_rfc3339_ns,
)
from atlas.paper.config import PaperConfig
from atlas.paper.feed import FeedDisconnected
from atlas.paper.session import PaperSession
from atlas.timeutils import MARKET_TZ, NS_PER_S, datetime_to_ns

DAY = date(2026, 6, 1)  # a Monday


def iso(ns: int) -> str:
    sec, frac = divmod(ns, NS_PER_S)
    return datetime.fromtimestamp(sec, tz=MARKET_TZ).astimezone().strftime("%Y-%m-%dT%H:%M:%S") \
        if False else datetime.utcfromtimestamp(sec).strftime("%Y-%m-%dT%H:%M:%S") + f".{frac:09d}Z"


class FakeAlpaca:
    """Deterministic stand-in for the two read-only endpoints, plus a controllable clock."""

    def __init__(self, start: datetime, fail_calls: set[int] | None = None) -> None:
        self.now = datetime_to_ns(start)
        self.calls: list[tuple[str, int]] = []
        self.fail_calls = fail_calls or set()
        self.mid = {"SPY": 580.0, "QQQ": 500.0}
        self.k = 0

    def clock(self) -> int:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(1, int(seconds * NS_PER_S))

    def fetch(self, path: str, params: dict[str, str]) -> dict:
        self.calls.append((path, self.now))
        self.now += 30_000_000  # 30 ms round trip
        if len(self.calls) in self.fail_calls:
            raise FeedDisconnected("simulated network error")
        syms = params["symbols"].split(",")
        t = self.now - 50_000_000
        if path == QUOTES_PATH:
            self.k += 1
            out = {}
            for i, s in enumerate(syms):
                if self.k % 7 == 3 and i == 1:  # QQQ sometimes unchanged between polls
                    continue
                self.mid[s] *= math.exp(0.0002 * math.sin(self.k * (i + 1) * 0.7))
                bid = round(self.mid[s] - 0.01, 2)
                out[s] = {"bp": bid, "bs": 300, "ap": round(bid + 0.02, 2), "as": 400, "t": iso(t), "c": ["R"]}
            return {"quotes": out}
        assert path == TRADES_PATH
        return {"trades": {s: {"p": round(self.mid[s], 2), "s": 10, "t": iso(t - 1000), "i": self.k, "c": ["@"]}
                           for s in syms}}


def feed(fake: FakeAlpaca, minutes: int = 3, **kw) -> AlpacaPollingFeed:
    return AlpacaPollingFeed(["SPY", "QQQ"], DAY, fake.fetch, start=time(9, 30), end=time(9, 30 + minutes),
                             clock=fake.clock, sleep=fake.sleep, **kw)


def test_rfc3339_parsing_is_exact() -> None:
    assert parse_rfc3339_ns("2026-06-01T13:30:00.000000001Z") % NS_PER_S == 1
    assert parse_rfc3339_ns("2026-06-01T09:30:00-04:00") == parse_rfc3339_ns("2026-06-01T13:30:00Z")


def test_credentials_from_env_or_dotenv_and_never_printed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(AlpacaAuthError):
        load_credentials(tmp_path / "missing.env")
    env = tmp_path / ".env"
    env.write_text("# comment\nAPCA_API_KEY_ID=PKTEST123\nAPCA_API_SECRET_KEY='s3cret'\n", encoding="utf-8")
    creds = load_credentials(env)
    assert creds.key_id == "PKTEST123" and creds.secret == "s3cret"
    assert "s3cret" not in repr(creds) and "s3cret" not in str(creds)
    monkeypatch.setenv("APCA_API_KEY_ID", "PKENV")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "envsecret")
    assert load_credentials(env).key_id == "PKENV"


def test_feed_waits_for_open_emits_only_new_data_and_stops_at_close() -> None:
    fake = FakeAlpaca(datetime.combine(DAY, time(9, 29, 50), tzinfo=MARKET_TZ))
    f = feed(fake, minutes=2)
    f.connect()
    events = []
    while (e := f.next_event()) is not None:
        events.append(e)
    open_ns = datetime_to_ns(datetime.combine(DAY, time(9, 30), tzinfo=MARKET_TZ))
    assert all(ts >= open_ns for p, ts in fake.calls[1:])  # no polling before the open (after connect)
    assert fake.now >= datetime_to_ns(datetime.combine(DAY, time(9, 32), tzinfo=MARKET_TZ))
    quotes = [e for e in events if e.event_type == "quote"]
    trades = [e for e in events if e.event_type == "trade"]
    assert quotes and trades
    for sym in ("SPY", "QQQ"):
        ts = [q.event_ts for q in quotes if q.symbol == sym]
        assert ts == sorted(set(ts))  # unchanged quotes are not re-emitted
    recv = [e.receive_ts for e in events]
    assert recv == sorted(recv)  # local arrival clock never runs backwards
    per_minute = sum(1 for p, ts in fake.calls if ts < open_ns + 60 * NS_PER_S)
    assert per_minute <= 200  # Alpaca free-plan limit
    assert {e.source for e in events} == {"alpaca-iex"}


def test_network_failure_raises_disconnect_then_reconnects() -> None:
    fake = FakeAlpaca(datetime.combine(DAY, time(9, 30), tzinfo=MARKET_TZ), fail_calls={5})
    f = feed(fake, minutes=1)
    f.connect()
    with pytest.raises(FeedDisconnected):
        for _ in range(100):
            f.next_event()
    f.reconnect()
    assert f.next_event() is not None


def live_config(tmp_path: Path) -> PaperConfig:
    return PaperConfig.model_validate({
        "mode": "paper", "live_orders_enabled": False, "session_name": "live_{date}",
        "simulator_config": "x", "risk_config": "x",
        "feed": {"kind": "alpaca", "first_quote_timeout_ms": 60_000, "alpaca": {}},
        "candidate": {"source": "baseline", "baseline": "baseline_momentum"},
        "shadows": [{"source": "baseline", "baseline": "baseline_random"},
                    {"source": "baseline", "baseline": "baseline_mean_reversion"}],
        "agent_limits": {"max_position_shares": 1000, "max_order_shares": 500, "max_position_frac": 0.5},
        "decision_interval_ms": 2000, "alerts": {"console": False}, "output_dir": str(tmp_path / "paper"),
        "state_flush_events": 100, "dashboard": {"live_snapshot_ms": 500},
    })


def run_live_session(tmp_path: Path, minutes: int = 6) -> PaperSession:
    fake = FakeAlpaca(datetime.combine(DAY, time(9, 29, 55), tzinfo=MARKET_TZ))
    risk = risk_config(stale_data_ms=30_000, max_clock_skew_ms=3_000)
    s = PaperSession(live_config(tmp_path), sim_config(stale_quote_ms=30_000.0), risk, base_dir=REPO,
                     feed=feed(fake, minutes=minutes))
    s.run()
    return s


def test_full_live_session_writes_leaderboard_snapshot(tmp_path: Path) -> None:
    s = run_live_session(tmp_path)
    assert s.dir.name == "live_2026-06-01"
    report = json.loads((s.dir / "daily_report.json").read_text(encoding="utf-8"))
    assert report["feed"]["kind"] == "alpaca" and report["live_orders_enabled"] is False
    assert report["counters"]["events"] > 300
    live = data.load_live(s.dir)
    assert live["status"] == "ended" and live["simulated"] is True
    assert {a["account"] for a in live["accounts"]} == {"champion", "shadow-1", "shadow-2", "Flip", "Hold"}
    board = data.leaderboard(live)
    ranked = [r for r in board if r["rank"]]
    assert [r["rank"] for r in ranked] == [1, 2, 3, 4]  # the coin flip is a ranked bot
    assert [r["net_pnl"] for r in ranked] == sorted((r["net_pnl"] for r in ranked), reverse=True)
    assert board[-1]["role"] == "benchmark" and board[-1]["rank"] is None
    assert any(a["fills"] for a in live["accounts"])
    assert live["recent_trades"] and {"symbol", "side", "qty", "price"} <= set(live["recent_trades"][0])
    for a in s.accounts:
        a.broker.portfolio.assert_reconciled()


def test_live_session_refuses_a_date_that_is_not_today(tmp_path: Path) -> None:
    from atlas.paper.session import HealthCheckError

    cfg = live_config(tmp_path).model_copy(update={"feed": live_config(tmp_path).feed.model_copy(
        update={"session_date": date(2020, 1, 6)})})
    with pytest.raises(HealthCheckError, match="not today"):
        PaperSession(cfg, sim_config(), risk_config()).health_checks()


def test_live_page_renders_leaderboard(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("streamlit")
    import sys

    from streamlit.testing.v1 import AppTest

    s = run_live_session(tmp_path, minutes=3)
    monkeypatch.setattr(sys, "argv", ["app.py", "--paper-dir", str(s.dir.parent), "--session", s.dir.name])
    at = AppTest.from_file(str(REPO / "src/atlas/dashboard/app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    heads = [h.value for h in at.subheader]
    assert "Leaderboard" in heads and "Recent simulated trades" in heads
    assert any(m.label == "Leader" for m in at.metric)


def test_terminal_leaderboard_renders_and_watch_once(tmp_path: Path, capsys) -> None:
    from atlas.cli import EXIT_OK, main
    from atlas.dashboard.terminal import render

    s = run_live_session(tmp_path, minutes=3)
    text = render(data.load_live(s.dir), color=False)
    assert "LEADERBOARD" in text and "RECENT SIMULATED TRADES" in text and "simulated orders only" in text
    lines = text.splitlines()
    board = lines[lines.index(next(l for l in lines if l.startswith(" #"))) + 2:]
    assert board[0].lstrip().startswith("1 ") and any(l.lstrip().startswith("- ") for l in board[:5])
    assert "\x1b[" not in text  # no color codes when color is off
    assert main(["paper-watch", "--session-dir", str(s.dir), "--once"]) == EXIT_OK
    assert "LEADERBOARD" in capsys.readouterr().out


def fund_session(tmp_path: Path, day: date, minutes: int = 4, **cfg_updates) -> PaperSession:
    fake = FakeAlpaca(datetime.combine(day, time(9, 29, 55), tzinfo=MARKET_TZ))
    cfg = live_config(tmp_path).model_copy(update={"fund": __import__("atlas.paper.config", fromlist=["FundConfig"])
                                                   .FundConfig(total_capital=90_000), **cfg_updates})
    f = AlpacaPollingFeed(["SPY", "QQQ"], day, fake.fetch, start=time(9, 30), end=time(9, 30 + minutes),
                          clock=fake.clock, sleep=fake.sleep)
    return PaperSession(cfg, sim_config(stale_quote_ms=30_000.0), risk_config(stale_data_ms=30_000), base_dir=REPO,
                        feed=f)


def test_fund_splits_equally_flattens_at_close_and_compounds(tmp_path: Path) -> None:
    from atlas.paper.fund import FundLedger

    day1 = fund_session(tmp_path, DAY)
    day1.run()
    bots = [a for a in day1.accounts if a.role != "benchmark"]
    assert len(bots) == 4  # champion, two shadows, and the permanent coin-flip bot
    assert all(a.broker.portfolio.initial_cash == pytest.approx(22_500) for a in day1.accounts)  # 90k / 4 bots
    assert all(not any(p.quantity for p in a.broker.portfolio.positions.values()) for a in day1.accounts)
    ledger = json.loads((tmp_path / "paper" / "fund_ledger.json").read_text(encoding="utf-8"))
    d1 = ledger["days"][0]
    assert d1["final"] is True and set(d1["bots"]) == {"champion", "shadow-1", "shadow-2", "Flip"}
    ends = {b: v["end"] for b, v in d1["bots"].items()}
    assert any(abs(e - 22_500) > 0.01 for e in ends.values())  # someone made or lost money

    day2 = fund_session(tmp_path, date(2026, 6, 2))
    day2.run()
    starts = {a.account_id: a.broker.portfolio.initial_cash for a in day2.accounts if a.role != "benchmark"}
    assert starts == pytest.approx(ends)  # each bot starts day 2 with exactly what it ended day 1 with
    bench = next(a for a in day2.accounts if a.role == "benchmark")
    assert bench.broker.portfolio.initial_cash == pytest.approx(22_500)  # benchmark: fresh share daily
    live = data.load_live(day2.dir)
    assert live["fund"]["days_settled"] == 2
    assert live["fund"]["equity"] == pytest.approx(sum(a["equity"] for a in live["accounts"] if a["role"] != "benchmark"))
    champ = next(a for a in live["accounts"] if a["account"] == "champion")
    assert champ["inception_capital"] == pytest.approx(22_500)
    assert champ["total_pnl"] == pytest.approx(champ["equity"] - 22_500)
    assert len(FundLedger(tmp_path / "paper" / "fund_ledger.json", 90_000).data["days"]) == 2


def test_fund_refuses_a_changed_lineup_until_reset(tmp_path: Path) -> None:
    from atlas.paper.config import CandidateSource
    from atlas.paper.session import HealthCheckError

    fund_session(tmp_path, DAY, minutes=2).run()
    changed = fund_session(tmp_path, date(2026, 6, 2), minutes=2,
                           candidate=CandidateSource(source="baseline", baseline="baseline_buy_hold"))
    with pytest.raises(HealthCheckError, match="lineup"):
        changed.health_checks()
    changed.reset_fund = True
    changed.dir.mkdir(parents=True, exist_ok=True)
    changed.health_checks()
    archived = list((tmp_path / "paper").glob("fund_ledger.*.json"))
    assert len(archived) == 1


def test_leaderboard_window_api_is_read_only_and_page_is_offline(tmp_path: Path) -> None:
    import inspect
    import re

    from atlas.dashboard.window import HTML, Api

    s = fund_session(tmp_path, DAY, minutes=2)
    s.run()
    api = Api(s.dir)
    public = [n for n, _ in inspect.getmembers(api, inspect.ismethod) if not n.startswith("_")]
    assert public == ["live"]  # the page can only read the snapshot
    d = api.live()
    assert d["live"]["fund"]["bots"] == 4 and len(d["rows"]) == 5 and d["age"] >= 0
    assert d["live"]["quotes"].keys() == {"SPY", "QQQ"}
    assert Api(tmp_path / "missing").live() is None
    page = HTML.read_text(encoding="utf-8")
    assert not re.search(r"https?://", page)  # no CDN, fonts, or any network resource
    assert "pywebview.api.live" in page


def recording_session(tmp_path: Path, minutes: int = 3) -> PaperSession:
    from atlas.paper.config import RecordConfig

    fake = FakeAlpaca(datetime.combine(DAY, time(9, 29, 55), tzinfo=MARKET_TZ))
    cfg = live_config(tmp_path).model_copy(update={"record": RecordConfig(enabled=True, dataset="rec_test",
                                                                          root=tmp_path / "data")})
    return PaperSession(cfg, sim_config(stale_quote_ms=30_000.0), risk_config(stale_data_ms=30_000), base_dir=REPO,
                        feed=feed(fake, minutes=minutes))


def test_recorder_saves_exactly_what_the_session_received(tmp_path: Path) -> None:
    from atlas.data.providers import ParquetProvider, load_raw
    from atlas.paper.recorder import read_raw_csv

    s = recording_session(tmp_path)
    report = s.run()
    raw = tmp_path / "data" / "raw" / "rec_test" / "2026-06-01.csv"
    events = read_raw_csv(raw)
    assert len(events) == report["counters"]["events"] > 100
    assert report["recording"]["events"] == len(events) and report["recording"]["session"]["rows"] == len(events)
    pq = ParquetProvider(tmp_path / "data" / "processed" / "rec_test")
    assert pq.sessions() == [DAY]
    assert load_raw(pq.session_ref(DAY)) == events  # lossless round trip, hash-checked by the provider
    manifest = json.loads((tmp_path / "data" / "processed" / "rec_test" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sessions"][0]["quotes"] > 0 and manifest["sessions"][0]["trades"] > 0
    assert "IEX" in manifest["fidelity"] and "never" not in manifest["license_notes"].lower()


def test_recorder_writes_a_header_into_an_empty_leftover_file(tmp_path: Path) -> None:
    # A file left empty by an earlier start used to get rows but no header, so finalize failed.
    from atlas.paper.recorder import read_raw_csv

    raw = tmp_path / "data" / "raw" / "rec_test" / "2026-06-01.csv"
    raw.parent.mkdir(parents=True)
    raw.touch()
    report = recording_session(tmp_path).run()
    assert raw.read_text(encoding="utf-8").startswith("event_type,")
    assert report["recording"]["session"]["rows"] == len(read_raw_csv(raw)) > 100


def test_read_raw_csv_accepts_a_headerless_recording(tmp_path: Path) -> None:
    from atlas.paper.recorder import read_raw_csv

    s = recording_session(tmp_path)
    s.run()
    raw = tmp_path / "data" / "raw" / "rec_test" / "2026-06-01.csv"
    events = read_raw_csv(raw)
    headless = tmp_path / "headless.csv"
    headless.write_text(raw.read_text(encoding="utf-8").split("\n", 1)[1], encoding="utf-8")
    assert read_raw_csv(headless) == events


def test_recorder_failure_never_stops_trading(tmp_path: Path, monkeypatch) -> None:
    from atlas.paper.recorder import MarketDataRecorder

    calls = {"n": 0}
    real = MarketDataRecorder.record

    def flaky(self, event):
        calls["n"] += 1
        if calls["n"] == 50:
            raise OSError("disk full")
        return real(self, event)

    monkeypatch.setattr(MarketDataRecorder, "record", flaky)
    s = recording_session(tmp_path)
    report = s.run()
    assert report["counters"]["events"] > 100  # the session ran to the close
    assert "disk full" in report["recording"]["error"]
    alerts = (s.dir / "alerts.jsonl").read_text(encoding="utf-8")
    assert "recording_off" in alerts and report["kill_switch"] is None
    for a in s.accounts:
        a.broker.portfolio.assert_reconciled()
