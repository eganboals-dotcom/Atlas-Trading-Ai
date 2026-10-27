from __future__ import annotations

from pathlib import Path

import pytest
from paper_helpers import RISK, audit, risk_config, session
from pydantic import ValidationError

from atlas.environment.orders import OrderIntent, Side
from atlas.paper.feed import ReplayLiveFeed
from atlas.paper.risk import KillSwitch, RiskManager, request_kill
from atlas.paper.session import HealthCheckError
from atlas.timeutils import NS_PER_S

BUY = lambda q, s="SPY": OrderIntent(s, Side.BUY, q)  # noqa: E731
SELL = lambda q, s="SPY": OrderIntent(s, Side.SELL, q)  # noqa: E731


def rm(**kw) -> RiskManager:
    return RiskManager(risk_config(**kw), "acct", 100_000.0)


def check(r: RiskManager, intent, held=0, price=100.0, now=0):
    return r.check_order(intent, now, price, 100_000.0, {"SPY": held} if held else {}, {"SPY": price})


# --------------------------------------------------------------- per-order rules


def test_symbol_not_allowed_is_rejected() -> None:
    d = check(rm(), BUY(10, "TSLA"))
    assert not d.approved and d.rule == "symbol"


def test_order_size_is_capped_by_shares_and_value() -> None:
    assert check(rm(max_order_shares=50), BUY(80)).quantity == 50
    d = check(rm(max_order_value=1_000), BUY(80))
    assert d.approved and d.quantity == 10 and d.rule == "order_size"


def test_position_limit_rejects_buys_that_would_exceed_it() -> None:
    d = check(rm(max_position_shares=100), BUY(60), held=50)
    assert not d.approved and d.rule == "max_position"
    d = check(rm(max_position_value=6_000), BUY(20), held=50)
    assert not d.approved and d.rule == "max_position"


def test_gross_exposure_reduces_then_rejects() -> None:
    r = rm(max_gross_exposure=0.1, max_net_exposure=0.1)  # $10k room on $100k equity
    d = check(r, BUY(80), held=50)  # $5k already held -> room for 50 more shares
    assert d.approved and d.quantity == 50 and d.rule == "exposure"
    d = check(r, BUY(10), held=100)
    assert not d.approved


def test_order_rate_limit_uses_a_sliding_minute() -> None:
    r = rm(max_orders_per_minute=3)
    for i in range(3):
        assert check(r, BUY(1), now=i * NS_PER_S).approved
    d = check(r, BUY(1), now=10 * NS_PER_S)
    assert not d.approved and d.rule == "order_rate"
    assert check(r, BUY(1), now=61 * NS_PER_S).approved
    assert r.rejections["order_rate"] == 1


def test_sells_reduce_risk_but_are_capped_at_holdings() -> None:
    r = rm(max_position_shares=10, max_order_value=1e6)  # position limit does not block exits
    d = check(r, SELL(500), held=300)
    assert d.approved and d.quantity == 300


def test_daily_loss_and_drawdown_halt_the_account() -> None:
    r = rm(max_daily_loss=0.01, max_drawdown=0.5)
    assert r.on_equity(99_500) is None
    assert "daily loss" in r.on_equity(98_999)
    assert not check(r, SELL(1), held=5).approved  # halted: nothing passes
    r2 = rm(max_daily_loss=0.5, max_drawdown=0.01)
    r2.on_equity(110_000)
    assert "drawdown" in r2.on_equity(108_800)


# ------------------------------------------------------------------- config


@pytest.mark.parametrize("field", sorted(RISK))
def test_every_risk_field_is_required(field: str) -> None:
    with pytest.raises(ValidationError):
        risk_config(**{field: None}) if False else __import__("atlas.paper.config", fromlist=["RiskConfig"]) \
            .RiskConfig.model_validate({k: v for k, v in RISK.items() if k != field})


@pytest.mark.parametrize("bad", [{"max_daily_loss": 0}, {"max_gross_exposure": 1.5}, {"max_order_shares": -1},
                                 {"allowed_symbols": []}, {"allowed_symbols": ["SPY", "spy"]}, {"unknown": 1}])
def test_invalid_risk_values_fail_closed(bad) -> None:
    with pytest.raises(ValidationError):
        risk_config(**bad)


def test_risk_limits_are_immutable() -> None:
    cfg = risk_config()
    with pytest.raises(ValidationError):
        cfg.max_order_shares = 10_000  # type: ignore[misc]


# ----------------------------------------------------------- session-level controls


def test_daily_loss_breach_halts_account_and_stops_its_orders(tmp_path: Path) -> None:
    s = session(tmp_path, risk=risk_config(max_daily_loss=0.0001))
    report = s.run()
    champ = next(a for a in report["accounts"] if a["account"] == "champion")
    assert champ["halted"] and "daily loss" in champ["halted"]
    halts = [r for r in audit(s, "risk_halt") if r["account"] == "champion"]
    assert halts
    later = [r for r in audit(s, "order") if r["account"] == "champion" and r["ts"] > halts[0]["ts"]]
    assert later == []


def test_stale_feed_enters_safe_state_and_blocks_orders(tmp_path: Path) -> None:
    s = session(tmp_path, faults={"stale_gaps": [[2000, 10_000]]})
    report = s.run()
    periods = [p for p in report["safe_state_periods"] if p["reason"] == "stale or missing data"]
    assert periods and max(p["duration_ms"] for p in periods) > 2_000
    for p in periods:
        assert not [r for r in audit(s, "order") if p["start"] < r["ts"] < p["end"]]


def test_clock_anomalies_enter_safe_state_until_ordering_recovers(tmp_path: Path) -> None:
    s = session(tmp_path, faults={"clock_backwards_at": [1500], "future_timestamp_at": [3000]})
    report = s.run()
    assert report["counters"]["clock_anomalies"] == 2
    clock = [p for p in report["safe_state_periods"] if p["reason"] == "clock anomaly"]
    assert len(clock) == 2
    assert len(audit(s, "clock_anomaly")) == 2


def test_disconnect_reconnects_within_bounded_retries(tmp_path: Path) -> None:
    s = session(tmp_path, faults={"disconnect_at_events": [1000], "disconnect_failures": 2})
    report = s.run()
    assert report["counters"]["disconnects"] == 1 and report["counters"]["reconnect_attempts"] == 3
    assert report["kill_switch"] is None
    assert any(p["reason"] == "feed disconnected" for p in report["safe_state_periods"])


def test_exhausted_reconnects_engage_the_kill_switch(tmp_path: Path) -> None:
    s = session(tmp_path, faults={"disconnect_at_events": [1000], "disconnect_failures": 99})
    report = s.run()
    assert "feed lost" in report["kill_switch"]
    assert report["counters"]["reconnect_attempts"] == RISK["reconnect_max_retries"]
    assert report["counters"]["events"] < 1_100  # session stopped at the disconnect
    assert (s.dir / KillSwitch.FILE).exists()


class _KillAfter(ReplayLiveFeed):
    def __init__(self, ref, n: int, session_dir: Path) -> None:
        super().__init__(ref)
        self.n, self.session_dir = n, session_dir

    def next_event(self):
        if self.position == self.n:
            request_kill(self.session_dir, "operator drill")
        return super().next_event()


def test_manual_kill_switch_stops_actions_and_survives_restart(tmp_path: Path) -> None:
    probe = session(tmp_path)
    probe.health_checks()
    ref = probe.feed.ref
    s = session(tmp_path / "k", feed_override=_KillAfter(ref, 2_000, tmp_path / "k" / "paper" / "test_session"))
    report = s.run()
    assert report["kill_switch"].startswith("manual: operator drill")
    kill = audit(s, "kill_switch")[0]
    assert not [r for r in audit(s, "order") if r["ts"] > kill["ts"]]
    with pytest.raises(HealthCheckError, match="kill switch engaged"):
        session(tmp_path / "k", resume=True).health_checks()
    session(tmp_path / "k", resume=True, reset_kill_switch=True).health_checks()


def test_feed_symbols_must_be_allowed(tmp_path: Path) -> None:
    with pytest.raises(HealthCheckError, match="allowed_symbols"):
        session(tmp_path, risk=risk_config(allowed_symbols=["SPY"])).health_checks()


def test_model_mismatch_refuses_to_start(tmp_path: Path) -> None:
    import json

    bad = tmp_path / "ckpt.json"
    bad.write_text(json.dumps({"population_version": "v1-non-rl", "archives": {"champions": [
        {"record": {"agent_id": "x", "family": "momentum", "generation": 0, "genome": {"size_frac": 0.2}}}]}}),
        encoding="utf-8")
    with pytest.raises(HealthCheckError, match="model mismatch"):
        session(tmp_path, candidate={"source": "checkpoint", "path": str(bad)}).health_checks()
    bad.write_text(json.dumps({"population_version": "v9-other", "archives": {"champions": [{"record": {}}]}}),
                   encoding="utf-8")
    with pytest.raises(HealthCheckError, match="model mismatch"):
        session(tmp_path / "b", candidate={"source": "checkpoint", "path": str(bad)}).health_checks()
