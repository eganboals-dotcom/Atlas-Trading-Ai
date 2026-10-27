"""Builders for paper-session tests: small synthetic sessions with baseline candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from helpers import sim_config

from atlas.paper.config import PaperConfig, RiskConfig
from atlas.paper.session import PaperSession

REPO = Path(__file__).resolve().parents[1]

RISK: dict[str, Any] = {
    "allowed_symbols": ["SPY", "QQQ"], "max_position_shares": 1000, "max_position_value": 50_000,
    "max_order_shares": 500, "max_order_value": 25_000, "max_gross_exposure": 0.8, "max_net_exposure": 0.8,
    "max_daily_loss": 0.01, "max_drawdown": 0.015, "max_orders_per_minute": 30, "stale_data_ms": 2_000,
    "max_clock_skew_ms": 1_000, "clock_recovery_events": 20, "reconnect_max_retries": 3,
    "reconnect_backoff_ms": 0,
}


def risk_config(**kw: Any) -> RiskConfig:
    return RiskConfig.model_validate({**RISK, **kw})


def paper_dict(tmp: Path, **kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "mode": "paper", "live_orders_enabled": False, "session_name": "test_session",
        "simulator_config": "unused.yaml", "risk_config": "unused.yaml",
        "feed": {
            "kind": "replay",
            "data_source": {"kind": "synthetic", "data_config": str(REPO / "configs/data.yaml"),
                            "start_date": "2026-06-01", "n_sessions": 1, "duration_s": 300,
                            "symbols": ["SPY", "QQQ"]},
            "session_date": "2026-06-01", "speed": 0, "first_quote_timeout_ms": 5_000,
        },
        "candidate": {"source": "baseline", "baseline": "baseline_momentum"},
        "shadows": [{"source": "baseline", "baseline": "baseline_random"}],
        "agent_limits": {"max_position_shares": 1000, "max_order_shares": 500, "max_position_frac": 0.5},
        "decision_interval_ms": 500,
        "alerts": {"console": False, "file": True},
        "output_dir": str(tmp / "paper"),
        "state_flush_events": 500,
    }
    for k, v in kw.items():
        if k == "faults":
            base["feed"]["faults"] = v
        elif isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return base


def paper_config(tmp: Path, **kw: Any) -> PaperConfig:
    return PaperConfig.model_validate(paper_dict(tmp, **kw))


def session(tmp: Path, risk: RiskConfig | None = None, resume: bool = False, feed_override=None,
            reset_kill_switch: bool = False, **kw: Any) -> PaperSession:
    return PaperSession(paper_config(tmp, **kw), sim_config(stale_quote_ms=2_000.0), risk or risk_config(),
                        base_dir=REPO, feed=feed_override, resume=resume, reset_kill_switch=reset_kill_switch)


def audit(s: PaperSession, *types: str) -> list[dict[str, Any]]:
    out = []
    for line in (s.dir / "audit.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if not types or rec.get("type") in types:
            out.append(rec)
    return out
