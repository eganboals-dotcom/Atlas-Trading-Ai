"""Read-only loaders for paper sessions and walk-forward runs.

The dashboard only reads files written by other processes. This module imports nothing
that can submit, cancel, or change orders, risk limits, or the kill switch (a test
enforces this).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl


def list_sessions(paper_dir: Path) -> list[Path]:
    if not paper_dir.is_dir():
        return []
    return sorted((p for p in paper_dir.iterdir() if (p / "state.json").is_file()), reverse=True)


def load_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def load_state(session_dir: Path) -> dict[str, Any] | None:
    return load_json(session_dir / "state.json")


def load_live(session_dir: Path) -> dict[str, Any] | None:
    """The running session's frequently refreshed snapshot (``live.json``)."""
    try:
        return load_json(session_dir / "live.json")
    except (json.JSONDecodeError, OSError):  # mid-replace on some filesystems; next refresh succeeds
        return None


def leaderboard(live: dict[str, Any]) -> list[dict[str, Any]]:
    """Bots ranked by net P&L (best first). The benchmark is shown but not ranked."""
    bots = sorted((a for a in live["accounts"] if a["role"] != "benchmark"), key=lambda a: -a["net_pnl"])
    rows = [{"rank": i + 1, **a} for i, a in enumerate(bots)]
    rows += [{"rank": None, **a} for a in live["accounts"] if a["role"] == "benchmark"]
    return rows


def list_live_sessions(paper_dir: Path) -> list[Path]:
    if not paper_dir.is_dir():
        return []
    return sorted((p for p in paper_dir.iterdir() if (p / "live.json").is_file()),
                  key=lambda p: (p / "live.json").stat().st_mtime, reverse=True)


def load_report(session_dir: Path) -> dict[str, Any] | None:
    return load_json(session_dir / "daily_report.json")


def equity_frame(session_dir: Path) -> pl.DataFrame:
    """Equity per account over time. Uses the live state file while a session runs."""
    path = session_dir / "equity.csv"
    if path.is_file():
        df = pl.read_csv(path)
    else:
        state = load_state(session_dir) or {"accounts": {}}
        rows = [{"account": a, "ts_ns": ts, "equity": eq}
                for a, s in state["accounts"].items() for ts, eq in s["equity_curve"]]
        df = pl.DataFrame(rows, schema={"account": pl.Utf8, "ts_ns": pl.Int64, "equity": pl.Float64})
    return df.with_columns(pl.from_epoch("ts_ns", time_unit="ns").dt.replace_time_zone("UTC").alias("time"))


def read_jsonl(path: Path, types: set[str] | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if types is None or rec.get("type") in types:
                out.append(rec)
    return out[-limit:] if limit else out


def fills(session_dir: Path) -> list[dict[str, Any]]:
    return [r for r in read_jsonl(session_dir / "audit.jsonl", {"execution"}) if r.get("fill_qty")]


def risk_events(session_dir: Path) -> list[dict[str, Any]]:
    recs = read_jsonl(session_dir / "audit.jsonl", {"risk_check", "risk_halt", "kill_switch", "clock_anomaly",
                                                     "disconnect", "reconnected", "reconnect_failed"})
    return [r for r in recs if r.get("type") != "risk_check" or not r.get("approved") or r.get("rule")]


def alerts(session_dir: Path) -> list[dict[str, Any]]:
    return read_jsonl(session_dir / "alerts.jsonl")


def list_walkforward_runs(reports_dir: Path) -> list[Path]:
    if not reports_dir.is_dir():
        return []
    return sorted((p for p in reports_dir.iterdir() if (p / "report.json").is_file()), reverse=True)
