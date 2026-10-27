"""Daily paper-trading report: per-account P&L against the benchmark, risk, execution, and data health.

Every figure is based on simulated fills from the internal paper broker. None of it is
an actual execution.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from atlas import __version__
from atlas.evaluation.metrics import latency_percentiles, max_drawdown

if TYPE_CHECKING:
    from atlas.paper.session import PaperSession

DISCLAIMER = ("Paper trading: live-style market data, simulated orders and fills only. No order was sent to any "
              "broker. Simulated fills are assumptions, not executions; results are not evidence of live performance.")


def account_summary(a) -> dict[str, Any]:
    p = a.broker.portfolio
    curve = [e for _, e in a.equity_curve] or [p.initial_cash]
    statuses: dict[str, int] = {}
    for o in a.broker.orders:
        statuses[str(o.status)] = statuses.get(str(o.status), 0) + 1
    return {
        "account": a.account_id,
        "role": a.role,
        "model": a.model,
        "net_pnl": p.equity - p.initial_cash,
        "net_return": p.equity / p.initial_cash - 1,
        "max_drawdown": max_drawdown([p.initial_cash, *curve]),
        "turnover": sum(f.quantity * f.price for f in p.fills) / p.initial_cash,
        "fills": len(p.fills),
        "fees": p.fees_paid,
        "orders_by_status": statuses,
        "broker_rejects": statuses.get("rejected", 0),
        "risk_rejections": a.risk.rejections,
        "halted": a.risk.halted,
        "open_positions": {s: q.quantity for s, q in p.positions.items() if q.quantity},
        "decision_latency": latency_percentiles(list(a.decision_wall_ns)),
        "simulated": True,
    }


def write_daily_report(session: PaperSession) -> dict[str, Any]:
    accounts = [account_summary(a) for a in session.accounts]
    bench = next(x for x in accounts if x["role"] == "benchmark")
    for x in accounts:
        x["excess_return_vs_benchmark"] = x["net_return"] - bench["net_return"]
    report = {
        "session": session.cfg.session_name,
        "session_date": session.session_date.isoformat(),
        "atlas_version": __version__,
        "config_fingerprint": session.fingerprint,
        "mode": "paper",
        "live_orders_enabled": False,
        "feed": {"kind": session.cfg.feed.kind, "symbols": session.symbols, "speed": session.cfg.feed.speed},
        "counters": session.counters,
        "validation_issues": session.validation_counts,
        "safe_state_periods": session.safe_periods,
        "kill_switch": session.kill.reason,
        "stopped_by": getattr(session, "stopped_by", None),
        "alerts": session.alerts.count,
        "restarts": getattr(session, "restarts", 0),
        "accounts": accounts,
        "fund": None if session.fund is None else session.fund.data,
        "disclaimer": DISCLAIMER,
    }
    (session.dir / "daily_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (session.dir / "daily_report.md").write_text(render(report), encoding="utf-8")
    with (session.dir / "equity.csv").open("w", encoding="utf-8") as fh:
        fh.write("account,ts_ns,equity\n")
        for a in session.accounts:
            for ts, eq in a.equity_curve:
                fh.write(f"{a.account_id},{ts},{eq}\n")
    return report


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 1e4:+.2f} bps"


def render(r: dict[str, Any]) -> str:
    c = r["counters"]
    lines = [
        f"# Paper session {r['session']} ({r['session_date']})", "",
        f"> {r['disclaimer']}", "",
        f"Mode `{r['mode']}` · live orders enabled: **{r['live_orders_enabled']}** · atlas {r['atlas_version']} · "
        f"config `{r['config_fingerprint']}`", "",
        "## Accounts", "",
        "| Account | Model | Net P&L | Return | vs benchmark | Max DD | Turnover | Fills | Rejects (broker / risk) "
        "| Halted | p50/p99 decision µs |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for a in r["accounts"]:
        lat = a["decision_latency"]
        risk = sum(a["risk_rejections"].values())
        lines.append(
            f"| {a['account']} | {a['model']['agent_id']} ({a['model']['family']}) | {a['net_pnl']:+.2f} "
            f"| {_pct(a['net_return'])} | {_pct(a['excess_return_vs_benchmark'])} | {_pct(a['max_drawdown'])} "
            f"| {a['turnover']:.2f} | {a['fills']} | {a['broker_rejects']} / {risk} | {a['halted'] or 'no'} "
            f"| {lat['p50_us'] if lat['p50_us'] is None else round(lat['p50_us'])}/"
            f"{lat['p99_us'] if lat['p99_us'] is None else round(lat['p99_us'])} |"
        )
    lines += [
        "", "## Data and operations", "",
        f"* Events processed: {c['events']}; dropped by validation: {c['dropped_events']} "
        f"({', '.join(f'{k}: {v}' for k, v in r['validation_issues'].items()) or 'no issues'})",
        f"* Feed gaps (quote silence beyond the stale limit): {c['feed_gaps']}",
        f"* Disconnects: {c['disconnects']} (reconnect attempts: {c['reconnect_attempts']}); clock anomalies: "
        f"{c['clock_anomalies']}; restarts: {r['restarts']}",
        f"* Safe-state periods: {len(r['safe_state_periods'])}"
        + "".join(f"\n  * {p['reason']}: {p['duration_ms']:.0f} ms" for p in r["safe_state_periods"][:20]),
        f"* Kill switch: {r['kill_switch'] or 'not engaged'}",
        f"* Alerts raised: {r['alerts']}", "",
        "## Models", "",
    ]
    for a in r["accounts"]:
        m = a["model"]
        lines.append(f"* {a['account']}: `{m['agent_id']}` family `{m['family']}`, genome/weights `{m['genome_hash']}`, "
                     f"population `{m['population_version']}`")
    return "\n".join(lines) + "\n"
