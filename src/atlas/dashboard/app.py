"""Atlas dashboard (read-only). Opens automatically with `paper --open-dashboard`, or run:

    streamlit run src/atlas/dashboard/app.py -- --paper-dir artifacts/paper --reports-dir artifacts/reports/walkforward

Tabs: **Live** (leaderboard, each bot's P&L, equity, positions, and trades, refreshed every
2 s from the running session's live.json), **Paper sessions** (finished daily reports),
and **Research** (walk-forward reports). It has no controls that affect trading: it
cannot submit or cancel orders, change risk limits, or engage or reset the kill switch.
Use the `paper-kill` command for that.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import plotly.express as px
import polars as pl
import streamlit as st

from atlas.dashboard import data

ET = "America/New_York"


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--paper-dir", default="artifacts/paper")
    p.add_argument("--reports-dir", default="artifacts/reports/walkforward")
    p.add_argument("--session", help="session to show on the Live tab (default: most recently updated)")
    return p.parse_known_args(sys.argv[1:])[0]  # ignore arguments meant for the host process


def _bps(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 1e4:+.2f} bps"


def _et(ns: int) -> str:
    return pl.Series([ns]).cast(pl.Datetime("ns", "UTC")).dt.convert_time_zone(ET).dt.strftime("%H:%M:%S")[0]


# --------------------------------------------------------------------- live


def live_view(paper_dir: Path, preferred: str | None) -> None:
    sessions = data.list_live_sessions(paper_dir)
    if not sessions:
        st.info(f"No running or recent session in {paper_dir}. Start one with "
                "`python -m atlas.cli paper --config configs/paper_live.yaml --open-dashboard`.")
        return
    names = [s.name for s in sessions]
    default = names.index(preferred) if preferred in names else 0
    session = paper_dir / st.selectbox("Session", names, index=default, key="live_session")

    @st.fragment(run_every="2s")
    def board() -> None:
        live = data.load_live(session)
        if not live:
            st.caption("Waiting for the session's first snapshot…")
            return
        updated = datetime.fromisoformat(live["updated_wall"]).astimezone(timezone.utc)
        age = (datetime.now(timezone.utc) - updated).total_seconds()
        status = live["status"]
        if live.get("kill"):
            st.error(f"Kill switch engaged: {live['kill']}")
        elif live.get("safe"):
            st.warning(f"Safe state (no new orders): {', '.join(live['safe'])}")
        elif status == "running" and age > 15:
            st.warning(f"No update for {age:.0f} s. The session may be waiting for data or has stopped.")
        st.caption(f"{live['session']} · {live['session_date']} · feed **{live['feed']}** · {', '.join(live['symbols'])}"
                   f" · status **{status}** · updated {age:.0f} s ago · simulated orders and fills only")

        rows = data.leaderboard(live)
        top = rows[0] if rows and rows[0]["rank"] else None
        c = st.columns(4)
        if top:
            c[0].metric("Leader", f"{top['account']}", f"{top['net_pnl']:+,.2f} USD")
        c[1].metric("Bots", sum(1 for r in rows if r["rank"]))
        c[2].metric("Simulated fills", sum(r["fills"] for r in rows))
        c[3].metric("Events", live["counters"]["events"])

        st.subheader("Leaderboard")
        table = pl.DataFrame([{
            "#": r["rank"] if r["rank"] else "—", "bot": r["account"], "model": f"{r['agent_id']} ({r['family']})",
            "net P&L $": round(r["net_pnl"], 2), "return": _bps(r["net_return"]),
            "vs buy & hold": _bps(r["vs_benchmark"]) if r["rank"] else "—",
            "status": ("HALTED: " + r["halted"]) if r["halted"] else ("order working" if r["working_orders"] else "ok"),
            "fills": r["fills"],
            "positions": ", ".join(f"{s} {q}" for s, q in r["positions"].items()) or "flat",
            "realized $": round(r["realized_pnl"], 2), "unrealized $": round(r["unrealized_pnl"], 2),
            "fees $": round(r["fees"], 2),
        } for r in rows])
        st.dataframe(table, hide_index=True, width="stretch")

        curves = [{"bot": a["account"], "ts": ts, "P&L $": eq - (a["equity"] - a["net_pnl"])}  # equity - start
                  for a in live["accounts"] for ts, eq in a["equity_curve"]]
        if curves:
            df = pl.DataFrame(curves).sort("ts").with_columns(
                pl.col("ts").cast(pl.Datetime("ns", "UTC")).dt.convert_time_zone(ET).alias("time (ET)"))
            st.subheader("P&L over time")
            st.plotly_chart(px.line(df.to_pandas(), x="time (ET)", y="P&L $", color="bot"), width="stretch")

        st.subheader("Recent simulated trades")
        trades = live["recent_trades"]
        if trades:
            st.dataframe(pl.DataFrame([{"time (ET)": _et(t["ts"]), "bot": t["account"], "symbol": t["symbol"],
                                        "side": t["side"], "qty": t["qty"], "price": round(t["price"], 4),
                                        "fee $": round(t["fee"], 4)} for t in trades]),
                         hide_index=True, width="stretch")
        else:
            st.caption("No trades yet.")

        bot = st.selectbox("Bot detail", [a["account"] for a in live["accounts"]], key="bot_detail")
        a = next(x for x in live["accounts"] if x["account"] == bot)
        d = st.columns(5)
        d[0].metric("Equity", f"{a['equity']:,.2f}")
        d[1].metric("Cash", f"{a['cash']:,.2f}")
        d[2].metric("Net P&L", f"{a['net_pnl']:+,.2f}")
        d[3].metric("Fills", a["fills"])
        d[4].metric("Risk rejections", a["risk_rejections"])
        mine = [t for t in trades if t["account"] == bot]
        if mine:
            st.dataframe(pl.DataFrame([{"time (ET)": _et(t["ts"]), "symbol": t["symbol"], "side": t["side"],
                                        "qty": t["qty"], "price": round(t["price"], 4)} for t in mine]),
                         hide_index=True, width="stretch")

    board()


# ----------------------------------------------------------------- history


def paper_view(paper_dir: Path) -> None:
    sessions = data.list_sessions(paper_dir)
    if not sessions:
        st.info(f"No paper sessions in {paper_dir}.")
        return
    session = Path(st.selectbox("Session", sessions, format_func=lambda p: Path(p).name, key="paper_session"))
    report, state = data.load_report(session), data.load_state(session) or {}
    st.caption("Simulated orders and fills only. Nothing here is an actual execution.")
    if state.get("kill"):
        st.error(f"Kill switch engaged: {state['kill']}")
    if report:
        rows = [{"account": a["account"], "model": f"{a['model']['agent_id']} ({a['model']['family']})",
                 "net P&L": round(a["net_pnl"], 2), "return": _bps(a["net_return"]),
                 "vs benchmark": _bps(a["excess_return_vs_benchmark"]), "max DD": _bps(a["max_drawdown"]),
                 "turnover": round(a["turnover"], 2), "fills": a["fills"], "fees": round(a["fees"], 2),
                 "risk rejections": sum(a["risk_rejections"].values()), "halted": a["halted"] or "",
                 "p99 decision µs": a["decision_latency"]["p99_us"]} for a in report["accounts"]]
        st.subheader("Accounts")
        st.dataframe(pl.DataFrame(rows), hide_index=True)
        c = report["counters"]
        cols = st.columns(5)
        cols[0].metric("Events", c["events"])
        cols[1].metric("Dropped", c["dropped_events"])
        cols[2].metric("Disconnects", c["disconnects"])
        cols[3].metric("Clock anomalies", c["clock_anomalies"])
        cols[4].metric("Alerts", report["alerts"])
    else:
        st.info("Session in progress or ended without a report; see the Live tab.")

    eq = data.equity_frame(session)
    if eq.height:
        st.subheader("Equity")
        st.plotly_chart(px.line(eq.to_pandas(), x="time", y="equity", color="account"), width="stretch")
    st.subheader("Risk and operational events")
    events = data.risk_events(session)
    st.dataframe(pl.DataFrame(events[-200:]) if events else pl.DataFrame(), hide_index=True)
    st.subheader("Alerts")
    al = data.alerts(session)
    st.dataframe(pl.DataFrame(al) if al else pl.DataFrame(), hide_index=True)


def research_view(reports_dir: Path) -> None:
    runs = data.list_walkforward_runs(reports_dir)
    if not runs:
        st.info(f"No walk-forward runs in {reports_dir}.")
        return
    run = Path(st.selectbox("Walk-forward run", runs, format_func=lambda p: Path(p).name, key="wf_run"))
    md = run / "report.md"
    if md.is_file():
        st.markdown(md.read_text(encoding="utf-8"))


def main() -> None:
    args = _args()
    st.set_page_config(page_title="Atlas live paper trading", layout="wide")
    st.title("Project Atlas — paper trading")
    live_tab, paper_tab, research_tab = st.tabs(["Live", "Paper sessions", "Research"])
    with live_tab:
        live_view(Path(args.paper_dir), args.session)
    with paper_tab:
        paper_view(Path(args.paper_dir))
    with research_tab:
        research_view(Path(args.reports_dir))


main()
