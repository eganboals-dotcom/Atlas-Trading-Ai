"""Atlas dashboard (read-only). Run with:

    streamlit run src/atlas/dashboard/app.py -- --paper-dir artifacts/paper --reports-dir artifacts/reports/walkforward

It shows paper sessions (accounts, P&L, costs, risk events, fills, latency, alerts) and
walk-forward results. It has no controls that affect trading: it cannot submit or
cancel orders, change risk limits, or engage or reset the kill switch. Use the
`paper-kill` command for that.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import plotly.express as px
import polars as pl
import streamlit as st

from atlas.dashboard import data


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--paper-dir", default="artifacts/paper")
    p.add_argument("--reports-dir", default="artifacts/reports/walkforward")
    return p.parse_known_args(sys.argv[1:])[0]  # ignore arguments meant for the host process


def _bps(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 1e4:+.2f} bps"


def paper_view(paper_dir: Path) -> None:
    sessions = data.list_sessions(paper_dir)
    if not sessions:
        st.info(f"No paper sessions in {paper_dir}. Run `python -m atlas.cli paper --config configs/paper.yaml`.")
        return
    session = Path(st.selectbox("Session", sessions, format_func=lambda p: Path(p).name))
    report, state = data.load_report(session), data.load_state(session) or {}
    st.caption("Simulated orders and fills only. Nothing here is an actual execution.")
    if state.get("kill"):
        st.error(f"Kill switch engaged: {state['kill']}")
    if state.get("safe"):
        st.warning(f"Safe state: {', '.join(state['safe'])}")

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
        st.info("Session in progress or ended without a report; showing live state.")

    eq = data.equity_frame(session)
    if eq.height:
        st.subheader("Equity")
        st.plotly_chart(px.line(eq.to_pandas(), x="time", y="equity", color="account"), width="stretch")

    st.subheader("Risk and operational events")
    st.dataframe(pl.DataFrame(data.risk_events(session)[-200:]) if data.risk_events(session) else pl.DataFrame(),
                 hide_index=True)
    st.subheader("Recent simulated fills")
    f = data.fills(session)[-200:]
    st.dataframe(pl.DataFrame([{k: r.get(k) for k in ("account", "ts", "order_id", "fill_qty", "fill_price", "fee",
                                                      "venue_bid", "venue_ask", "status")} for r in f])
                 if f else pl.DataFrame(), hide_index=True)
    st.subheader("Alerts")
    al = data.alerts(session)
    st.dataframe(pl.DataFrame(al) if al else pl.DataFrame(), hide_index=True)


def research_view(reports_dir: Path) -> None:
    runs = data.list_walkforward_runs(reports_dir)
    if not runs:
        st.info(f"No walk-forward runs in {reports_dir}.")
        return
    run = Path(st.selectbox("Walk-forward run", runs, format_func=lambda p: Path(p).name))
    report = data.load_json(run / "report.json") or {}
    md = run / "report.md"
    if md.is_file():
        st.markdown(md.read_text(encoding="utf-8"))
    else:
        st.json(report.get("summary", {}))


def main() -> None:
    args = _args()
    st.set_page_config(page_title="Atlas dashboard", layout="wide")
    st.title("Project Atlas")
    paper_tab, research_tab = st.tabs(["Paper trading", "Research"])
    with paper_tab:
        paper_view(Path(args.paper_dir))
    with research_tab:
        research_view(Path(args.reports_dir))


main()
