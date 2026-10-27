"""Render walk-forward (and holdout) JSON results as a readable Markdown report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _bps(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 1e4:+.2f}"


def _num(x: float | None, fmt: str = ".2f") -> str:
    return "n/a" if x is None else format(x, fmt)


def _ci(ci: dict[str, Any]) -> str:
    if ci.get("mean") is None:
        return "n/a"
    return f"{_bps(ci['mean'])} [{_bps(ci['low'])}, {_bps(ci['high'])}] (n={ci['n']})"


def render(report: dict[str, Any], holdout: dict[str, Any] | None = None) -> str:
    s = report["summary"]
    lines = [
        f"# Walk-forward report: {report['experiment_name']}",
        "",
        f"Run `{report.get('run_id', '')}` · design `{report['design_fingerprint']}` · dataset `{report['dataset_id']}`",
        "",
        f"> {report['disclaimer']}",
        "",
        "## Out-of-sample result",
        "",
        "Mean per-session net return of each fold's top champion on that fold's validation sessions, in bps, "
        "with a bootstrap confidence interval. Base and pessimistic costs are the main evidence.",
        "",
        "| Scenario | Champion | Sessions positive | Cash | Buy & hold | Random control |",
        "|---|---|---|---|---|---|",
    ]
    bl = s["baselines_oos"]
    for sc, ci in s["champion_oos"].items():
        mark = " (emphasized)" if sc in report["emphasized_scenarios"] else ""
        pos = s["champion_positive_session_fraction"][sc]
        lines.append(
            f"| {sc}{mark} | {_ci(ci)} | {_num(pos, '.0%')} | {_bps(bl.get('baseline-cash', {}).get(sc, {}).get('mean'))} "
            f"| {_bps(bl.get('baseline-buy_hold', {}).get(sc, {}).get('mean'))} "
            f"| {_bps(bl.get('baseline-random', {}).get(sc, {}).get('mean'))} |"
        )
    gap = s["selection_gap"]
    lines += [
        "",
        "## Selection bias",
        "",
        f"* Mean in-sample return of fold champions: **{_bps(gap['mean_in_sample_return'])} bps**",
        f"* Mean out-of-sample (base) return: **{_bps(gap['mean_oos_base_return'])} bps**",
        f"* Gap: **{_bps(gap['mean_gap'])} bps** — {gap['note']}.",
        f"* Selection trials: {report['selection_trials']['agent_session_evaluations']} agent-session evaluations "
        f"across {report['selection_trials']['folds']} folds.",
        "",
        "## Folds",
        "",
        "| Fold | Train | Validation | Champion | Family | In-sample | OOS base | OOS pessimistic | Fills (base) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for f in report["folds"]:
        if not f["candidates"]:
            lines.append(f"| {f['index']} | {f['train'][0]}…{f['train'][-1]} | {f['validation'][0]}…"
                         f"{f['validation'][-1]} | none eligible | | | | | |")
            continue
        c = f["candidates"][0]
        lines.append(
            f"| {f['index']} | {f['train'][0]}…{f['train'][-1]} | {f['validation'][0]}…{f['validation'][-1]} "
            f"| {c['agent_id']} | {c['family']} | {_bps(c['in_sample']['mean_net_return'])} "
            f"| {_bps(c['out_of_sample']['base']['mean_net_return'])} "
            f"| {_bps(c['out_of_sample']['pessimistic']['mean_net_return'])} "
            f"| {c['out_of_sample']['base']['fills']} |"
        )
    stab = s["stability"]
    lines += ["", "## Stability", "", "By volatility regime (base, champion):", ""]
    for k, (mean, n) in stab["by_volatility_regime"].items():
        lines.append(f"* {k}: {_bps(mean)} bps over {n} sessions")
    lines += ["", "By symbol (base, champion, summed P&L): " +
              ", ".join(f"{k} {v:+.2f}" for k, v in stab["by_symbol_base_pnl"].items()), ""]
    lines += ["## Champion detail (base scenario)", "",
              "| Fold | Sharpe | Sortino | Profit factor | Win rate | Fill rate | Avg exposure | p50 / p99 decision µs |",
              "|---|---|---|---|---|---|---|---|"]
    for f in report["folds"]:
        if f["candidates"]:
            c, b = f["candidates"][0], f["candidates"][0]["out_of_sample"]["base"]
            lat = c["latency"]
            lines.append(f"| {f['index']} | {_num(b['sharpe'])} | {_num(b['sortino'])} | {_num(b['profit_factor'])} "
                         f"| {_num(b['win_rate'], '.0%')} | {_num(b['fill_rate'], '.0%')} "
                         f"| {_num(b['avg_exposure'], '.1%')} | {_num(lat['p50_us'], '.0f')} / "
                         f"{_num(lat['p99_us'], '.0f')} |")
    dq = report["data_quality"]
    dropped = sum(d["dropped_events"] for d in dq)
    lines += ["", "## Data", "",
              f"* Source: {report['data_source'].get('provider')} — {report['data_source'].get('fidelity')}",
              f"* License: {report['data_source'].get('license_notes')}",
              f"* Development sessions: {len(report['plan']['development'])}; events dropped by validation: {dropped}",
              f"* Outer holdout: {len(report['holdout']['sessions'])} sessions "
              f"({report['holdout']['sessions'][0]}…{report['holdout']['sessions'][-1]}) — "
              f"{report['holdout']['status']}", ""]
    if holdout:
        lines += _render_holdout(holdout)
    lines += ["## Limitations", ""] + [f"* {x}" for x in report["limitations"]] + [""]
    return "\n".join(lines)


def _render_holdout(h: dict[str, Any]) -> list[str]:
    c = h["candidate"]["agent_id"]
    lines = ["## Outer holdout (frozen design)", ""]
    if h["holdout_reused"]:
        lines += ["> **Warning:** this holdout has been accessed by more than one design. "
                  "It is no longer an untouched test set.", ""]
    lines += [f"Candidate `{c}` ({h['candidate']['family']}) on {len(h['holdout_sessions'])} sessions. "
              f"Total holdout accesses recorded: {h['holdout_accesses_total']}.", "",
              "| Scenario | Candidate (bps, CI) | Cash | Buy & hold | Random control |", "|---|---|---|---|---|"]
    for sc, ci in h["champion_ci"].items():
        r = h["results"][sc]
        lines.append(f"| {sc} | {_ci(ci)} | {_bps(r['baseline-cash']['mean_net_return'])} "
                     f"| {_bps(r['baseline-buy_hold']['mean_net_return'])} "
                     f"| {_bps(r['baseline-random']['mean_net_return'])} |")
    return lines + [""]


def write_markdown(run_dir: Path) -> Path:
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    holdout_path = run_dir / "holdout_report.json"
    holdout = json.loads(holdout_path.read_text(encoding="utf-8")) if holdout_path.is_file() else None
    out = run_dir / "report.md"
    out.write_text(render(report, holdout), encoding="utf-8")
    return out
